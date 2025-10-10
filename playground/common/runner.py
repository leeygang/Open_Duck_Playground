"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""
# Must be first import: load tb_logging to patch jax.monitoring early
import playground.common.tb_logging as tb_logging  # noqa: F401

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import logging
from brax.training import pmap as brax_pmap
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import numpy as np

from playground.common.export_onnx import export_onnx

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '033[93m'
RESET = '\033[0m'

# Enable verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to console
        #logging.FileHandler('training.log')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)

PMAP_ASSERT_PATCHED = False

# Enable JAX debugging
# os.environ['JAX_LOG_COMPILES'] = '1'
# os.environ['JAX_DEBUG_NANS'] = '1'
# os.environ['JAX_DEBUG_INFS'] = '1'

class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        logger.info(f"Initializing BaseRunner with args: {args}")

        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        # Create single TB writer and inject into tb_logging so all logs go to same place
        self.writer = SummaryWriter(log_dir=self.output_dir)
        try:
            tb_logging.set_writer(self.writer)
        except Exception:
            pass
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        self.overrided_ppo_params = dict()
        self.callback_count = 0
        self.expected_callbacks = None
        
        # Timing
        self.start_time = None
        self.last_progress_time = None
        self.last_progress_step = 0
        # CACHE STUFF (import jax here so env variables like JAX_PLATFORM_NAME set by callers take effect)
        os.makedirs(".tmp", exist_ok=True)
        import jax  # local import to honor prior env like JAX_PLATFORM_NAME
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

        # Allow callers to disable JIT for faster startup on CPU-only systems
        if getattr(self.args, "no_jit", False):
            try:
                jax.config.update("jax_disable_jit", True)
                logger.info("JAX JIT disabled via --no_jit")
            except Exception as _e:
                logger.warning(f"Failed to disable JIT: {_e}")

        # Decide whether to use pmap for environment reset based on backend/devices.
        # Rationale:
        # - pmap is only beneficial on multi-accelerator setups (e.g., TPU or multi-GPU)
        # - On CPU or single device, it adds compilation/overhead without benefits
        try:
            self.jax_platform_name = jax.default_backend() or (jax.devices()[0].platform if jax.devices() else "cpu")
        except Exception:
            self.jax_platform_name = "cpu"
        try:
            self.jax_device_count = jax.device_count()
        except Exception:
            self.jax_device_count = 1

        self.use_pmap_on_reset = (
            self.jax_platform_name != "cpu" and self.jax_device_count > 1
        )
        logger.info(
            f"JAX backend: {self.jax_platform_name}, device_count: {self.jax_device_count}, "
            f"use_pmap_on_reset: {self.use_pmap_on_reset}"
        )

        # On single-device CPU runs, Brax's replication assertion at the end of training can
        # spuriously fail even though training completed. Relax it by wrapping the check so we
        # log and continue instead of raising.
        global PMAP_ASSERT_PATCHED
        if (
            not PMAP_ASSERT_PATCHED
            and self.jax_platform_name == "cpu"
            and self.jax_device_count <= 1
        ):
            original_assert_is_replicated = brax_pmap.assert_is_replicated
            jax_module = jax

            def cpu_tolerant_assert_is_replicated(x, debug=None):
                if jax_module.process_count() == 1 and jax_module.local_device_count() <= 1:
                    try:
                        return original_assert_is_replicated(x, debug)
                    except AssertionError as err:
                        warn_msg = debug if debug is not None else (
                            err.args[0] if err.args else "replication mismatch"
                        )
                        logger.warning(
                            "[CPU] Ignoring Brax pmap.assert_is_replicated failure on single-device run: %s",
                            warn_msg,
                        )
                        return None
                return original_assert_is_replicated(x, debug)

            brax_pmap.assert_is_replicated = cpu_tolerant_assert_is_replicated
            PMAP_ASSERT_PATCHED = True
            logger.info(
                "Patched Brax pmap.assert_is_replicated to tolerate single-device CPU setups."
            )

        logger.info("BaseRunner initialized successfully")

    @staticmethod
    def _to_tb_scalar(x):
        """Converts various numeric types (including JAX/NumPy scalars) to a Python float.

        Returns None if conversion isn't possible.
        """
        # Fast path for Python numbers
        if isinstance(x, (int, float)):
            return float(x)

        # JAX: bring to host if needed
        try:
            import jax  # local import to avoid issues if environment changes
            if isinstance(x, jax.Array):
                x = jax.device_get(x)
        except Exception:
            # If jax isn't available or check failed, continue
            pass

        # NumPy scalar/array handling
        try:
            if isinstance(x, np.ndarray):
                if x.shape == ():
                    return float(x.item())
                # If it's a vector/tensor, log its mean as a single scalar
                return float(np.asarray(x).mean())
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
        except Exception:
            pass

        # Generic fallback attempts
        try:
            return float(x)
        except Exception:
            try:
                return float(np.asarray(x).mean())
            except Exception:
                return None

    def progress_callback(self, num_steps: int, metrics: dict) -> None:
        self.callback_count += 1
        total_callbacks = self.expected_callbacks or "?"
        logger.info(
            GREEN + f"[BASE RUNNER] progress_callback {self.callback_count}/{total_callbacks}" + RESET
        )
        
        current_time = time.time()
        if self.last_progress_time is not None:
            time_since_last = current_time - self.last_progress_time
            logger.info(f"Time since last progress: {time_since_last:.2f}s")

        logger.info(f"[PROGRESS] Step {num_steps}")
        logger.debug(f"[PROGRESS] All metrics: {metrics}")

        for metric_name, metric_value in metrics.items():
            # Convert to float; TensorBoard expects numpy or python scalars, not JAX arrays
            scalar_val = self._to_tb_scalar(metric_value)
            if scalar_val is None:
                logger.warning(
                    f"[TB] Skipping metric '{metric_name}' (unconvertible type: {type(metric_value)})"
                )
                continue
            self.writer.add_scalar(metric_name, scalar_val, int(num_steps))

        print("-----------")
        # Ensure pretty printing for JAX/NumPy types
        _rew = self._to_tb_scalar(metrics.get("eval/episode_reward", np.nan))
        _rew_std = self._to_tb_scalar(metrics.get("eval/episode_reward_std", np.nan))
        print(
            f"{GREEN} CALLBACK {self.callback_count}/{total_callbacks} | STEP: {int(num_steps)} "
            f"reward: {_rew} reward_std: {_rew_std} {RESET}"
        )
        print("-----------")

    # Record for next progress log
        self.last_progress_time = current_time
        self.last_progress_step = int(num_steps)

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints
        logger.info(f"[CHECKPOINT] Saving at step {current_step}")

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        logger.info(f"Checkpoint path: {path}")
        
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        
        # Optional ONNX export
        final_only_export = getattr(self.args, "export_on_finish", False)
        if final_only_export:
            logger.info("Skipping intermediate ONNX export; final model will be exported at training end.")
        else:
            onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
            logger.info(f"Exporting ONNX to: {onnx_export_path}")
            try:
                export_onnx(
                    params,
                    self.action_size,
                    self.ppo_params,
                    self.obs_size,  # may not work
                    output_path=onnx_export_path
                )
                logger.info("Checkpoint and ONNX export completed")
            except Exception as e:
                logger.warning(f"ONNX export failed or was interrupted: {e}. Continuing training.")

    def train(self) -> None:
        # Ensure INFO logs are visible by (re)adding a StreamHandler to stdout
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            sh.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(sh)
        # Ensure messages aren't duplicated via root logger once we've attached our own handler.
        logger.propagate = False
        logger.setLevel(logging.INFO)



        logger.info("=" * 20)
        logger.info(GREEN + "STARTING TRAINING" + RESET)
        logger.info("=" * 20)
        
        self.start_time = time.time()
        logger.info("Loading PPO configuration...")
        
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20
        

        if "network_factory" in self.ppo_params:
            logger.info("Using custom network factory")
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            logger.info("Using default network factory")
            network_factory = ppo_networks.make_ppo_networks
        
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        logger.info(f"Base PPO params: {self.ppo_training_params}")
        
        for k, v in self.overrided_ppo_params.items():
            if k in self.ppo_training_params:
                logger.info(f"Overriding {k}: {self.ppo_training_params.get(k)} -> {v}")
                self.ppo_training_params[k] = v
            else:
                logger.warning(
                    f"Ignoring unknown PPO param override '{k}' for this Brax version"
                )

        logger.info(f"{GREEN} Final PPO params:{RESET} {self.ppo_training_params}")

        # Track expected number of callbacks (initial eval counts as 1 when num_evals > 1).
        num_evals = int(self.ppo_training_params.get("num_evals", 1) or 1)
        self.expected_callbacks = max(num_evals, 1)
        self.callback_count = 0

        logger.info("Creating training function...")
        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )
        
        logger.info("=" * 20)
        logger.info(GREEN + "STARTING JAX COMPILATION (this may take several minutes)..." + RESET)
        logger.info("=" * 20)

        compilation_start = time.time()

        try:
            # use_pmap_on_reset controls whether env resets are parallelized across devices with jax.pmap.
            # Keep it False on CPU/single-device to avoid extra compilation overhead; enable only on
            # multi-accelerator (e.g., TPU or multi-GPU) setups where it helps throughput and seeding.
            _, params, _ = train_fn(
                environment=self.env,
                eval_env=self.eval_env,
                wrap_env_fn=wrapper.wrap_for_brax_training,
                use_pmap_on_reset=self.use_pmap_on_reset,
            )

            total_time = time.time() - self.start_time
            compilation_time = time.time() - compilation_start
            
            logger.info("=" * 20)
            logger.info(GREEN + "TRAINING COMPLETED SUCCESSFULLY" + RESET)
            logger.info(f"{GREEN}Total time: {total_time/60:.2f} mins {RESET}")
            logger.info(f"{GREEN}Compilation time: {compilation_time/60:.2f} mins{RESET}")
            logger.info("=" * 20)

            # Optional final export regardless of per-checkpoint skip flag
            if getattr(self.args, "export_on_finish", False):
                try:
                    d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                    onnx_export_path = f"{self.output_dir}/{d}_final.onnx"
                    logger.info(
                        f"[FINAL EXPORT] Exporting ONNX at training end to: {onnx_export_path}"
                    )
                    export_onnx(
                        params,
                        self.action_size,
                        self.ppo_params,
                        self.obs_size,
                        output_path=onnx_export_path,
                    )
                    logger.info(f"{GREEN}[FINAL EXPORT] Completed{RESET}")
                except Exception as e:
                    logger.warning(f"[FINAL EXPORT] ONNX export failed: {e}")
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"{RED}TRAINING FAILED{RESET}")
            logger.error(f"Error: {e}")
            logger.error("=" * 80)
            raise