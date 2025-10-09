"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""
# Must be first import
import playground.common.tb_logging  # noqa: F401

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
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax
import numpy as np

from playground.common.export_onnx import export_onnx


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
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        self.overrided_ppo_params = dict()
        
        # Timing
        self.start_time = None
        self.last_progress_time = None

        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

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
        logger.info("[BASE RUNNER] progress_callback")
        
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
        print(f"STEP: {int(num_steps)} reward: {_rew} reward_std: {_rew_std}")
        print("-----------")

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints
        logger.info(f"[CHECKPOINT] Saving at step {current_step}")

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        logger.info(f"Checkpoint path: {path}")
        
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        
        onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        logger.info(f"Exporting ONNX to: {onnx_export_path}")
        
        export_onnx(
            params,
            self.action_size,
            self.ppo_params,
            self.obs_size,  # may not work
            output_path=onnx_export_path
        )
        logger.info(f"Checkpoint and ONNX export completed")

    def train(self) -> None:
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        
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
            logger.info(f"Overriding {k}: {self.ppo_training_params.get(k)} -> {v}")
            self.ppo_training_params[k] = v

        logger.info(f"Final PPO params: {self.ppo_training_params}")

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
        
        logger.info("=" * 80)
        logger.info("STARTING JAX COMPILATION (this may take several minutes)...")
        logger.info("=" * 80)

        compilation_start = time.time()

        try:
            _, params, _ = train_fn(
                environment=self.env,
                eval_env=self.eval_env,
                wrap_env_fn=wrapper.wrap_for_brax_training,
            )

            total_time = time.time() - self.start_time
            compilation_time = time.time() - compilation_start
            
            logger.info("=" * 80)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Compilation time: {compilation_time:.2f}s")
            logger.info("=" * 80)
        except Exception as e:
            logger.error("=" * 80)
            logger.error("TRAINING FAILED")
            logger.error(f"Error: {e}")
            logger.error("=" * 80)
            raise