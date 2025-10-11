"""Runs training and evaluation loop for WildRobot and Open Duck Mini

Provides a --profile flag with presets: 'high' (higher quality), 'fast' (balanced CPU runtime),
and 'debug' (drastically reduced workload for fast iteration).
Also prints JAX backend/devices to help diagnose long compile times or missing GPU.
"""

# Redirect .pyc files to output/pycache for this runner only
import os as _os  # placed before other imports to affect subsequent module imports
_os.environ.setdefault("PYTHONPYCACHEPREFIX", _os.path.join("output", "__pycache__"))

import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.wildrobot_dev import standing
import constants
import os
import playground.common.tb_logging as tb_logging

# os.environ['JAX_LOG_COMPILES'] = '1'  # Add at top of runner.py
# from jax import config
# config.update("jax_disable_jit", True)

class WildRobotRunner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
    # Redirect TensorBoard logs to output/runs while checkpoints stay in output/checkpoints
        try:
            out_path = Path(self.output_dir)
            output_root = out_path.parent if out_path.name == "checkpoints" else out_path
            runs_dir = output_root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            # Close existing writer if present, then replace and inform tb_logging
            try:
                if hasattr(self, "writer") and self.writer is not None:
                    self.writer.close()
            except Exception:
                pass
            self.writer = SummaryWriter(log_dir=runs_dir)
            try:
                tb_logging.set_writer(self.writer)
            except Exception:
                pass
            print(f"[Runner] TensorBoard logs -> {runs_dir}")
        except Exception as _e:
            print("[Runner] Failed to redirect TensorBoard logs:", _e)
        available_envs = {
            "joystick": (standing, standing.Standing),
            "standing": (standing, standing.Standing),
        }

        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        if constants.is_valid_task(args.task) is False:
            raise ValueError(f"Unknown task {args.task}, available tasks: {constants.tasks}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path

        # Print device/backend info to help diagnose performance
        try:
            import jax as _jax
            print("[JAX] default backend:", _jax.default_backend())
            print("[JAX] devices:", _jax.devices())
        except Exception as _e:
            print("[JAX] Unable to query devices:", _e)

        # Profiles (including a debug preset)
        self.overrided_ppo_params = {}
        profiles = {
            "high": {
                # Higher quality settings (CPU-feasible with JIT). Expect long runs.
                "num_envs": 16,
                "unroll_length": 128,
                "num_minibatches": 8,
                "batch_size": 2048,  # ~= num_envs * unroll_length
                "learning_rate": 3e-4,
                "discounting": 0.99,
                "gae_lambda": 0.95,
                "entropy_cost": 1e-3,
                "normalize_advantage": True,
                "num_evals": 100,  # More frequent progress callbacks
            },
            "fast": {
                # Balanced runtime/quality for CPU with JIT.
                "num_envs": 8,
                "unroll_length": 32,
                "num_minibatches": 4,
                "batch_size": 1024,
                "learning_rate": 3e-4,
                "discounting": 0.99,
                "gae_lambda": 0.95,
                "entropy_cost": 1e-3,
                "normalize_advantage": True,
                "num_evals": 1,  # Moderate callback frequency
            },
            "debug": {
                # Minimal workload for quick iteration.
                "num_envs": 1,
                "unroll_length": 8,
                "num_minibatches": 1,
                "batch_size": 8,
                "num_evals": 1,
            },
        }

        if getattr(args, "profile", None) in profiles:
            sel = args.profile
            self.overrided_ppo_params.update(profiles[sel])
            print(f"[PROFILE] Using '{sel}' PPO overrides: {self.overrided_ppo_params}")
            if sel == "debug":
                self.num_timesteps = min(self.num_timesteps, 20_000)
                self.randomizer = None
                print("[DEBUG PROFILE] Using reduced PPO parameters and timesteps for faster iteration.")
                print("[DEBUG PROFILE] Domain randomization disabled for stability in debug mode.")

        # CLI num_evals override takes precedence over profile defaults
        if getattr(args, "num_evals", None) is not None:
            self.overrided_ppo_params["num_evals"] = args.num_evals
            print(f"[CLI] Overriding num_evals (callback frequency): {args.num_evals}")

        print(f"Observation size: {self.obs_size}")

    # Also export/copy ONNX to the output folder (i.e., parent of 'checkpoints')
    # in addition to the default export handled by BaseRunner.
    def policy_params_fn(self, current_step, make_policy, params):
        # First, call the base implementation to save checkpoint and export ONNX under output_dir
        try:
            super().policy_params_fn(current_step, make_policy, params)
        except Exception as _e:
            print("[Runner Override] Base policy_params_fn failed:", _e)

        try:
            # Locate the most recent ONNX exported by the base implementation
            from pathlib import Path
            import shutil
            out_dir = Path(self.output_dir)
            if out_dir.exists():
                # Find ONNX files matching the naming pattern in output_dir
                onnx_files = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
            else:
                onnx_files = []
            if not onnx_files:
                return
            latest_onnx = onnx_files[0]

            # Determine output root (parent of 'checkpoints' if present)
            output_root = out_dir.parent if out_dir.name == "checkpoints" else out_dir
            output_root.mkdir(parents=True, exist_ok=True)

            # Name ONNX as <task>_<env>.onnx (e.g., wildrobot_terrain_standing.onnx)
            task_name = getattr(self.args, "task", "task")
            env_name = getattr(self.args, "env", "env")
            target_name = f"{task_name}_{env_name}.onnx"
            target_path = output_root / target_name
            shutil.copy2(latest_onnx, target_path)
            print(f"[Runner Override] Copied ONNX to {target_path}")
        except Exception as _e:
            print("[Runner Override] Failed to copy ONNX to task folder:", _e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wild Robot Runner Script")
    # Output directory for wildrobot_dev is fixed to output/checkpoints
    # 120M steps for fast profile
    parser.add_argument("--num_timesteps", type=int, default=120000000)
    #parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="standing", help="env")
    parser.add_argument("--task", type=str, default="leg_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force JAX to run on CPU (sets JAX_PLATFORM_NAME=cpu). Useful when CUDA is misconfigured.",
    )
    parser.add_argument(
        "--no_jit",
        action="store_true",
        help="Disable JAX JIT compilation for faster startup on CPU-only systems (slower per-step but no long compiles).",
    )
    parser.add_argument(
        "--export_on_finish",
        action="store_true",
        help=(
            "Skip intermediate ONNX exports but emit a final ONNX model when training completes."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["high", "fast", "debug"],
        default=None,
        help=(
            "Preset PPO config: 'high' for higher quality (longer runs), 'fast' for balanced CPU runtime, "
            "'debug' for minimal workload and higher log frequency."
        ),
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=1,
        help=(
            "Number of evaluation iterations (controls progress callback frequency). "
            "Training is divided into num_evals iterations, with a callback after each. "
            "Higher values = more frequent callbacks but smaller steps per callback. "
            "If not set, uses the profile default."
        ),
    )
    args = parser.parse_args()

    # Force outputs under output/checkpoints for wildrobot_dev only
    try:
        args.output_dir = os.path.join("output", "checkpoints")
    except Exception:
        # Fallback if args is not writable
        pass

    if args.cpu:
        # Set before JAX initializes
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        print("[Runner] Forcing JAX to CPU via JAX_PLATFORM_NAME=cpu")

    # Optional: show JAX compile logs when debugging with JIT enabled
    if getattr(args, "profile", None) == "debug" and not args.no_jit:
        os.environ.setdefault("JAX_LOG_COMPILES", "1")

    runner = WildRobotRunner(args)

    runner.train()


if __name__ == "__main__":
    main()
