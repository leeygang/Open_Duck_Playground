"""Runs training and evaluation loop for WildRobot and Open Duck Mini

Adds a --debug flag to drastically reduce PPO workload and increase log frequency.
Adds a --profile flag with presets: 'high' (higher quality) and 'fast' (balanced CPU runtime).
Also prints JAX backend/devices to help diagnose long compile times or missing GPU.
"""

import argparse

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.wildrobot_dev import standing
import constants
import os

# os.environ['JAX_LOG_COMPILES'] = '1'  # Add at top of runner.py
# from jax import config
# config.update("jax_disable_jit", True)

class WildRobotRunner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
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

        # Profiles and debug overrides
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
                "log_frequency": 100,
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
                "log_frequency": 1,
            },
        }

        if getattr(args, "profile", None) in profiles:
            sel = args.profile
            self.overrided_ppo_params.update(profiles[sel])
            print(f"[PROFILE] Using '{sel}' PPO overrides: {self.overrided_ppo_params}")
            if getattr(args, "debug", False):
                print("[PROFILE] --profile provided; ignoring --debug overrides.")
        elif getattr(args, "debug", False):
            # Optional debug overrides to shrink compile/training time drastically
            self.overrided_ppo_params.update({
                "num_envs": 1,
                "unroll_length": 8,
                "num_minibatches": 1,
                "batch_size": 8,
                # Make logging very frequent so progress is visible
                "log_frequency": 1,
            })
            # Also clamp total timesteps for debug sessions
            self.num_timesteps = min(self.num_timesteps, 20_000)
            # Disable domain randomization in debug to avoid vmap over model which
            # can trigger JAX/device_put issues on some CPU-only setups.
            self.randomizer = None
            print("[DEBUG] Using reduced PPO parameters and timesteps for faster iteration.")
            print("[DEBUG] Domain randomization disabled for stability in debug mode.")

        # CLI log_frequency override takes precedence over profile/debug
        if getattr(args, "log_frequency", None) is not None:
            self.overrided_ppo_params["log_frequency"] = args.log_frequency
            print(f"[CLI] Overriding log_frequency: {args.log_frequency}")

        print(f"Observation size: {self.obs_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wild Robot Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--num_timesteps", type=int, default=300000000)
    #parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="standing", help="env")
    parser.add_argument("--task", type=str, default="wildrobot_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Reduce PPO workload and increase logging frequency for faster iteration",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force JAX to run on CPU (sets JAX_PLATFORM_NAME=cpu). Useful when CUDA is misconfigured.",
    )
    parser.add_argument(
        "--skip_onnx_export",
        action="store_true",
        help="Skip ONNX export during checkpoints to avoid TensorFlow/CUDA initialization overhead.",
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
            "Export a single ONNX model when training completes, even if per-checkpoint export is skipped."
        ),
    )
    parser.add_argument(
        "--heartbeat_interval",
        type=float,
        default=60.0,
        help=(
            "Seconds between heartbeat logs estimating progress between callbacks. "
            "Use 0 or negative to disable the heartbeat."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["high", "fast"],
        default=None,
        help=(
            "Preset PPO config: 'high' for higher quality (longer runs), 'fast' for balanced CPU runtime. "
            "If set, takes precedence over --debug."
        ),
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=None,
        help=(
            "How many PPO updates between progress callbacks. Lower values = more frequent logging. "
            "If not set, uses profile/debug default. Set to 1 for per-update logging."
        ),
    )
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    if args.cpu:
        # Set before JAX initializes
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        print("[Runner] Forcing JAX to CPU via JAX_PLATFORM_NAME=cpu")

    # Optional: show JAX compile logs when debugging with JIT enabled
    if args.debug and not args.no_jit:
        os.environ.setdefault("JAX_LOG_COMPILES", "1")

    runner = WildRobotRunner(args)

    runner.train()


if __name__ == "__main__":
    main()
