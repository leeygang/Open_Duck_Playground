import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.wildrobot_dev.mujoco_infer_base import MJInferBase
from playground.wildrobot_dev import constants as wr_constants

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(
        self, model_path: str, onnx_model_path: str, standing: bool, task: str
    ):
        super().__init__(model_path, task=task)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)


        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        contacts = self.get_feet_contacts(data)

        # Fixed observation layout matching training (standing task):
        # [gyro(3), accel(3), command(3), joint_angle_delta(dof), joint_vel_scaled(dof),
        #  last_action(dof), last_last_action(dof), last_last_last_action(dof), contacts(2), ref_motion(4 zeros)]
        # Only first 3 entries of command (lin x, lin y, ang yaw) were used during training.
        cmd3 = np.array(command[:3], dtype=np.float32)
        obs = np.concatenate([
            gyro,
            accelerometer,
            cmd3,
            joint_angles - self.default_actuator,
            joint_vel * self.dof_vel_scale,
            self.last_action,
            self.last_last_action,
            self.last_last_last_action,
            contacts,
            np.zeros(4, dtype=np.float32),  # placeholder current reference motion
        ])
        if not hasattr(self, "_printed_obs_debug"):
            print("[OBS DEBUG] segment sizes -> gyro:", gyro.shape[0],
                  "accel:", accelerometer.shape[0],
                  "command(used):", len(cmd3),
                  "joint_angle_delta:", (joint_angles - self.default_actuator).shape[0],
                  "joint_vel:", (joint_vel * self.dof_vel_scale).shape[0],
                  "last_action:", self.last_action.shape[0],
                  "last_last_action:", self.last_last_action.shape[0],
                  "last_last_last_action:", self.last_last_last_action.shape[0],
                  "contacts:", len(contacts),
                  "placeholder:", 4,
                  "TOTAL:", obs.shape[0])
            self._printed_obs_debug = True
        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode == 72:  # toggle head control mode
            self.head_control_mode = not self.head_control_mode
            return
        # Movement or head control depending on mode
        if not self.head_control_mode:
            if keycode == 265:  # up
                self.commands[0] = self.COMMANDS_RANGE_X[1]
            elif keycode == 264:  # down
                self.commands[0] = self.COMMANDS_RANGE_X[0]
            elif keycode == 263:  # left
                self.commands[1] = self.COMMANDS_RANGE_Y[1]
            elif keycode == 262:  # right
                self.commands[1] = self.COMMANDS_RANGE_Y[0]
            elif keycode == 81:  # a spin left
                self.commands[2] = self.COMMANDS_RANGE_THETA[1]
            elif keycode == 69:  # e spin right
                self.commands[2] = self.COMMANDS_RANGE_THETA[0]
            else:
                # release resets velocity commands
                self.commands[0] = 0
                self.commands[1] = 0
                self.commands[2] = 0
        else:
            # Head control indices: 3:neck_pitch 4:head_pitch 5:head_yaw 6:head_roll
            if keycode == 265:  # up
                self.commands[4] = self.NECK_PITCH_RANGE[1]
            elif keycode == 264:  # down
                self.commands[4] = self.NECK_PITCH_RANGE[0]
            elif keycode == 263:  # left
                self.commands[5] = self.HEAD_YAW_RANGE[1]
            elif keycode == 262:  # right
                self.commands[5] = self.HEAD_YAW_RANGE[0]
            elif keycode == 81:  # a
                self.commands[6] = self.HEAD_ROLL_RANGE[1]
            elif keycode == 69:  # e
                self.commands[6] = self.HEAD_ROLL_RANGE[0]
            else:
                self.commands[3] = 0
                self.commands[4] = 0
                self.commands[5] = 0
                self.commands[6] = 0
    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1
                    if counter % self.decimation == 0:
                        obs = self.get_obs(self.data, self.commands)
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)
                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()
                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )
                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )
                            self.prev_motor_targets = self.motor_targets.copy()
                        self.data.ctrl = self.motor_targets.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, default=None,
                        help="Path to ONNX model; if omitted, uses output/<env>_<task>.onnx")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to MuJoCo XML; if omitted, resolved from task via constants.task_to_xml(task)")
    parser.add_argument("--standing", action="store_true", default=True)
    parser.add_argument("--task", type=str, default="wildrobot_terrain", help="Task name matching ROBOT_CONFIGS key")
    parser.add_argument("--env", type=str, default="standing", help="Env name used for ONNX filename (default: standing)")

    args = parser.parse_args()

    # Resolve defaults for model and ONNX paths
    from pathlib import Path
    if args.model_path is None:
        try:
            xml_path = wr_constants.task_to_xml(args.task)
            model_path = str(xml_path)
        except Exception as _e:
            raise SystemExit(f"Unable to resolve XML for task '{args.task}': {_e}")
    else:
        model_path = args.model_path

    if args.onnx_model_path is None:
        # Use unified filename policy from training: output/<env>_<task>.onnx
        output_root = Path("output")
        auto_onnx = output_root / f"{args.env}_{args.task}.onnx"
        onnx_path = str(auto_onnx)
    else:
        onnx_path = args.onnx_model_path

    print(f"[INFER] XML: {model_path}")
    print(f"[INFER] ONNX: {onnx_path}")
    if not Path(onnx_path).exists():
        print(f"[WARN] ONNX file not found at {onnx_path}. Ensure training exported the model or pass -o explicitly.")

    mjinfer = MjInfer(model_path, onnx_path, args.standing, args.task)
    mjinfer.run()
