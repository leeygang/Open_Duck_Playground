# import jax
# import jax.numpy as np
import numpy as np


def reward_imitation(
    base_qpos,
    base_qvel,
    joints_qpos,
    joints_qvel,
    contacts,
    reference_frame,
    cmd,
    use_imitation_reward=False,
):
    if not use_imitation_reward:
        return np.nan_to_num(0.0)

    # TODO don't reward for moving when the command is zero.
    cmd_norm = np.linalg.norm(cmd[:3])

    w_torso_pos = 1.0
    w_torso_orientation = 1.0
    w_lin_vel_xy = 1.0
    w_lin_vel_z = 1.0
    w_ang_vel_xy = 0.5
    w_ang_vel_z = 0.5
    w_joint_pos = 15.0
    w_joint_vel = 1.0e-3
    w_contact = 1.0

    #  TODO : double check if the slices are correct
    linear_vel_slice_start = 34
    linear_vel_slice_end = 37

    angular_vel_slice_start = 37
    angular_vel_slice_end = 40

    joint_pos_slice_start = 0
    joint_pos_slice_end = 16

    joint_vels_slice_start = 16
    joint_vels_slice_end = 32

    # root_pos_slice_start = 0
    # root_pos_slice_end = 3

    root_quat_slice_start = 3
    root_quat_slice_end = 7

    # left_toe_pos_slice_start = 23
    # left_toe_pos_slice_end = 26

    # right_toe_pos_slice_start = 26
    # right_toe_pos_slice_end = 29

    foot_contacts_slice_start = 32
    foot_contacts_slice_end = 34

    # ref_base_pos = reference_frame[root_pos_slice_start:root_pos_slice_end]
    # base_pos = qpos[:3]

    ref_base_orientation_quat = reference_frame[
        root_quat_slice_start:root_quat_slice_end
    ]
    ref_base_orientation_quat = ref_base_orientation_quat / np.linalg.norm(
        ref_base_orientation_quat
    )  # normalize the quat
    base_orientation = base_qpos[3:7]
    base_orientation = base_orientation / np.linalg.norm(
        base_orientation
    )  # normalize the quat

    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]

    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    # remove neck head and antennas
    ref_joint_pos = np.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])
    # joint_pos = joints_qpos
    joint_pos = np.concatenate([joints_qpos[:5], joints_qpos[9:]])

    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    # remove neck head and antennas
    ref_joint_vels = np.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    # joint_vel = joints_qvel
    joint_vel = np.concatenate([joints_qvel[:5], joints_qvel[9:]])

    # ref_left_toe_pos = reference_frame[left_toe_pos_slice_start:left_toe_pos_slice_end]
    # ref_right_toe_pos = reference_frame[right_toe_pos_slice_start:right_toe_pos_slice_end]

    ref_foot_contacts = reference_frame[
        foot_contacts_slice_start:foot_contacts_slice_end
    ]

    # reward
    # torso_pos_rew = np.exp(-200.0 * np.sum(np.square(base_pos[:2] - ref_base_pos[:2]))) * w_torso_pos

    # real quaternion angle doesn't have the expected  effect, switching back for now
    # torso_orientation_rew = np.exp(-20 * self.quaternion_angle(base_orientation, ref_base_orientation_quat)) * w_torso_orientation

    # TODO ignore yaw here, we just want xy orientation
    torso_orientation_rew = (
        np.exp(-20.0 * np.sum(np.square(base_orientation - ref_base_orientation_quat)))
        * w_torso_orientation
    )

    lin_vel_xy_rew = (
        np.exp(-8.0 * np.sum(np.square(base_lin_vel[:2] - ref_base_lin_vel[:2])))
        * w_lin_vel_xy
    )
    lin_vel_z_rew = (
        np.exp(-8.0 * np.sum(np.square(base_lin_vel[2] - ref_base_lin_vel[2])))
        * w_lin_vel_z
    )

    ang_vel_xy_rew = (
        np.exp(-2.0 * np.sum(np.square(base_ang_vel[:2] - ref_base_ang_vel[:2])))
        * w_ang_vel_xy
    )
    ang_vel_z_rew = (
        np.exp(-2.0 * np.sum(np.square(base_ang_vel[2] - ref_base_ang_vel[2])))
        * w_ang_vel_z
    )

    joint_pos_rew = -np.sum(np.square(joint_pos - ref_joint_pos)) * w_joint_pos
    joint_vel_rew = -np.sum(np.square(joint_vel - ref_joint_vels)) * w_joint_vel

    ref_foot_contacts = np.where(
        np.array(ref_foot_contacts) > 0.5,
        np.ones_like(ref_foot_contacts),
        np.zeros_like(ref_foot_contacts),
    )
    contact_rew = np.sum(contacts == ref_foot_contacts) * w_contact

    reward = (
        lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + joint_pos_rew
        + joint_vel_rew
        + contact_rew
        # + torso_orientation_rew
    )

    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return np.nan_to_num(reward)
