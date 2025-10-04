# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from etils import epath
from dataclasses import dataclass
from typing import List

ROOT_PATH = epath.Path(__file__).parent
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_flat_terrain.xml"
ROUGH_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_rough_terrain.xml"
DUCK_TERRAIN_XML = ROOT_PATH / "../open_duck_mini_v2" /"xmls" / "scene_flat_terrain.xml"


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "wildrobot_terrain": FLAT_TERRAIN_XML,
        "duck_terrain": DUCK_TERRAIN_XML
    }[task_name]


from dataclasses import dataclass
from typing import List

@dataclass
class RobotConfig:
    """Configuration for a single robot."""
    # Feet and contact geoms
    feet_sites: List[str]
    left_feet_geoms: List[str]
    right_feet_geoms: List[str]
    
    # Joint names
    hip_joint_names: List[str]
    knee_joint_names: List[str]
    joints_order_no_head: List[str]
    
    # Root body
    root_body: str
    trunk_imu: str
    
    # Sensor names
    gravity_sensor: str
    global_linvel_sensor: str
    global_angvel_sensor: str
    local_linvel_sensor: str
    accelerometer_sensors: List[str]  # Changed to List[str]
    gyro_sensors: List[str]  # Changed to List[str]
    
    @property
    def feet_geoms(self) -> List[str]:
        """Convenience property that combines left and right feet geoms."""
        return self.left_feet_geoms + self.right_feet_geoms
    
    @property
    def feet_pos_sensor(self) -> List[str]:
        """Generate feet position sensor names from feet sites."""
        return [f"{site}_pos" for site in self.feet_sites]
    
    @property
    def all_joint_names(self) -> List[str]:
        """Returns all joint names (hip + knee + ankle)."""
        return self.joints_order_no_head
    
    @property
    def left_leg_joints(self) -> List[str]:
        """Returns only left leg joint names."""
        return [j for j in self.joints_order_no_head if j.startswith("left_")]
    
    @property
    def right_leg_joints(self) -> List[str]:
        """Returns only right leg joint names."""
        return [j for j in self.joints_order_no_head if j.startswith("right_")]


# Define robot configurations
ROBOT_CONFIGS = {
    "wildrobot_terrain": RobotConfig(
        trunk_imu="left_leg_imu",
        feet_sites=["left_foot_site"],
        left_feet_geoms=["left_foot_btm_front", "left_foot_btm_back"],
        right_feet_geoms=[],
        hip_joint_names=[],
        knee_joint_names=["left_knee"],
        joints_order_no_head=["left_knee", "left_ankle", "left_foot"],
        root_body="upper_leg",
        gravity_sensor="upvector",
        global_linvel_sensor="global_linvel",
        global_angvel_sensor="global_angvel",
        local_linvel_sensor="local_linvel",
        accelerometer_sensors=["left_leg_accelerometer"],
        gyro_sensors=["left_leg_gyro"],
    ),
    "duck_terrain": RobotConfig(
        trunk_imu="imu",
        feet_sites=["left_foot", "right_foot"],
        left_feet_geoms=["left_foot_bottom_tpu"],
        right_feet_geoms=["right_foot_bottom_tpu"],
        hip_joint_names=[
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
        ],
        knee_joint_names=["left_knee", "right_knee"],
        joints_order_no_head=[
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
        ],
        root_body="trunk_assembly",
        gravity_sensor="upvector",
        global_linvel_sensor="global_linvel",
        global_angvel_sensor="global_angvel",
        local_linvel_sensor="local_linvel",
        accelerometer_sensors=["accelerometer"],
        gyro_sensors=["gyro"],
    ),
}


# # used to training debug info.
# FEET_SITES = [
#     "left_foot_site",
# ]

# # Used for contact
# LEFT_FEET_GEOMS = [
#     "left_foot_btm_front",
#     "left_foot_btm_back",
# ]

# # There should be a way to get that from the mjModel...
# JOINTS_ORDER_NO_HEAD = [
#     "left_knee",
#     "left_ankle",
#     "left_foot",
# ]


# SENSOR_GYRO = [
#     "left_leg_gyro",
# ]

# SENSOR_ACCELEROMETER = [
#     "left_leg_accelerometer",
# ]

# FEET_GEOMS = LEFT_FEET_GEOMS


# ROOT_BODY = "upper_leg"
# TRUNK_IMU = "left_leg_imu"

# GRAVITY_SENSOR = "upvector"
# LOCAL_LINVEL_SENSOR = "local_linvel"
# GLOBAL_LINVEL_SENSOR = "global_linvel"
# GLOBAL_ANGVEL_SENSOR = "global_angvel"

