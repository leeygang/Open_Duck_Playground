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


ROOT_PATH = epath.Path(__file__).parent
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_flat_terrain.xml"
ROUGH_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_rough_terrain.xml"
FLAT_TERRAIN_BACKLASH_XML = ROOT_PATH / "xmls" / "scene_flat_terrain_backlash.xml"
ROUGH_TERRAIN_BACKLASH_XML = ROOT_PATH / "xmls" / "scene_rough_terrain_backlash.xml"


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "flat_terrain": FLAT_TERRAIN_XML,
    }[task_name]


FEET_SITES = [
    "left_foot_front",
    "left_foot_back",
    "right_foot_front",
    "right_foot_back",
]

LEFT_FEET_GEOMS = [
    "left_foot_btm_front",
    "left_foot_btm_back",
]

RIGHT_FEET_GEOMS = [
    "right_foot_btm_front",
    "right_foot_btm_back",
]


# There should be a way to get that from the mjModel...
JOINTS_ORDER_NO_HEAD = [
    "waist",
    "left_waist_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_waist_hip",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
]


SENSOR_GYRO = [
    "trunk_gyro",
    "left_leg_gyro",
    "right_leg_gyro",
]

SENSOR_ACCELEROMETER = [
    "trunk_accelerometer",
    "left_leg_accelerometer",
    "right_leg_accelerometer",
]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS


ROOT_BODY = "waist"
TRUNK_IMU = "trunk_imu"
GRAVITY_SENSOR = "trunk_upvector"
MAGNETOMETER_SENSOR = "trunk_magnetometer"

# debug
GLOBAL_LINVEL_SENSOR = "trunk_global_linvel"
GLOBAL_ANGVEL_SENSOR = "trunk_global_angvel"
LOCAL_LINVEL_SENSOR = "trunk_local_linvel"

