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


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "flat_terrain": FLAT_TERRAIN_XML,
    }[task_name]


# used to training debug info.
FEET_SITES = [
    "left_foot_site",
]

# Used for contact
LEFT_FEET_GEOMS = [
    "left_foot_btm_front",
    "left_foot_btm_back",
]

# There should be a way to get that from the mjModel...
JOINTS_ORDER_NO_HEAD = [
    "left_knee",
    "left_ankle",
    "left_foot",
]


SENSOR_GYRO = [
    "left_leg_gyro",
]

SENSOR_ACCELEROMETER = [
    "left_leg_accelerometer",
]

FEET_GEOMS = LEFT_FEET_GEOMS


ROOT_BODY = "upper_leg"
TRUNK_IMU = "left_leg_imu"

GRAVITY_SENSOR = "upvector"
LOCAL_LINVEL_SENSOR = "local_linvel"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"

