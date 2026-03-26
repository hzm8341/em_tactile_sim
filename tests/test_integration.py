"""Integration tests: requires MuJoCo + em_sensor_flat.xml."""
import os
import numpy as np
import pytest
import mujoco

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.mujoco.callback import EMSensorCallback
from em_tactile_sim.mujoco.env import EMTactileEnv

XML = os.path.join(
    os.path.dirname(__file__),
    "../em_tactile_sim/mujoco/models/em_sensor_flat.xml",
)


@pytest.fixture
def env():
    cfg = SensorConfig()
    return EMTactileEnv(XML, cfg)


def test_import_callback():
    from em_tactile_sim.mujoco.callback import EMSensorCallback
    assert EMSensorCallback is not None
