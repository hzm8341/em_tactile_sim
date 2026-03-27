"""Isaac Sim integration tests — auto-skipped if isaacsim not installed."""
import numpy as np
import pytest

isaacsim = pytest.importorskip(
    "isaacsim",
    reason="Isaac Sim not installed; skipping integration tests.",
)

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.isaac.env import EMTactileIsaacEnv

import os
USD = os.path.join(
    os.path.dirname(__file__),
    "../em_tactile_sim/isaac/models/em_sensor_flat.usda",
)


@pytest.fixture(scope="module")
def env():
    cfg = SensorConfig()
    e = EMTactileIsaacEnv(USD, cfg)
    e.setup()
    yield e
    e.close()


def test_env_setup_succeeds(env):
    assert env is not None


def test_output_shapes_after_step(env):
    env.step()
    assert env.get_tactile().shape      == (7, 7, 3)
    assert env.get_resultant().shape    == (3,)
    assert env.get_tactile_flat().shape == (151,)


def test_no_contact_at_start(env):
    env.reset()
    env.step()
    tactile = env.get_tactile()
    np.testing.assert_array_almost_equal(tactile[:, :, 0], 0.0)


def test_contact_produces_nonzero_after_fall(env):
    env.reset()
    for _ in range(300):   # ~2.5s @ 120Hz, ball falls 5cm
        env.step()
    fn_max = env.get_tactile()[:, :, 0].max()
    assert fn_max > 0.0, "Expected nonzero normal force after ball contact"


def test_reset_clears_output(env):
    for _ in range(300):
        env.step()
    env.reset()
    env.step()
    fn_max = env.get_tactile()[:, :, 0].max()
    assert fn_max == pytest.approx(0.0, abs=1e-6)


def test_get_temperature_zero(env):
    env.step()
    assert env.get_temperature() == pytest.approx(0.0)
