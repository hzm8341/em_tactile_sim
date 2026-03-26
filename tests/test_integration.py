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
    e = EMTactileEnv(XML, cfg)
    yield e
    e.close()


def test_import_callback():
    from em_tactile_sim.mujoco.callback import EMSensorCallback
    assert EMSensorCallback is not None


def test_callback_no_contact():
    """Zero contacts → 151-dim zero vector in sensordata."""
    import mujoco
    from em_tactile_sim.core.sensor_config import SensorConfig
    from em_tactile_sim.mujoco.callback import EMSensorCallback

    xml = os.path.join(
        os.path.dirname(__file__),
        "../em_tactile_sim/mujoco/models/em_sensor_flat.xml",
    )
    cfg = SensorConfig()
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    cb = EMSensorCallback(model, cfg)   # note: data removed from signature
    cb.register()

    mujoco.mj_resetData(model, data)
    mujoco.mj_step(model, data)   # no ball contact at t=0 (ball starts 5cm above)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "EM_SENSOR")
    adr = int(model.sensor_adr[sid])
    sensor_out = data.sensordata[adr: adr + cfg.sensor_dim]

    assert data.ncon == 0
    assert sensor_out.shape == (151,)
    # At t=0 the ball hasn't landed yet — expect zeros (or near-zero)
    assert np.max(np.abs(sensor_out)) < 1e-6

    cb.unregister()


def test_callback_output_shape_after_contact():
    """After ball lands, sensordata has correct shape and non-zero values."""
    import mujoco
    from em_tactile_sim.core.sensor_config import SensorConfig
    from em_tactile_sim.mujoco.callback import EMSensorCallback

    xml = os.path.join(
        os.path.dirname(__file__),
        "../em_tactile_sim/mujoco/models/em_sensor_flat.xml",
    )
    cfg = SensorConfig()
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    cb = EMSensorCallback(model, cfg)
    cb.register()

    mujoco.mj_resetData(model, data)
    # Step ~200 times (200 * 0.00833s ≈ 1.67s) to let ball fall and contact pad
    for _ in range(200):
        mujoco.mj_step(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "EM_SENSOR")
    adr = int(model.sensor_adr[sid])
    sensor_out = data.sensordata[adr: adr + cfg.sensor_dim]

    assert data.ncon > 0, "Expected contact after 200 steps"
    assert sensor_out.shape == (151,)
    # Ball should have contacted pad — resultant force (cfg.array_dim to +3) should exceed sensitivity
    resultant = sensor_out[cfg.array_dim: cfg.array_dim + 3]
    assert resultant[0] > cfg.sensitivity, "Expected resultant[0] > sensitivity after contact"

    cb.unregister()


def test_env_creates_without_error():
    cfg = SensorConfig()
    env = EMTactileEnv(XML, cfg)
    assert env is not None
    env.close()


def test_output_shapes_after_step(env):
    env.step()
    assert env.get_tactile().shape      == (7, 7, 3)
    assert env.get_resultant().shape    == (3,)
    assert env.get_tactile_flat().shape == (env._cfg.sensor_dim,)


def test_no_contact_zero_output(env):
    """At t=0 the ball is 5cm above pad — no contact yet."""
    env.step()
    tactile = env.get_tactile()
    # fn channel should be 0 (ball hasn't fallen yet)
    np.testing.assert_array_almost_equal(tactile[:, :, 0], 0.0)


def test_contact_produces_nonzero_after_fall(env):
    """After ~300 steps ball should reach pad (0.05m fall, g=9.81, dt=0.00833s)."""
    for _ in range(300):
        env.step()
    fn_max = env.get_tactile()[:, :, 0].max()
    assert fn_max > env._cfg.sensitivity, "Expected fn_max > sensitivity after contact"


def test_reset_clears_output(env):
    for _ in range(300):
        env.step()
    env.reset()
    env.step()
    fn_max = env.get_tactile()[:, :, 0].max()
    assert fn_max == pytest.approx(0.0, abs=1e-6)


def test_get_temperature_default(env):
    env.step()
    assert env.get_temperature() == pytest.approx(0.0)
