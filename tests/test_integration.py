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

    assert sensor_out.shape == (151,)
    # Ball should have contacted pad — resultant force (indices 147-149) should be non-zero
    resultant = sensor_out[147:150]
    assert np.max(np.abs(resultant)) > 0.0, "Expected non-zero resultant after contact"

    cb.unregister()
