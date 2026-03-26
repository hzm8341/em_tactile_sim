import numpy as np
import pytest
from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.core import hall_response


@pytest.fixture
def cfg() -> SensorConfig:
    return SensorConfig()


def test_output_shape(cfg):
    arr = np.ones((cfg.rows, cfg.cols, 3))
    out = hall_response.compute_output(arr, cfg)
    assert out.shape == (cfg.sensor_dim,)  # (151,)


def test_zero_input_zero_array(cfg):
    arr = np.zeros((cfg.rows, cfg.cols, 3))
    out = hall_response.compute_output(arr, cfg)
    np.testing.assert_array_equal(out[:cfg.array_dim], 0.0)


def test_zero_input_zero_resultant(cfg):
    arr = np.zeros((cfg.rows, cfg.cols, 3))
    out = hall_response.compute_output(arr, cfg)
    np.testing.assert_array_equal(out[cfg.array_dim:cfg.array_dim + 3], 0.0)


def test_default_temperature_zero(cfg):
    arr = np.zeros((cfg.rows, cfg.cols, 3))
    out = hall_response.compute_output(arr, cfg)
    assert out[-1] == pytest.approx(0.0)


def test_temperature_passthrough(cfg):
    arr = np.zeros((cfg.rows, cfg.cols, 3))
    out = hall_response.compute_output(arr, cfg, temperature=25.5)
    assert out[-1] == pytest.approx(25.5)


def test_resultant_fn_equals_array_sum(cfg):
    arr = np.zeros((cfg.rows, cfg.cols, 3))
    arr[3, 3, 0] = 1.0   # set one cell fn
    out = hall_response.compute_output(arr, cfg)
    # Resultant fn (index array_dim+0) should equal 1.0
    assert out[cfg.array_dim] == pytest.approx(1.0)


def test_hall_sensitivity_scales_array(cfg):
    cfg.hall_sensitivity = 2.0
    arr = np.ones((cfg.rows, cfg.cols, 3))
    out = hall_response.compute_output(arr, cfg)
    np.testing.assert_array_almost_equal(out[:cfg.array_dim], 2.0)


def test_custom1_output_shape(cfg):
    from em_tactile_sim.core.sensor_config import SensorVariant
    cfg2 = SensorConfig(variant=SensorVariant.CUSTOM1)
    arr = np.zeros((cfg2.rows, cfg2.cols, 3))
    out = hall_response.compute_output(arr, cfg2)
    assert out.shape == (cfg2.sensor_dim,)   # (112,)
