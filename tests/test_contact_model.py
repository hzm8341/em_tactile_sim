import numpy as np
import pytest
from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.core.contact_model import ContactModel


@pytest.fixture
def cfg() -> SensorConfig:
    return SensorConfig()


@pytest.fixture
def model(cfg: SensorConfig) -> ContactModel:
    return ContactModel(cfg)


def _contact(pos=(0.0, 0.0), fn=1.0, ftx=0.0, fty=0.0, r=0.005):
    return {"pos": np.array(pos), "fn": fn,
            "ft": np.array([ftx, fty]), "radius": r}


def test_output_shape(model, cfg):
    result = model.compute(_contact())
    assert result.shape == (cfg.rows, cfg.cols, 3)


def test_zero_force_returns_zeros(model, cfg):
    result = model.compute(_contact(fn=0.0))
    np.testing.assert_array_equal(result, 0.0)


def test_below_sensitivity_returns_zeros(model, cfg):
    result = model.compute(_contact(fn=cfg.sensitivity * 0.5))
    np.testing.assert_array_equal(result, 0.0)


def test_center_press_max_at_center(model, cfg):
    result = model.compute(_contact(pos=(0.0, 0.0), fn=2.0))
    fn_map = result[:, :, 0]
    cx, cy = cfg.rows // 2, cfg.cols // 2
    assert fn_map[cx, cy] == pytest.approx(fn_map.max())


def test_center_press_symmetric(model, cfg):
    result = model.compute(_contact(pos=(0.0, 0.0), fn=2.0))
    fn_map = result[:, :, 0]
    # Left-right symmetry
    np.testing.assert_array_almost_equal(fn_map, fn_map[:, ::-1])
    # Top-bottom symmetry
    np.testing.assert_array_almost_equal(fn_map, fn_map[::-1, :])


def test_fn_clamped_at_max(model, cfg):
    r1 = model.compute(_contact(fn=cfg.fn_max * 2))
    r2 = model.compute(_contact(fn=cfg.fn_max))
    np.testing.assert_array_almost_equal(r1, r2)


def test_ft_clamped_at_max(model, cfg):
    r1 = model.compute(_contact(fn=2.0, ftx=cfg.ft_max * 3))
    r2 = model.compute(_contact(fn=2.0, ftx=cfg.ft_max))
    np.testing.assert_array_almost_equal(r1, r2)


def test_tangential_direction_positive(model):
    result = model.compute(_contact(fn=2.0, ftx=1.0, fty=0.0))
    # All cells: ftx channel should be >= 0
    assert np.all(result[:, :, 1] >= 0)


def test_larger_radius_flatter_distribution(model):
    r_small = model.compute(_contact(fn=2.0, r=0.001))[:, :, 0]
    r_large = model.compute(_contact(fn=2.0, r=0.020))[:, :, 0]
    # Larger contact radius → lower peak pressure
    assert r_small.max() > r_large.max()


def test_superposition(model, cfg):
    c1 = _contact(pos=(0.0, 0.0), fn=2.0)
    c2 = _contact(pos=(1e-3, 0.0), fn=1.0)
    r1 = model.compute(c1)
    r2 = model.compute(c2)
    r_super = model.compute_multi([c1, c2])
    np.testing.assert_array_almost_equal(r_super, r1 + r2)


def test_compute_multi_empty(model, cfg):
    result = model.compute_multi([])
    assert result.shape == (cfg.rows, cfg.cols, 3)
    np.testing.assert_array_equal(result, 0.0)
