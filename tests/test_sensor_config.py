import numpy as np
import pytest
from em_tactile_sim.core.sensor_config import SensorConfig, SensorVariant


def test_standard_defaults():
    cfg = SensorConfig()
    assert cfg.rows == 7
    assert cfg.cols == 7
    assert cfg.cell_spacing == pytest.approx(1e-3)
    assert cfg.fn_max == pytest.approx(20.0)
    assert cfg.ft_max == pytest.approx(10.0)
    assert cfg.sample_rate == pytest.approx(120.0)


def test_custom1_overrides():
    cfg = SensorConfig(variant=SensorVariant.CUSTOM1)
    assert cfg.rows == 6
    assert cfg.cols == 6
    assert cfg.product_length == pytest.approx(20e-3)
    assert cfg.product_width == pytest.approx(17e-3)
    assert cfg.product_height == pytest.approx(9e-3)


def test_sensor_dim_standard():
    cfg = SensorConfig()
    assert cfg.sensor_dim == 7 * 7 * 3 + 3 + 1  # 151


def test_sensor_dim_custom1():
    cfg = SensorConfig(variant=SensorVariant.CUSTOM1)
    assert cfg.sensor_dim == 6 * 6 * 3 + 3 + 1  # 112


def test_array_dim():
    cfg = SensorConfig()
    assert cfg.array_dim == 7 * 7 * 3  # 147


def test_cell_centers_shape():
    cfg = SensorConfig()
    centers = cfg.cell_centers
    assert centers.shape == (7, 7, 2)


def test_cell_centers_symmetric_about_origin():
    cfg = SensorConfig()
    centers = cfg.cell_centers
    assert np.abs(centers[:, :, 0].mean()) < 1e-12
    assert np.abs(centers[:, :, 1].mean()) < 1e-12


def test_cell_centers_spacing():
    cfg = SensorConfig()
    centers = cfg.cell_centers
    # Adjacent column spacing should equal cell_spacing
    dx = centers[0, 1, 0] - centers[0, 0, 0]
    assert dx == pytest.approx(cfg.cell_spacing)


def test_sensing_span():
    cfg = SensorConfig()
    assert cfg.sensing_span_x == pytest.approx(6e-3)   # (7-1)*1mm
    assert cfg.sensing_span_y == pytest.approx(6e-3)


def test_cell_area():
    cfg = SensorConfig()
    assert cfg.cell_area == pytest.approx(1e-6)          # 1mm²
