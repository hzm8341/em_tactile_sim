"""Sensor configuration dataclass and variants for the EM tactile sensor simulation.

Defines physical and electrical parameters for both STANDARD (7×7, 43×28×8mm)
and CUSTOM1 (6×6, 20×17×9mm) sensor configurations.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SensorVariant(Enum):
    STANDARD = "standard"   # 7x7, 43x28x8mm, Type C
    CUSTOM1  = "custom1"    # 6x6, 20x17x9mm, CAN (fingertip)


@dataclass
class SensorConfig:
    """Configuration parameters for the 元触科技 electromagnetic tactile sensor.

    Supports two physical variants (STANDARD and CUSTOM1). Use SensorVariant
    to select the variant; __post_init__ applies variant-specific overrides.
    All dimensional fields are in SI units (m, N, Pa, Hz).
    """
    variant: SensorVariant = SensorVariant.STANDARD

    # Array geometry
    rows:         int   = 7
    cols:         int   = 7
    cell_spacing: float = 1e-3      # m, 1mm spatial resolution

    # Product dimensions (m)
    product_length: float = 43e-3
    product_width:  float = 28e-3
    product_height: float = 8e-3

    # Force range & sensitivity
    fn_max:      float = 20.0   # N, normal force max
    ft_max:      float = 10.0   # N, shear force max (per axis)
    sensitivity: float = 0.05   # N, min detectable
    sample_rate: float = 120.0  # Hz

    # Elastic layer (soft silicone, to be calibrated)
    elastic_modulus: float = 0.2e6  # Pa
    poisson_ratio:   float = 0.47
    layer_thickness: float = 2e-3   # m

    # Hall mapping (Phase 1: unity, Phase 4: calibrated)
    hall_sensitivity: float = 1.0

    # MuJoCo sensor name (must match the <sensor><user name=...> in the XML)
    sensor_name: str = "EM_SENSOR"

    def __post_init__(self) -> None:
        if self.variant == SensorVariant.CUSTOM1:
            self.rows           = 6
            self.cols           = 6
            self.product_length = 20e-3
            self.product_width  = 17e-3
            self.product_height = 9e-3
            self.sensor_name    = "EM_SENSOR_CUSTOM1"

    @property
    def sensor_dim(self) -> int:
        """Total output dim: array(rows*cols*3) + resultant(3) + temp(1)."""
        return self.rows * self.cols * 3 + 3 + 1

    @property
    def array_dim(self) -> int:
        """Dimension of flattened array output: rows * cols * 3 force channels."""
        return self.rows * self.cols * 3

    @property
    def sensing_span_x(self) -> float:
        """Width of sensing area in X direction, m: (cols-1) * cell_spacing."""
        return (self.cols - 1) * self.cell_spacing

    @property
    def sensing_span_y(self) -> float:
        """Height of sensing area in Y direction, m: (rows-1) * cell_spacing."""
        return (self.rows - 1) * self.cell_spacing

    @property
    def cell_centers(self) -> np.ndarray:
        """Cell center coordinates, shape (rows, cols, 2), origin at pad center."""
        xs = np.linspace(-self.sensing_span_x / 2,
                          self.sensing_span_x / 2, self.cols)
        ys = np.linspace(-self.sensing_span_y / 2,
                          self.sensing_span_y / 2, self.rows)
        xx, yy = np.meshgrid(xs, ys)
        return np.stack([xx, yy], axis=-1)   # (rows, cols, 2)

    @property
    def cell_area(self) -> float:
        """Area of one sensing cell, m²: cell_spacing²."""
        return self.cell_spacing ** 2
