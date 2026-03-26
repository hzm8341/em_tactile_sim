from dataclasses import dataclass
from enum import Enum
import numpy as np


class SensorVariant(Enum):
    STANDARD = "standard"   # 7x7, 43x28x8mm, Type C
    CUSTOM1  = "custom1"    # 6x6, 20x17x9mm, CAN (fingertip)


@dataclass
class SensorConfig:
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

    def __post_init__(self) -> None:
        if self.variant == SensorVariant.CUSTOM1:
            self.rows           = 6
            self.cols           = 6
            self.product_length = 20e-3
            self.product_width  = 17e-3
            self.product_height = 9e-3

    @property
    def sensor_dim(self) -> int:
        """Total output dim: array(rows*cols*3) + resultant(3) + temp(1)."""
        return self.rows * self.cols * 3 + 3 + 1

    @property
    def array_dim(self) -> int:
        return self.rows * self.cols * 3

    @property
    def sensing_span_x(self) -> float:
        return (self.cols - 1) * self.cell_spacing

    @property
    def sensing_span_y(self) -> float:
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
        return self.cell_spacing ** 2
