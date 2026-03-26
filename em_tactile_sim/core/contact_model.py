"""Hertz contact mechanics model for the EM tactile sensor simulation.

Maps contact events (position, force, radius) to a (rows, cols, 3) force
distribution array using Hertz contact theory with linear superposition
for multiple contacts.
"""
import numpy as np
from .sensor_config import SensorConfig


class ContactModel:
    """Hertz contact mechanics: contact dict → (rows, cols, 3) force array."""

    def __init__(self, config: SensorConfig) -> None:
        self._cfg = config
        # Effective elastic modulus for Hertz model
        self._E_star = config.elastic_modulus / (2.0 * (1.0 - config.poisson_ratio ** 2))

    def compute(self, contact: dict) -> np.ndarray:
        """
        Compute force distribution for a single contact event.

        Args:
            contact: {
                "pos":    np.ndarray([cx, cy])   sensor-local coords, m
                "fn":     float                   normal force, N (compression > 0)
                "ft":     np.ndarray([Ftx, Fty]) tangential force, N
                "radius": float                   contact body equivalent radius, m
            }
        Returns:
            np.ndarray shape (rows, cols, 3): [fn, ftx, fty] per cell, N
        """
        fn = float(np.clip(contact["fn"], 0.0, self._cfg.fn_max))
        ft = np.clip(np.asarray(contact["ft"], dtype=float),
                     -self._cfg.ft_max, self._cfg.ft_max)

        if fn < self._cfg.sensitivity:
            return np.zeros((self._cfg.rows, self._cfg.cols, 3))

        cx, cy = float(contact["pos"][0]), float(contact["pos"][1])
        R = float(contact["radius"])

        # Hertz contact radius: a = (3*F*R / 4*E*)^(1/3)
        a = (3.0 * fn * R / (4.0 * self._E_star)) ** (1.0 / 3.0)

        centers = self._cfg.cell_centers        # (rows, cols, 2)
        dx = centers[:, :, 0] - cx              # (rows, cols)
        dy = centers[:, :, 1] - cy
        r2 = dx ** 2 + dy ** 2

        # Hertz pressure: p0 * sqrt(max(0, 1 - (r/a)^2))
        # p0 = 3*F / (2*pi*a^2)
        arg = np.maximum(0.0, 1.0 - r2 / (a ** 2))
        sigma_n = (3.0 * fn / (2.0 * np.pi * a ** 2)) * np.sqrt(arg)  # Pa

        # Tangential stress proportional to normal stress
        eps = 1e-9
        sigma_tx = sigma_n * (ft[0] / max(fn, eps))
        sigma_ty = sigma_n * (ft[1] / max(fn, eps))

        area = self._cfg.cell_area
        return np.stack([
            sigma_n  * area,
            sigma_tx * area,
            sigma_ty * area,
        ], axis=-1)                             # (rows, cols, 3)

    def compute_multi(self, contacts: list) -> np.ndarray:
        """Linear superposition over multiple contacts.

        Args:
            contacts: list of contact dicts (same format as compute())
        Returns:
            np.ndarray shape (rows, cols, 3): summed force array
        """
        result = np.zeros((self._cfg.rows, self._cfg.cols, 3))
        for c in contacts:
            result += self.compute(c)
        return result
