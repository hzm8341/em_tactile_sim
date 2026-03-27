"""Hall sensor response mapping for the EM tactile sensor simulation.

Phase 1: linear mapping (force_array × hall_sensitivity).
Phase 4: replace with LookupTableResponse or MLPResponse — same interface.
"""
import numpy as np
from .sensor_config import SensorConfig


def compute_output(force_array: np.ndarray,
                   config: SensorConfig,
                   temperature: float = 0.0) -> np.ndarray:
    """
    Map physical force array to the full sensor output vector.

    Args:
        force_array: shape (rows, cols, 3), forces in N [fn, ftx, fty]
        config:      SensorConfig instance
        temperature: temperature in °C (Phase 1: pass 0.0)

    Returns:
        np.ndarray shape (sensor_dim,):
            [array_distributed_force(rows*cols*3), resultant_3D(3), temperature(1)]
    """
    # Phase 1: linear Hall mapping
    array_out = force_array * config.hall_sensitivity    # (rows, cols, 3)

    # Resultant in sensor-local frame: [Fn_sum, Ftx_sum, Fty_sum]
    resultant = array_out.sum(axis=(0, 1))               # (3,)

    # Temperature (Phase 1: pass-through)
    temp = np.array([temperature], dtype=float)          # (1,)

    return np.concatenate([
        array_out.reshape(-1),   # (rows*cols*3,)
        resultant,               # (3,)
        temp,                    # (1,)
    ])
