import mujoco
from mujoco import viewer
import numpy as np
from .callback import EMSensorCallback
from ..core.sensor_config import SensorConfig


class EMTactileEnv:
    """
    High-level interface for the EM tactile sensor simulation.

    Usage:
        cfg = SensorConfig()
        env = EMTactileEnv("path/to/em_sensor_flat.xml", cfg)
        env.step()
        tactile = env.get_tactile()   # (7, 7, 3)
        env.close()
    """

    def __init__(self, xml_path: str, config: SensorConfig | None = None) -> None:
        self._cfg = config or SensorConfig()
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data  = mujoco.MjData(self._model)
        self._cb    = EMSensorCallback(self._model, self._cfg)
        self._cb.register()

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance simulation by one timestep (triggers mjcb_sensor callback)."""
        mujoco.mj_step(self._model, self._data)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self._model, self._data)

    def close(self) -> None:
        """Unregister callback (good practice when done)."""
        self._cb.unregister()

    # ------------------------------------------------------------------
    # Sensor readout
    # ------------------------------------------------------------------

    def get_tactile(self) -> np.ndarray:
        """Array distributed force. Shape: (rows, cols, 3) -> [fn, ftx, fty] in N."""
        adr = self._cb.sensor_adr
        return self._data.sensordata[adr: adr + self._cfg.array_dim].reshape(
            self._cfg.rows, self._cfg.cols, 3).copy()

    def get_resultant(self) -> np.ndarray:
        """3D resultant force [Fx, Fy, Fz] in N. Shape: (3,)."""
        adr = self._cb.sensor_adr + self._cfg.array_dim
        return self._data.sensordata[adr: adr + 3].copy()

    def get_temperature(self) -> float:
        """Temperature in degrees C (Phase 1 always returns 0.0)."""
        adr = self._cb.sensor_adr + self._cfg.array_dim + 3
        return float(self._data.sensordata[adr])

    def get_tactile_flat(self) -> np.ndarray:
        """Full sensor output vector. Shape: (sensor_dim,) = (151,)."""
        adr = self._cb.sensor_adr
        return self._data.sensordata[adr: adr + self._cfg.sensor_dim].copy()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Open MuJoCo passive viewer. Blocks until window is closed."""
        with viewer.launch_passive(self._model, self._data) as v:
            while v.is_running():
                self.step()
                v.sync()
