"""SensorBridge: ContactSource → core layer → (sensor_dim,) output buffer."""
import numpy as np
from ..core.sensor_config import SensorConfig
from ..core import contact_model, hall_response
from .contact_source import ContactSource


class SensorBridge:
    """Connects a ContactSource to the core layer, maintaining a (sensor_dim,) output buffer.

    Usage:
        bridge = SensorBridge(config, source)
        bridge.update()          # call each physics step
        data = bridge.output     # numpy array, shape (sensor_dim,)
    """

    def __init__(self, config: SensorConfig, source: ContactSource) -> None:
        self._contact_model = contact_model.ContactModel(config)
        self._source = source
        self._cfg = config
        self._buffer = np.zeros(config.sensor_dim, dtype=np.float64)

    def update(self) -> None:
        """Pull contacts from source, run core pipeline, update internal buffer."""
        contacts = self._source.get_contacts()
        force_array = self._contact_model.compute_multi(contacts)
        self._buffer[:] = hall_response.compute_output(force_array, self._cfg)

    @property
    def output(self) -> np.ndarray:
        """Current sensor output. Shape: (sensor_dim,). Returns a copy."""
        return self._buffer.copy()
