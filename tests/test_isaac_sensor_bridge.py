import numpy as np
import pytest
from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.isaac.contact_source import ContactSource
from em_tactile_sim.isaac.sensor_bridge import SensorBridge


class FakeContactSource(ContactSource):
    """Test double — no Isaac Sim needed."""
    def __init__(self, contacts: list[dict]):
        self._contacts = contacts
    def initialize(self, world, pad_prim_path: str) -> None:
        pass
    def get_contacts(self) -> list[dict]:
        return self._contacts


def _single_contact(fn: float = 2.0) -> dict:
    return {"pos": np.array([0.0, 0.0]), "fn": fn,
            "ft": np.array([0.0, 0.0]), "radius": 0.005}


def test_bridge_output_shape():
    cfg = SensorConfig()
    bridge = SensorBridge(cfg, FakeContactSource([_single_contact()]))
    bridge.update()
    assert bridge.output.shape == (cfg.sensor_dim,)


def test_bridge_zero_on_no_contact():
    cfg = SensorConfig()
    bridge = SensorBridge(cfg, FakeContactSource([]))
    bridge.update()
    np.testing.assert_array_equal(bridge.output, 0.0)


def test_bridge_nonzero_on_contact():
    cfg = SensorConfig()
    bridge = SensorBridge(cfg, FakeContactSource([_single_contact(fn=5.0)]))
    bridge.update()
    assert np.any(bridge.output != 0.0)


def test_bridge_fn_clamped_to_range():
    cfg = SensorConfig()
    bridge = SensorBridge(cfg, FakeContactSource([_single_contact(fn=5.0)]))
    bridge.update()
    out = bridge.output
    assert np.all(out >= 0.0)
    assert np.all(out <= cfg.fn_max * cfg.hall_sensitivity * 10)  # generous upper bound


def test_bridge_update_idempotent():
    cfg = SensorConfig()
    contacts = [_single_contact(fn=3.0)]
    bridge = SensorBridge(cfg, FakeContactSource(contacts))
    bridge.update()
    out1 = bridge.output
    bridge.update()
    out2 = bridge.output
    np.testing.assert_array_equal(out1, out2)


def test_bridge_multi_contact_superposition():
    cfg = SensorConfig()
    # Two contacts at opposite ends of the sensor array (sensing_span_x ~= 6 mm, spans +-3mm)
    # Use positions near opposite corners so each activates different cells.
    c1 = {"pos": np.array([-0.002, 0.0]), "fn": 2.0,
          "ft": np.array([0.0, 0.0]), "radius": 0.0005}
    c2 = {"pos": np.array([0.002, 0.0]),  "fn": 2.0,
          "ft": np.array([0.0, 0.0]), "radius": 0.0005}
    bridge_single = SensorBridge(cfg, FakeContactSource([c1]))
    bridge_both   = SensorBridge(cfg, FakeContactSource([c1, c2]))
    bridge_single.update()
    bridge_both.update()
    assert bridge_both.output.sum() > bridge_single.output.sum()


def test_bridge_output_copy_is_independent():
    cfg = SensorConfig()
    bridge = SensorBridge(cfg, FakeContactSource([_single_contact()]))
    bridge.update()
    out = bridge.output
    out[:] = 0.0
    assert np.any(bridge.output != 0.0)  # internal buffer unchanged
