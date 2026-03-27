"""EMTactileIsaacEnv — Isaac Sim counterpart of mujoco/env.py.

Public API is intentionally identical to EMTactileEnv so downstream code
can swap backends by changing one import line.
"""
from __future__ import annotations

import numpy as np

from ..core.sensor_config import SensorConfig
from .contact_source import ContactSource, RigidContactViewSource
from .sensor_bridge import SensorBridge
from ._compat import get_world_class


class EMTactileIsaacEnv:
    """High-level EM tactile sensor interface for Isaac Sim.

    Usage (standalone script):
        cfg = SensorConfig()
        env = EMTactileIsaacEnv("isaac/models/em_sensor_flat.usda", cfg)
        env.setup()
        for _ in range(600):
            env.step()
            tactile = env.get_tactile()   # (7, 7, 3)  ← same as MuJoCo env
        env.close()
    """

    def __init__(
        self,
        usd_path: str,
        config: SensorConfig | None = None,
        pad_prim_path: str = "/World/sensor_body/sensor_pad",
        contact_source: ContactSource | None = None,
        use_rerun: bool = False,
    ) -> None:
        self._cfg = config or SensorConfig()
        self._usd_path = usd_path
        self._pad_prim_path = pad_prim_path
        self._use_rerun = use_rerun
        self._world = None

        source = contact_source or RigidContactViewSource(self._cfg)
        self._bridge = SensorBridge(self._cfg, source)
        self._source = source

        if use_rerun:
            try:
                import rerun as rr  # type: ignore[import]
                rr.init("em_tactile_isaac", spawn=True)
                self._rr = rr
            except ImportError:
                print("[EMTactileIsaacEnv] rerun not installed; disabling.")
                self._use_rerun = False

    # ── Simulation control ────────────────────────────────────────────────

    def setup(self) -> None:
        """Initialize Isaac Sim World, load USD, register physics callback."""
        World = get_world_class()
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()
        self._world.scene.add_reference_to_stage(
            usd_path=self._usd_path, prim_path="/World"
        )
        self._source.initialize(self._world, self._pad_prim_path)
        self._world.add_physics_callback(
            "em_tactile_sensor", self._on_physics_step
        )
        self._world.reset()

    def step(self) -> None:
        """Advance simulation by one timestep."""
        if self._world is None:
            raise RuntimeError("Call setup() before step().")
        self._world.step(render=False)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        if self._world is None:
            raise RuntimeError("Call setup() before reset().")
        self._world.reset()

    def close(self) -> None:
        """Shutdown Isaac Sim world."""
        if self._world is not None:
            self._world.stop()
            self._world = None

    # ── Sensor readout (identical signatures to mujoco/env.py) ───────────

    def get_tactile(self) -> np.ndarray:
        """Array distributed force. Shape: (rows, cols, 3)  → [fn, ftx, fty] in N."""
        # bridge.output already returns a copy; reshape returns a view of that copy.
        return self._bridge.output[:self._cfg.array_dim].reshape(
            self._cfg.rows, self._cfg.cols, 3
        )

    def get_resultant(self) -> np.ndarray:
        """Resultant force [Fn_sum, Ftx_sum, Fty_sum] in N. Shape: (3,)."""
        # bridge.output already returns a copy.
        out = self._bridge.output
        return out[self._cfg.array_dim: self._cfg.array_dim + 3]

    def get_temperature(self) -> float:
        """Temperature in °C (Phase 2: always 0.0)."""
        return float(self._bridge.output[self._cfg.array_dim + 3])

    def get_tactile_flat(self) -> np.ndarray:
        """Full sensor output vector. Shape: (sensor_dim,)."""
        return self._bridge.output

    # ── Internal ──────────────────────────────────────────────────────────

    def _on_physics_step(self, step_size: float) -> None:
        self._bridge.update()
        if self._use_rerun:
            r = self.get_resultant()
            self._rr.log("sensors/fn_sum",  self._rr.Scalar(float(r[0])))
            self._rr.log("sensors/ftx_sum", self._rr.Scalar(float(r[1])))
            self._rr.log("sensors/fty_sum", self._rr.Scalar(float(r[2])))
