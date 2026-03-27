# em_tactile_sim/isaac/contact_source.py
"""Contact data source abstraction for Isaac Sim (PhysX) backend.

ContactSource is the replacement for MuJoCo's _get_pad_contacts().
The returned dict format is identical, allowing core/ to be reused unchanged.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from ..core.sensor_config import SensorConfig
from ._compat import get_rigid_contact_view_class


class ContactSource(ABC):
    """Abstract contact data provider.

    Implement this to swap out the PhysX contact reading strategy.
    """

    @abstractmethod
    def initialize(self, world, pad_prim_path: str) -> None:
        """Called once after Isaac Sim world is initialized.

        Args:
            world:         Isaac Sim World instance.
            pad_prim_path: USD prim path of the sensor pad geom.
        """

    @abstractmethod
    def get_contacts(self) -> list[dict]:
        """Return contact list for current physics step.

        Each dict matches the format expected by ContactModel.compute():
            pos:    np.ndarray([cx, cy])   sensor-local coords, m
            fn:     float                  normal force, N (compression > 0)
            ft:     np.ndarray([ftx,fty]) tangential force, N
            radius: float                  contact body equivalent radius, m
        """


class RigidContactViewSource(ContactSource):
    """Default ContactSource using RigidContactView PhysX API.

    Phase 2 simplification: net contact force is used; contact position
    is approximated as pad centre. Phase 4 can improve with per-contact
    position data from get_contact_offsets().
    """

    def __init__(self, config: SensorConfig) -> None:
        self._cfg = config
        self._view = None

    def initialize(self, world, pad_prim_path: str) -> None:
        RigidContactView = get_rigid_contact_view_class()
        self._view = RigidContactView(
            prim_paths_expr=pad_prim_path,
            max_contact_count=64,
        )
        world.scene.add(self._view)

    def get_contacts(self) -> list[dict]:
        if self._view is None:
            return []

        # get_net_contact_forces returns shape (n_prims, 3) in world frame.
        # Phase 2 assumption: pad is parallel to world XY plane →
        #   world Z = sensor normal,  world X/Y = sensor tangential axes.
        forces = self._view.get_net_contact_forces(dt=1.0 / self._cfg.sample_rate)
        if forces is None or forces.shape[0] == 0:
            return []

        fx, fy, fz = float(forces[0, 0]), float(forces[0, 1]), float(forces[0, 2])
        fn = max(0.0, fz)          # normal force (upward reaction, positive)
        ftx, fty = fx, fy          # tangential forces

        if fn < self._cfg.sensitivity:
            return []

        return [{
            "pos":    np.array([0.0, 0.0]),   # pad centre (Phase 2 approximation)
            "fn":     fn,
            "ft":     np.array([ftx, fty]),
            "radius": 0.005,                  # default 5 mm sphere fallback
        }]
