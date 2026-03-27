# em_tactile_sim/isaac/__init__.py
"""Isaac Sim (PhysX) backend for em_tactile_sim — Phase 2."""
from ._compat import is_isaac_available, get_isaac_version

__all__ = ["is_isaac_available", "get_isaac_version"]
