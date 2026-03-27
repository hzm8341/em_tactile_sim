# em_tactile_sim/isaac/_compat.py
"""Isaac Sim version detection and API compatibility layer.

All Isaac Sim imports in the package go through this module.
Other modules never import isaacsim directly.
"""
from __future__ import annotations


def get_isaac_version() -> str | None:
    """Return Isaac Sim version string, or None if not installed."""
    try:
        import omni.kit.app
        return omni.kit.app.get_app().get_app_version()
    except (ImportError, AttributeError):
        return None


def is_isaac_available() -> bool:
    """Return True if Isaac Sim is installed and accessible."""
    return get_isaac_version() is not None


def get_rigid_contact_view_class():
    """Return the correct RigidContactView class for the installed version.

    Raises RuntimeError if Isaac Sim is not available or version unsupported.
    """
    version = get_isaac_version()
    if version in ("4.5.0", "5.0.0"):
        from isaacsim.core.prims import RigidContactView  # type: ignore[import]
        return RigidContactView
    if version is None:
        raise RuntimeError("Isaac Sim is not installed.")
    raise RuntimeError(f"Unsupported Isaac Sim version: {version!r}")


def get_world_class():
    """Return World class for the installed Isaac Sim version."""
    version = get_isaac_version()
    if version in ("4.5.0", "5.0.0"):
        from isaacsim.core.api import World  # type: ignore[import]
        return World
    raise RuntimeError(f"Unsupported Isaac Sim version: {version!r}")
