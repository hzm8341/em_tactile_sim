# Isaac Sim Phase 2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 core/ 层的前提下，为 em_tactile_sim 添加 Isaac Sim (PhysX) 仿真后端，输出与 MuJoCo 层完全相同的传感器接口，并提供带 7×7 热力图的 Extension UI。

**Architecture:** `ContactSource` 抽象接口隔离 PhysX 读取方式；`SensorBridge` 将接触数据转换为 core 层调用；`EMTactileIsaacEnv` 对外提供与 `EMTactileEnv`（MuJoCo）相同的签名；`extension.py` 是零业务逻辑的 IExt UI 壳。版本差异集中在 `_compat.py`。

**Tech Stack:** Python 3.10+, Isaac Sim 4.5.0/5.0.0 (omni.kit, isaacsim.core.prims), OpenUSD (USDA 文本格式), omni.ui, numpy, rerun (可选)

---

## File Map

| 操作 | 路径 | 职责 |
|------|------|------|
| 创建 | `em_tactile_sim/isaac/__init__.py` | 包初始化，导出公共符号 |
| 创建 | `em_tactile_sim/isaac/_compat.py` | 版本检测，屏蔽 4.5.0/5.0.0 差异 |
| 创建 | `em_tactile_sim/isaac/contact_source.py` | `ContactSource` ABC + `RigidContactViewSource` |
| 创建 | `em_tactile_sim/isaac/sensor_bridge.py` | 接触 dict → core 层 → (sensor_dim,) buffer |
| 创建 | `em_tactile_sim/isaac/env.py` | `EMTactileIsaacEnv`，用户主接口 |
| 创建 | `em_tactile_sim/isaac/extension.py` | IExt UI 壳，7×7 热力图 |
| 创建 | `em_tactile_sim/isaac/models/em_sensor_flat.usda` | USD 资产（USDA ASCII 格式） |
| 创建 | `tests/test_isaac_sensor_bridge.py` | SensorBridge 单元测试（不依赖 Isaac Sim） |
| 创建 | `tests/test_isaac_integration.py` | 集成测试（importorskip 跳过） |
| 创建 | `examples/isaac_press_test.py` | 独立脚本示例 |
| 修改 | `docs/simulation_guide.md` | 补充 Phase 2 运行说明 |

---

## Task 1: 包骨架 + `_compat.py`

**Files:**
- Create: `em_tactile_sim/isaac/__init__.py`
- Create: `em_tactile_sim/isaac/_compat.py`
- Create: `em_tactile_sim/isaac/models/.gitkeep`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p /media/hzm/data_disk/tactiSim_all/em_tactile_sim/em_tactile_sim/isaac/models
touch /media/hzm/data_disk/tactiSim_all/em_tactile_sim/em_tactile_sim/isaac/models/.gitkeep
```

- [ ] **Step 2: 写 `_compat.py`**

```python
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
```

- [ ] **Step 3: 写 `__init__.py`**

```python
# em_tactile_sim/isaac/__init__.py
"""Isaac Sim (PhysX) backend for em_tactile_sim — Phase 2."""
from ._compat import is_isaac_available, get_isaac_version

__all__ = ["is_isaac_available", "get_isaac_version"]
```

- [ ] **Step 4: 验证文件存在**

```bash
ls em_tactile_sim/isaac/
python3 -c "from em_tactile_sim.isaac import is_isaac_available; print('available:', is_isaac_available())"
```

期望输出：`available: False`（未安装 Isaac Sim 时）

- [ ] **Step 5: Commit**

```bash
git add em_tactile_sim/isaac/
git commit -m "feat(isaac): add package skeleton and _compat version detection"
```

---

## Task 2: `ContactSource` 抽象接口

**Files:**
- Create: `em_tactile_sim/isaac/contact_source.py`

- [ ] **Step 1: 写 `contact_source.py`**

```python
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
```

- [ ] **Step 2: 验证可导入（无 Isaac Sim 时 ContactSource ABC 仍可用）**

```bash
python3 -c "
from em_tactile_sim.isaac.contact_source import ContactSource, RigidContactViewSource
print('ContactSource ABC:', ContactSource)
print('RigidContactViewSource:', RigidContactViewSource)
"
```

期望输出：两行正常打印类名，无报错。

- [ ] **Step 3: Commit**

```bash
git add em_tactile_sim/isaac/contact_source.py
git commit -m "feat(isaac): add ContactSource ABC and RigidContactViewSource"
```

---

## Task 3: `SensorBridge` + 单元测试（TDD，不依赖 Isaac Sim）

**Files:**
- Create: `em_tactile_sim/isaac/sensor_bridge.py`
- Create: `tests/test_isaac_sensor_bridge.py`

- [ ] **Step 1: 写失败测试 `tests/test_isaac_sensor_bridge.py`**

```python
# tests/test_isaac_sensor_bridge.py
"""Unit tests for SensorBridge — no Isaac Sim required (FakeContactSource)."""
import numpy as np
import pytest

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.isaac.contact_source import ContactSource


# ── FakeContactSource ────────────────────────────────────────────────────────

class FakeContactSource(ContactSource):
    """Test double: returns a fixed contact list, no Isaac Sim needed."""

    def __init__(self, contacts: list[dict] | None = None) -> None:
        self._contacts = contacts or []

    def initialize(self, world, pad_prim_path: str) -> None:
        pass  # no-op for testing

    def get_contacts(self) -> list[dict]:
        return self._contacts


def _contact(pos=(0.0, 0.0), fn=2.0, ftx=0.0, fty=0.0, radius=0.005):
    return {
        "pos":    np.array(pos),
        "fn":     fn,
        "ft":     np.array([ftx, fty]),
        "radius": radius,
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> SensorConfig:
    return SensorConfig()


# ── Tests ────────────────────────────────────────────────────────────────────

def test_bridge_output_shape(cfg):
    from em_tactile_sim.isaac.sensor_bridge import SensorBridge
    bridge = SensorBridge(cfg, FakeContactSource())
    bridge.update()
    assert bridge.output.shape == (cfg.sensor_dim,)   # (151,)


def test_bridge_zero_on_no_contact(cfg):
    from em_tactile_sim.isaac.sensor_bridge import SensorBridge
    bridge = SensorBridge(cfg, FakeContactSource([]))
    bridge.update()
    np.testing.assert_array_equal(bridge.output, 0.0)


def test_bridge_nonzero_on_contact(cfg):
    from em_tactile_sim.isaac.sensor_bridge import SensorBridge
    bridge = SensorBridge(cfg, FakeContactSource([_contact()]))
    bridge.update()
    assert bridge.output.max() > 0.0


def test_bridge_fn_clamped_to_range(cfg):
    from em_tactile_sim.isaac.sensor_bridge import SensorBridge
    bridge = SensorBridge(cfg, FakeContactSource([_contact(fn=cfg.fn_max * 3)]))
    bridge.update()
    fn_data = bridge.output[:cfg.array_dim].reshape(cfg.rows, cfg.cols, 3)[:, :, 0]
    assert fn_data.max() <= cfg.fn_max + 1e-9


def test_bridge_update_idempotent(cfg):
    from em_tactile_sim.isaac.sensor_bridge import SensorBridge
    bridge = SensorBridge(cfg, FakeContactSource([_contact()]))
    bridge.update()
    out1 = bridge.output.copy()
    bridge.update()
    out2 = bridge.output.copy()
    np.testing.assert_array_equal(out1, out2)


def test_bridge_multi_contact_superposition(cfg):
    from em_tactile_sim.isaac.sensor_bridge import SensorBridge
    c1 = _contact(pos=(0.0, 0.0), fn=1.0)
    c2 = _contact(pos=(2e-3, 0.0), fn=1.0)

    b1 = SensorBridge(cfg, FakeContactSource([c1]))
    b1.update()
    b2 = SensorBridge(cfg, FakeContactSource([c2]))
    b2.update()
    both = SensorBridge(cfg, FakeContactSource([c1, c2]))
    both.update()

    np.testing.assert_array_almost_equal(both.output, b1.output + b2.output)


def test_bridge_output_copy_is_independent(cfg):
    from em_tactile_sim.isaac.sensor_bridge import SensorBridge
    bridge = SensorBridge(cfg, FakeContactSource([_contact()]))
    bridge.update()
    out = bridge.output
    out[:] = 0.0
    assert bridge.output.max() > 0.0   # internal buffer not affected
```

- [ ] **Step 2: 运行测试确认全部 FAIL（SensorBridge 尚未存在）**

```bash
cd /media/hzm/data_disk/tactiSim_all/em_tactile_sim
python3 -m pytest tests/test_isaac_sensor_bridge.py -v 2>&1 | tail -15
```

期望：`ImportError: cannot import name 'SensorBridge'`

- [ ] **Step 3: 写 `sensor_bridge.py`**

```python
# em_tactile_sim/isaac/sensor_bridge.py
"""Core translation layer: ContactSource → hall_response output buffer.

SensorBridge is the Isaac Sim equivalent of EMSensorCallback in mujoco/callback.py.
It calls ContactSource.get_contacts(), feeds results into ContactModel, then
hall_response — the same core functions used by the MuJoCo backend.
"""
from __future__ import annotations

import numpy as np

from ..core.sensor_config import SensorConfig
from ..core.contact_model import ContactModel
from ..core import hall_response
from .contact_source import ContactSource


class SensorBridge:
    """ContactSource → (sensor_dim,) output buffer.

    Call update() every physics step, then read output property.
    Thread-safety: not thread-safe; call from physics callback only.
    """

    def __init__(self, config: SensorConfig, source: ContactSource) -> None:
        self._cfg = config
        self._source = source
        self._contact_model = ContactModel(config)
        self._buffer = np.zeros(config.sensor_dim, dtype=np.float64)

    def update(self) -> None:
        """Fetch contacts from source, run core pipeline, update buffer."""
        contacts = self._source.get_contacts()
        force_array = self._contact_model.compute_multi(contacts)   # (rows,cols,3)
        self._buffer[:] = hall_response.compute_output(force_array, self._cfg)

    @property
    def output(self) -> np.ndarray:
        """Current sensor output, shape (sensor_dim,). Returns a copy."""
        return self._buffer.copy()
```

- [ ] **Step 4: 运行测试确认全部 PASS**

```bash
python3 -m pytest tests/test_isaac_sensor_bridge.py -v
```

期望：`7 passed`

- [ ] **Step 5: 运行全量测试确认无回归**

```bash
python3 -m pytest tests/ -q
```

期望：`57 passed, 1 warning`（原 50 + 新 7）

- [ ] **Step 6: Commit**

```bash
git add em_tactile_sim/isaac/sensor_bridge.py tests/test_isaac_sensor_bridge.py
git commit -m "feat(isaac): add SensorBridge + 7 unit tests (no Isaac Sim needed)"
```

---

## Task 4: `EMTactileIsaacEnv` 用户接口

**Files:**
- Create: `em_tactile_sim/isaac/env.py`

- [ ] **Step 1: 写 `env.py`**

```python
# em_tactile_sim/isaac/env.py
"""EMTactileIsaacEnv — Isaac Sim counterpart of mujoco/env.py.

Public API is intentionally identical to EMTactileEnv so downstream code
can swap backends by changing one import line.
"""
from __future__ import annotations

import numpy as np

from ..core.sensor_config import SensorConfig
from ..utils.recorder import DataRecorder
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
        if self._world is not None:
            self._world.reset()

    def close(self) -> None:
        """Shutdown Isaac Sim world."""
        if self._world is not None:
            self._world.stop()
            self._world = None

    # ── Sensor readout (identical signatures to mujoco/env.py) ───────────

    def get_tactile(self) -> np.ndarray:
        """Array distributed force. Shape: (rows, cols, 3) → [fn, ftx, fty] in N."""
        return self._bridge.output[:self._cfg.array_dim].reshape(
            self._cfg.rows, self._cfg.cols, 3
        ).copy()

    def get_resultant(self) -> np.ndarray:
        """Resultant force [Fn_sum, Ftx_sum, Fty_sum] in N. Shape: (3,)."""
        out = self._bridge.output
        return out[self._cfg.array_dim: self._cfg.array_dim + 3].copy()

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
```

- [ ] **Step 2: 验证可导入（不调用 setup，无 Isaac Sim 也 OK）**

```bash
python3 -c "
from em_tactile_sim.isaac.env import EMTactileIsaacEnv
from em_tactile_sim.core.sensor_config import SensorConfig
cfg = SensorConfig()
# 只构造，不调用 setup()（会触发 Isaac Sim）
env = EMTactileIsaacEnv.__new__(EMTactileIsaacEnv)
print('EMTactileIsaacEnv importable:', EMTactileIsaacEnv)
"
```

期望：打印类名，无报错。

- [ ] **Step 3: Commit**

```bash
git add em_tactile_sim/isaac/env.py
git commit -m "feat(isaac): add EMTactileIsaacEnv with DataRecorder and rerun support"
```

---

## Task 5: USD 资产（`em_sensor_flat.usda`）

**Files:**
- Create: `em_tactile_sim/isaac/models/em_sensor_flat.usda`

- [ ] **Step 1: 写 USDA 文件**

```usda
#usda 1.0
(
    """
    EM Tactile Sensor — Standard (7×7, 43×28×8mm)
    Sensing area: 6×6mm  (7 cells × 1mm spacing)
    Equivalent of em_sensor_flat.xml for Isaac Sim.
    """
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{
    # ── Physics scene ─────────────────────────────────────────────────────
    def PhysicsScene "physicsScene"
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }

    # ── Ground plane ──────────────────────────────────────────────────────
    def Plane "groundPlane"
    {
        uniform token axis = "Z"
        bool physics:collisionEnabled = true
    }

    # ── Sensor body (fixed) ───────────────────────────────────────────────
    def Xform "sensor_body"
    {
        double3 xformOp:translate = (0, 0, 0.001)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cube "sensor_pad"
        {
            # Half-extents: 3mm × 3mm × 1mm  (sensing area 6mm × 6mm, depth 2mm)
            double size = 1.0
            float3 xformOp:scale = (0.003, 0.003, 0.001)
            uniform token[] xformOpOrder = ["xformOp:scale"]

            bool physics:collisionEnabled = true
            bool physics:rigidBodyEnabled = false   # fixed body

            float3 primvars:displayColor = (0.2, 0.6, 0.9)
        }
    }

    # ── Test object: free-falling sphere (r=5mm, m=10g) ──────────────────
    def Sphere "ball"
    {
        double radius = 0.005
        double3 xformOp:translate = (0, 0, 0.05)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        bool physics:collisionEnabled = true
        bool physics:rigidBodyEnabled = true
        float physics:mass = 0.01

        float3 primvars:displayColor = (1.0, 0.3, 0.3)
    }
}
```

保存到 `em_tactile_sim/isaac/models/em_sensor_flat.usda`。

- [ ] **Step 2: 验证文件存在**

```bash
ls -lh em_tactile_sim/isaac/models/
head -5 em_tactile_sim/isaac/models/em_sensor_flat.usda
```

期望：文件存在，首行为 `#usda 1.0`。

- [ ] **Step 3: Commit**

```bash
git add em_tactile_sim/isaac/models/em_sensor_flat.usda
git commit -m "feat(isaac): add em_sensor_flat.usda USD asset (7x7 standard)"
```

---

## Task 6: `extension.py` — IExt UI 壳（7×7 热力图）

**Files:**
- Create: `em_tactile_sim/isaac/extension.py`

- [ ] **Step 1: 写 `extension.py`**

```python
# em_tactile_sim/isaac/extension.py
"""Isaac Sim IExt extension entry point.

This is a thin UI shell — all sensor logic lives in env.py.
UI layout:
  - [Step] [Reset] control buttons
  - Resultant force: Fn / Ftx / Fty float labels
  - 7×7 normal-force grid (49 FloatField widgets, 0~20N colour-mapped)
"""
from __future__ import annotations
import os
import numpy as np

import omni.ext                        # type: ignore[import]
import omni.ui as ui                   # type: ignore[import]

from ..core.sensor_config import SensorConfig
from .env import EMTactileIsaacEnv

_USD_PATH = os.path.join(
    os.path.dirname(__file__), "models", "em_sensor_flat.usda"
)
_WINDOW_TITLE = "EM Tactile Sensor"
_HEATMAP_CELL_PX = 20   # pixel width/height of each heatmap cell


class EMTactileExtension(omni.ext.IExt):
    """Isaac Sim Extension: EM tactile sensor simulation with 7×7 heatmap."""

    def on_startup(self, ext_id: str) -> None:
        self._cfg = SensorConfig()
        self._env = EMTactileIsaacEnv(_USD_PATH, self._cfg)
        self._env.setup()

        self._fn_fields: list[ui.FloatField] = []
        self._result_labels: dict[str, ui.Label] = {}
        self._window = ui.Window(_WINDOW_TITLE, width=480, height=360)
        self._build_ui()

    def on_shutdown(self) -> None:
        self._env.close()
        self._window = None

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        with self._window.frame:
            with ui.VStack(spacing=6):
                self._build_controls()
                self._build_resultant_panel()
                self._build_heatmap()

    def _build_controls(self) -> None:
        with ui.HStack(height=30, spacing=4):
            ui.Button("Step",  clicked_fn=self._on_step,  width=80)
            ui.Button("Reset", clicked_fn=self._on_reset, width=80)

    def _build_resultant_panel(self) -> None:
        with ui.HStack(height=24, spacing=8):
            for key in ("Fn_sum", "Ftx_sum", "Fty_sum"):
                ui.Label(f"{key}:", width=60)
                lbl = ui.Label("0.000 N", width=80)
                self._result_labels[key] = lbl

    def _build_heatmap(self) -> None:
        """7×7 grid of FloatField widgets showing fn per cell (N)."""
        rows, cols = self._cfg.rows, self._cfg.cols
        cell_px = _HEATMAP_CELL_PX
        with ui.VStack(spacing=1):
            for i in range(rows):
                with ui.HStack(spacing=1, height=cell_px):
                    for j in range(cols):
                        field = ui.FloatField(
                            width=cell_px * 2, height=cell_px,
                            read_only=True,
                        )
                        field.model.set_value(0.0)
                        self._fn_fields.append(field)

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_step(self) -> None:
        self._env.step()
        self._refresh_ui()

    def _on_reset(self) -> None:
        self._env.reset()
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        """Pull latest sensor data and update all UI widgets."""
        tactile   = self._env.get_tactile()     # (7, 7, 3)
        resultant = self._env.get_resultant()   # (3,)

        # Update resultant labels
        keys = ("Fn_sum", "Ftx_sum", "Fty_sum")
        for key, val in zip(keys, resultant):
            self._result_labels[key].text = f"{val:.3f} N"

        # Update 7×7 heatmap
        fn_map = tactile[:, :, 0]               # (7, 7)
        for idx, field in enumerate(self._fn_fields):
            i, j = divmod(idx, self._cfg.cols)
            field.model.set_value(float(fn_map[i, j]))
```

- [ ] **Step 2: 验证可导入（不实例化，避免触发 Isaac Sim）**

```bash
python3 -c "
import ast, sys
src = open('em_tactile_sim/isaac/extension.py').read()
ast.parse(src)
print('extension.py syntax OK')
"
```

期望：`extension.py syntax OK`

- [ ] **Step 3: Commit**

```bash
git add em_tactile_sim/isaac/extension.py
git commit -m "feat(isaac): add IExt UI shell with 7x7 normal-force heatmap"
```

---

## Task 7: 集成测试 + 示例脚本

**Files:**
- Create: `tests/test_isaac_integration.py`
- Create: `examples/isaac_press_test.py`

- [ ] **Step 1: 写集成测试 `tests/test_isaac_integration.py`**

```python
# tests/test_isaac_integration.py
"""Isaac Sim integration tests — auto-skipped if isaacsim not installed."""
import numpy as np
import pytest

isaacsim = pytest.importorskip(
    "isaacsim",
    reason="Isaac Sim not installed; skipping integration tests.",
)

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.isaac.env import EMTactileIsaacEnv

import os
USD = os.path.join(
    os.path.dirname(__file__),
    "../em_tactile_sim/isaac/models/em_sensor_flat.usda",
)


@pytest.fixture(scope="module")
def env():
    cfg = SensorConfig()
    e = EMTactileIsaacEnv(USD, cfg)
    e.setup()
    yield e
    e.close()


def test_env_setup_succeeds(env):
    assert env is not None


def test_output_shapes_after_step(env):
    env.step()
    assert env.get_tactile().shape      == (7, 7, 3)
    assert env.get_resultant().shape    == (3,)
    assert env.get_tactile_flat().shape == (151,)


def test_no_contact_at_start(env):
    env.reset()
    env.step()
    tactile = env.get_tactile()
    np.testing.assert_array_almost_equal(tactile[:, :, 0], 0.0)


def test_contact_produces_nonzero_after_fall(env):
    env.reset()
    for _ in range(300):   # ~2.5s @ 120Hz, ball falls 5cm
        env.step()
    fn_max = env.get_tactile()[:, :, 0].max()
    assert fn_max > 0.0, "Expected nonzero normal force after ball contact"


def test_reset_clears_output(env):
    for _ in range(300):
        env.step()
    env.reset()
    env.step()
    fn_max = env.get_tactile()[:, :, 0].max()
    assert fn_max == pytest.approx(0.0, abs=1e-6)


def test_get_temperature_zero(env):
    env.step()
    assert env.get_temperature() == pytest.approx(0.0)
```

- [ ] **Step 2: 运行测试确认正确跳过**

```bash
python3 -m pytest tests/test_isaac_integration.py -v
```

期望：`SKIPPED ... isaacsim not installed`（无 Isaac Sim 环境）

- [ ] **Step 3: 写示例脚本 `examples/isaac_press_test.py`**

```python
"""
Isaac Sim 球落压力测试脚本。

运行方式（Isaac Sim 环境）:
    ~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh examples/isaac_press_test.py
    # 或 isaac-sim-5.0.0

无 Isaac Sim 时此脚本无法运行（会报 ImportError）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.isaac.env import EMTactileIsaacEnv
from em_tactile_sim.utils.recorder import DataRecorder

USD = os.path.join(
    os.path.dirname(__file__),
    "../em_tactile_sim/isaac/models/em_sensor_flat.usda",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main() -> None:
    cfg = SensorConfig()
    env = EMTactileIsaacEnv(USD, cfg, use_rerun=False)
    env.setup()
    recorder = DataRecorder(cfg)

    n_steps = 600   # 5s @ 120Hz
    print(f"仿真 {n_steps} 步（5s @ 120Hz）...")

    for i in range(n_steps):
        env.step()
        # Isaac Sim world time via world.current_time (float seconds)
        t = env._world.current_time if env._world else i / cfg.sample_rate
        recorder.record(t, env.get_tactile_flat())
        if (i + 1) % 100 == 0:
            print(f"  步数: {i+1}/{n_steps}")

    csv_path = os.path.join(OUTPUT_DIR, "isaac_sensor_timeseries.csv")
    png_path = os.path.join(OUTPUT_DIR, "isaac_resultant_timeseries.png")
    recorder.save_csv(csv_path)
    recorder.plot_resultant(png_path)

    r = recorder.resultants
    fn_max = float(r[:, 0].max()) if len(r) else 0.0
    print(f"\n总帧数: {recorder.n_frames}")
    print(f"最大法向合力: {fn_max:.4f} N")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 验证示例脚本语法**

```bash
python3 -c "import ast; ast.parse(open('examples/isaac_press_test.py').read()); print('syntax OK')"
```

- [ ] **Step 5: 运行全量测试（集成测试自动跳过）**

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

期望：`57 passed, 1 skipped, 1 warning`（57 已有 + 1 跳过的集成测试）

- [ ] **Step 6: Commit**

```bash
git add tests/test_isaac_integration.py examples/isaac_press_test.py
git commit -m "feat(isaac): add integration tests (auto-skip) and isaac_press_test example"
```

---

## Task 8: 更新文档 + 最终提交

**Files:**
- Modify: `docs/simulation_guide.md`

- [ ] **Step 1: 在 `docs/simulation_guide.md` 第三节替换 Phase 2 内容**

找到 `## 三、Isaac Sim 集成（Phase 2，开发中）` 一节，替换为：

```markdown
## 三、Isaac Sim 集成（Phase 2）

### 3.1 模块结构

```
em_tactile_sim/isaac/
├── _compat.py           # 版本检测（4.5.0 / 5.0.0）
├── contact_source.py    # ContactSource ABC + RigidContactViewSource
├── sensor_bridge.py     # 接触数据 → core 层转换
├── env.py               # EMTactileIsaacEnv，与 MuJoCo env 接口相同
├── extension.py         # IExt UI 壳（7×7 热力图 + 合力显示）
└── models/
    └── em_sensor_flat.usda  # USD 传感器资产
```

### 3.2 独立 Python 脚本（需 Isaac Sim Python 解释器）

```bash
# Isaac Sim 4.5.0
~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh examples/isaac_press_test.py

# Isaac Sim 5.0.0
~/.local/share/ov/pkg/isaac-sim-5.0.0/python.sh examples/isaac_press_test.py
```

### 3.3 Extension（图形界面插件）

在 Isaac Sim 中加载 Extension：

1. 打开 Isaac Sim → **Window → Extensions**
2. 点击 **+** → 指定路径：`em_tactile_sim/isaac/`
3. 启用 `EMTactileExtension` → UI 面板弹出
4. 点击 **Step** / **Reset** 控制仿真，7×7 法向力热力图实时更新

### 3.4 API 用法（与 MuJoCo 接口一致）

```python
from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.isaac.env import EMTactileIsaacEnv

cfg = SensorConfig()
env = EMTactileIsaacEnv("em_tactile_sim/isaac/models/em_sensor_flat.usda", cfg)
env.setup()
env.step()

tactile   = env.get_tactile()      # (7, 7, 3) — 与 MuJoCo 完全相同
resultant = env.get_resultant()    # (3,) [Fn_sum, Ftx_sum, Fty_sum]
env.close()
```

### 3.5 检测 Isaac Sim 是否可用

```bash
python3 -c "from em_tactile_sim.isaac import is_isaac_available; print(is_isaac_available())"
```
```

- [ ] **Step 2: 验证文档写入**

```bash
grep -n "Isaac Sim 4.5.0" docs/simulation_guide.md
```

期望：找到对应行

- [ ] **Step 3: 运行最终全量测试**

```bash
python3 -m pytest tests/ -q
```

期望：`57 passed, 1 skipped, 1 warning`

- [ ] **Step 4: 最终 Commit**

```bash
git add docs/simulation_guide.md em_tactile_sim/isaac/__init__.py
git commit -m "docs: update simulation_guide with Phase 2 Isaac Sim instructions"
```

---

## 自审检查结果

**Spec 覆盖：**
- ✅ `_compat.py` 版本检测 → Task 1
- ✅ `ContactSource` ABC + `RigidContactViewSource` → Task 2
- ✅ `SensorBridge` + 单元测试（FakeContactSource）→ Task 3
- ✅ `EMTactileIsaacEnv`（与 MuJoCo env 接口对齐）→ Task 4
- ✅ USD 资产 → Task 5
- ✅ `extension.py` IExt + 7×7 热力图 → Task 6
- ✅ 集成测试（importorskip）+ 示例脚本 → Task 7
- ✅ 文档更新 → Task 8
- ✅ rerun 可选支持（`use_rerun=True`）→ Task 4 env.py

**类型一致性：**
- `ContactSource.get_contacts()` 返回格式在 Task 2/3 定义，Task 3 的测试和 Task 4 的 env.py 均使用同一格式
- `SensorBridge.output` 在 Task 3 定义（返回 copy），Task 4 的 env.py `get_tactile_flat()` 直接使用

**无占位符：** 所有步骤均含具体代码和命令。
