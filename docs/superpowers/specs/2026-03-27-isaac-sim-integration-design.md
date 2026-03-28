# Isaac Sim 集成设计规范 — Phase 2

**日期**：2026-03-27
**状态**：待实现
**作者**：用户 + Claude

---

## 1. 背景与目标

### 1.1 基础

Phase 1（MuJoCo）已完成，50 个测试全部通过。Phase 2 在不修改任何 `core/` 层代码的前提下，将 Isaac Sim（PhysX）作为仿真后端接入，输出与 MuJoCo 层完全相同的传感器接口。

### 1.2 设计约束

| 约束 | 说明 |
|------|------|
| core 层零修改 | `ContactModel`、`hall_response`、`SensorConfig` 不改动 |
| 双版本兼容 | Isaac Sim 4.5.0 和 5.0.0，通过 `_compat.py` 屏蔽差异 |
| 两种使用方式 | 独立 Python 脚本 + Isaac Sim Extension（UI 壳） |
| 接触读取可替换 | `RigidContactView` 为默认实现，预留抽象接口支持替换 |
| 可视化分层 | `DataRecorder`（无依赖，默认）+ rerun（可选，`use_rerun=True` 开关） |

---

## 2. 目录结构

```
em_tactile_sim/isaac/
├── __init__.py
├── _compat.py           # Isaac Sim 版本检测，屏蔽 4.5.0/5.0.0 API 差异
├── contact_source.py    # ContactSource ABC + RigidContactViewSource 默认实现
├── sensor_bridge.py     # 接触数据 → core 层转换（主业务逻辑）
├── env.py               # EMTactileIsaacEnv，用户主接口
└── extension.py         # Isaac Sim IExt 入口（可选 UI 壳）

em_tactile_sim/isaac/models/
└── em_sensor_flat.usd   # 传感器 USD 资产（43×28×8mm 标准品）

tests/
├── test_isaac_sensor_bridge.py   # 单元测试（不依赖 Isaac Sim）
└── test_isaac_integration.py     # 集成测试（pytest.importorskip 跳过）
```

---

## 3. 模块详细设计

### 3.1 `_compat.py` — 版本兼容层

```python
def get_isaac_version() -> str | None:
    """返回 Isaac Sim 版本字符串，未安装返回 None。"""
    try:
        import omni.kit.app
        return omni.kit.app.get_app().get_app_version()
    except ImportError:
        return None

def get_rigid_contact_view_class():
    """根据版本返回正确的 RigidContactView 类。"""
    version = get_isaac_version()
    if version in ("4.5.0", "5.0.0"):
        from isaacsim.core.prims import RigidContactView
        return RigidContactView
    raise RuntimeError(f"Unsupported Isaac Sim version: {version}")

def is_isaac_available() -> bool:
    return get_isaac_version() is not None
```

**原则**：其余所有模块只通过 `_compat` 间接访问 isaacsim，不直接 `import isaacsim`。

---

### 3.2 `contact_source.py` — 接触数据源抽象

#### 抽象基类

```python
class ContactSource(ABC):
    @abstractmethod
    def initialize(self, world, pad_prim_path: str) -> None:
        """场景初始化后调用一次。"""

    @abstractmethod
    def get_contacts(self) -> list[dict]:
        """
        每物理步调用，返回接触列表。
        dict 格式与 MuJoCo _get_pad_contacts() 完全相同：
          pos:    np.ndarray([cx, cy])   传感器局部坐标系，m
          fn:     float                  法向力，N（压入为正）
          ft:     np.ndarray([ftx,fty]) 切向力，N
          radius: float                  等效半径，m
        """
```

#### 默认实现：RigidContactViewSource

```python
class RigidContactViewSource(ContactSource):
    def initialize(self, world, pad_prim_path: str) -> None:
        RigidContactView = get_rigid_contact_view_class()
        self._view = RigidContactView(
            prim_paths_expr=pad_prim_path,
            max_contact_count=64,
        )
        world.scene.add(self._view)
        self._pad_prim_path = pad_prim_path

    def get_contacts(self) -> list[dict]:
        # 读取 net_contact_forces / contact_offsets
        # 转换为 sensor-local 坐标系
        # 返回 contact dict 列表
        ...
```

**坐标系转换**：从 PhysX 世界坐标转为传感器 pad 局部坐标（XY 平面），与 MuJoCo callback 中的 `pad_mat.T @ (world_pos - pad_pos)` 等价。

---

### 3.3 `sensor_bridge.py` — 核心转换层

```python
class SensorBridge:
    """ContactSource → core 层，维护 (sensor_dim,) 输出 buffer。"""

    def __init__(self, config: SensorConfig, source: ContactSource):
        self._contact_model = ContactModel(config)
        self._source = source
        self._cfg = config
        self._buffer = np.zeros(config.sensor_dim)

    def update(self) -> None:
        """每物理步调用一次，更新内部 buffer。"""
        contacts = self._source.get_contacts()
        force_array = self._contact_model.compute_multi(contacts)
        self._buffer[:] = hall_response.compute_output(force_array, self._cfg)

    @property
    def output(self) -> np.ndarray:
        return self._buffer.copy()
```

---

### 3.4 `env.py` — 用户主接口

```python
class EMTactileIsaacEnv:
    def __init__(
        self,
        usd_path: str,
        config: SensorConfig | None = None,
        pad_prim_path: str = "/World/sensor_body/sensor_pad",
        contact_source: ContactSource | None = None,
        use_rerun: bool = False,
    ): ...

    # ── 仿真控制 ──────────────────────────────────────────────────
    def setup(self) -> None:
        """初始化 Isaac Sim world，加载 USD，注册 physics_callback。"""

    def step(self) -> None:
        """推进一步物理仿真（world.step()）。"""

    def reset(self) -> None:
        """重置仿真状态（world.reset()）。"""

    def close(self) -> None:
        """停止 world，关闭 rerun（若启用）。"""

    # ── 传感器读取（与 mujoco/env.py 签名完全相同）────────────────
    def get_tactile(self) -> np.ndarray:
        """阵列分布力，shape (rows, cols, 3)。"""

    def get_resultant(self) -> np.ndarray:
        """合力 [Fn_sum, Ftx_sum, Fty_sum]，shape (3,)。"""

    def get_temperature(self) -> float:
        """温度（Phase 2 固定返回 0.0）。"""

    def get_tactile_flat(self) -> np.ndarray:
        """完整输出向量，shape (sensor_dim,)。"""
```

**physics_callback 注册方式：**

```python
world.add_physics_callback("em_tactile_sensor", self._on_physics_step)

def _on_physics_step(self, step_size: float) -> None:
    self._bridge.update()
    if self._use_rerun:
        self._log_rerun()
```

---

### 3.5 `extension.py` — IExt 薄壳

```python
class EMTactileExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._env = EMTactileIsaacEnv(USD_PATH, SensorConfig())
        self._env.setup()
        self._build_ui()

    def on_shutdown(self) -> None:
        self._env.close()
        self._window = None

    def _build_ui(self) -> None:
        # omni.ui：
        #   - Step / Reset 按钮
        #   - 合力三分量数值显示（Fn_sum, Ftx_sum, Fty_sum）
        #   - 7×7 法向力热力图（FloatField 网格，颜色映射 0~20N）
        ...
```

Extension 不含任何业务逻辑，所有传感器调用通过 `self._env` 代理。

---

## 4. 数据流

```
world.step()
    │
    └─ _on_physics_step(step_size)
              │
              ▼
    ContactSource.get_contacts()        ← RigidContactViewSource（默认）
              │ list[dict]（与 MuJoCo 格式相同）
              ▼
    SensorBridge.update()
              │
              ├─ ContactModel.compute_multi()   → (rows, cols, 3)
              └─ hall_response.compute_output() → (sensor_dim,)
                            │
                            ▼
              EMTactileIsaacEnv._buffer         numpy array
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
    get_tactile() / get_resultant()    DataRecorder / rerun（可选）
```

---

## 5. 测试策略

### 5.1 三层测试

| 层级 | 文件 | 依赖 Isaac Sim | 说明 |
|------|------|:-:|------|
| core（已有） | `tests/test_*.py`（50个） | 否 | 零修改 |
| isaac 单元 | `tests/test_isaac_sensor_bridge.py` | **否** | FakeContactSource 注入 |
| isaac 集成 | `tests/test_isaac_integration.py` | 是 | `pytest.importorskip("isaacsim")` |

### 5.2 FakeContactSource 模式

```python
class FakeContactSource(ContactSource):
    def __init__(self, contacts: list[dict]):
        self._contacts = contacts
    def initialize(self, world, pad_prim_path): pass
    def get_contacts(self) -> list[dict]:
        return self._contacts

# 测试用例（不依赖 Isaac Sim）
def test_bridge_output_shape():
    cfg = SensorConfig()
    source = FakeContactSource([
        {"pos": np.array([0., 0.]), "fn": 2.0,
         "ft": np.array([0., 0.]), "radius": 0.005}
    ])
    bridge = SensorBridge(cfg, source)
    bridge.update()
    assert bridge.output.shape == (cfg.sensor_dim,)

def test_bridge_zero_on_no_contact():
    cfg = SensorConfig()
    bridge = SensorBridge(cfg, FakeContactSource([]))
    bridge.update()
    np.testing.assert_array_equal(bridge.output, 0.0)
```

### 5.3 单元测试用例清单

`test_isaac_sensor_bridge.py`：
- `test_bridge_output_shape` — 输出 shape = (151,)
- `test_bridge_zero_on_no_contact` — 空接触 → 全零
- `test_bridge_nonzero_on_contact` — 有接触 → 非零
- `test_bridge_fn_in_range` — fn ∈ [0, fn_max]
- `test_bridge_update_idempotent` — 相同输入多次 update → 结果相同
- `test_bridge_multi_contact_superposition` — 两个接触叠加 ≈ 各自之和

---

## 6. USD 资产说明

`isaac/models/em_sensor_flat.usd` 对应 MuJoCo `em_sensor_flat.xml`：

| 属性 | 值 |
|------|-----|
| 传感器 prim 路径 | `/World/sensor_body/sensor_pad` |
| 产品尺寸 | 43mm × 28mm × 8mm |
| 感知区域 | 6mm × 6mm（7×7 @ 1mm） |
| 碰撞体 | Box，与 MJCF `sensor_pad` 对应 |
| 测试物体 | 自由落体球（半径 5mm，5cm 高） |

USD 文件通过 Isaac Sim 内置工具或手工编写生成，Phase 2 实现时确认。

---

## 7. 独立脚本使用示例

```python
from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.isaac.env import EMTactileIsaacEnv
from em_tactile_sim.utils.recorder import DataRecorder

cfg = SensorConfig()
env = EMTactileIsaacEnv(
    usd_path="em_tactile_sim/isaac/models/em_sensor_flat.usd",
    config=cfg,
    use_rerun=False,   # True 启用实时可视化
)
env.setup()

recorder = DataRecorder(cfg)
for _ in range(600):              # 5s @ 120Hz
    env.step()
    recorder.record(env._world.current_time, env.get_tactile_flat())

recorder.save_csv("isaac_output.csv")
recorder.plot_resultant("isaac_output.png")
env.close()
```

---

## 8. 未解决问题

| 编号 | 问题 | 状态 |
|------|------|------|
| Q1 | `RigidContactView` 在 5.0.0 中 import 路径是否有变化？ | 待验证 |
| Q2 | USD 文件手工编写 vs Isaac Sim 内置转换工具？ | Phase 2 实现时决定 |
| Q3 | rerun 集成：`rr.log` 调用频率是否需要节流（每 N 步记录一次）？ | 待设计 |
| Q4 | Extension UI：是否需要 7×7 热力图（omni.ui）还是只显示合力数值？ | **已决策：需要 7×7 热力图**，用 omni.ui FloatField 网格或 ImageWithProvider 实现 |
