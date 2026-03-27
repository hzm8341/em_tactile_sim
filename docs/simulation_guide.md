# em_tactile_sim 仿真测试操作说明

> 版本：Phase 1（MuJoCo 完整，Isaac Sim 开发中）
> 最后更新：2026-03-27

---

## 一、环境准备

### 1.1 依赖版本（已验证）

| 包 | 版本 |
|----|------|
| Python | ≥ 3.10 |
| mujoco | 3.6.0 |
| numpy | 2.2.6 |
| matplotlib | 3.10.8 |

### 1.2 安装包

```bash
cd /media/hzm/data_disk/tactiSim_all/em_tactile_sim

# 开发模式安装（含测试依赖）
pip install -e ".[dev]"

# 验证安装
python3 -c "import em_tactile_sim; print('OK')"
```

---

## 二、MuJoCo 仿真测试（Phase 1，已可用）

所有命令均在 `em_tactile_sim/` 目录下执行。

### 2.1 单元测试（无需 GPU，纯 Python）

```bash
# 全部测试（50 个，约 1s）
python3 -m pytest tests/ -v

# 只跑 core 层（不加载 MuJoCo，最快）
python3 -m pytest tests/test_sensor_config.py tests/test_contact_model.py tests/test_hall_response.py -v

# 只跑 DataRecorder
python3 -m pytest tests/test_recorder.py -v

# 只跑 MuJoCo 集成测试（需要加载 XML，约 0.3s）
python3 -m pytest tests/test_integration.py -v
```

**期望输出：** `50 passed, 1 warning`

### 2.2 标准品（7×7）— 基础压力测试

```bash
python3 examples/flat_press_test.py
```

模拟球（半径 5mm，质量 10g）从 5cm 高自由落下，撞击传感器垫，每 0.1s 打印一次最大法向力和合力。

**期望输出：**

```
Sensor: 7x7, dim=151
Timestep: 8.33 ms, sim for 5 s (600 steps)

t=0.00s  max_fn=0.0000 N  resultant=[0.000, 0.000, 0.000] N
t=0.10s  max_fn=0.0282 N  resultant=[0.184, 0.000, 0.000] N
...
```

### 2.3 实时 7×7 热力图可视化

```bash
python3 examples/visualize_array.py
```

弹出 matplotlib 窗口，实时渲染 7×7 法向力分布（颜色范围 0~20N），关闭窗口即退出。

> **注意**：需要 GUI 环境（X11/Wayland）。若在无头服务器上，改用 2.4 时序方案。

### 2.4 时序数据录制与导出

```bash
python3 examples/timeseries_plot.py
```

无 GUI，仿真 5s 后自动生成：

| 输出文件 | 内容 |
|----------|------|
| `examples/output/resultant_timeseries.png` | 合力三分量（Fn/Ftx/Fty）随时间折线图 |
| `examples/output/sensor_timeseries.csv` | 每帧：time, fn_max, resultant_fn, resultant_ftx, resultant_fty |

**期望输出：**

```
开始仿真...
完成步数: 100/600
...
完成步数: 600/600

总帧数: 600
最大法向合力: X.XX N
CSV 导出路径: examples/output/sensor_timeseries.csv
图表导出路径: examples/output/resultant_timeseries.png
```

### 2.5 在自己的脚本中使用传感器

```python
from em_tactile_sim.core.sensor_config import SensorConfig, SensorVariant
from em_tactile_sim.mujoco.env import EMTactileEnv
from em_tactile_sim.utils.recorder import DataRecorder

# 选择变体
config = SensorConfig(variant=SensorVariant.STANDARD)   # 7×7, dim=151
# config = SensorConfig(variant=SensorVariant.CUSTOM1)  # 6×6, dim=112

xml = "em_tactile_sim/mujoco/models/em_sensor_flat.xml"
# xml = "em_tactile_sim/mujoco/models/em_sensor_custom1.xml"  # CUSTOM1

env = EMTactileEnv(xml, config)
recorder = DataRecorder(config)

for _ in range(600):                        # 5s @ 120Hz
    env.step()
    recorder.record(env._data.time, env.get_tactile_flat())

tactile   = env.get_tactile()               # (7, 7, 3) — 阵列分布力
resultant = env.get_resultant()             # (3,) — [Fn_sum, Ftx_sum, Fty_sum]
temp      = env.get_temperature()           # float — 温度 (Phase 1 = 0.0)

recorder.save_csv("output.csv")
recorder.plot_resultant("output.png")

env.close()
```

### 2.6 CUSTOM1 指尖型（6×6）测试

```bash
python3 - <<'EOF'
import mujoco
from em_tactile_sim.core.sensor_config import SensorConfig, SensorVariant
from em_tactile_sim.mujoco.env import EMTactileEnv

cfg = SensorConfig(variant=SensorVariant.CUSTOM1)
env = EMTactileEnv("em_tactile_sim/mujoco/models/em_sensor_custom1.xml", cfg)
env.step()
print(f"CUSTOM1: rows={cfg.rows}, cols={cfg.cols}, dim={cfg.sensor_dim}")
print(f"tactile shape: {env.get_tactile().shape}")   # (6, 6, 3)
print(f"flat shape: {env.get_tactile_flat().shape}") # (112,)
env.close()
EOF
```

### 2.7 带 cell site 可视化的模型（调试用）

在 MuJoCo Viewer 中可看到 49 个橙色小点标记测点位置：

```bash
# 生成带 sites 的 XML（已存在，无需重新生成）
python3 em_tactile_sim/mujoco/models/gen_sites.py

# 用带 sites 的模型启动 Viewer（需要 GUI）
python3 - <<'EOF'
import mujoco
from mujoco import viewer
m = mujoco.MjModel.from_xml_path("em_tactile_sim/mujoco/models/em_sensor_flat_with_sites.xml")
d = mujoco.MjData(m)
with viewer.launch_passive(m, d) as v:
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
EOF
```

---

## 三、Isaac Sim 集成（Phase 2，开发中）

> Phase 2 正在开发，本节将随实现进度持续更新。

### 3.1 当前状态

`isaac/` 目录功能正在实现中，预计包含：

```
em_tactile_sim/isaac/
├── models/
│   └── em_sensor_flat.usd      # 传感器 USD 资产
├── sensor_callback.py           # PhysX 接触 → core 层转换
└── env.py                       # EMTactileIsaacEnv，与 MuJoCo env 接口对齐
```

### 3.2 Phase 2 计划架构

**core 层完全复用**，只需替换接触数据读取方式：

| 模块 | MuJoCo (Phase 1) | Isaac Sim (Phase 2) |
|------|-----------------|---------------------|
| 接触读取 | `mujoco.mj_contactForce` | `RigidContactView` PhysX API |
| 回调注册 | `set_mjcb_sensor` | `world.add_physics_callback` |
| 输出格式 | `data.sensordata` | numpy buffer（相同接口） |
| 可视化 | matplotlib | rerun / OmniGraph |

### 3.3 Isaac Sim 环境检测

```bash
python3 -c "import isaacsim; print('Isaac Sim available')" 2>&1
```

---

## 四、快速参考

| 命令 | 用途 |
|------|------|
| `pytest tests/ -v` | 全部测试（50个） |
| `pytest tests/test_integration.py -v` | MuJoCo 集成测试 |
| `python3 examples/flat_press_test.py` | 球落下压力测试（终端输出） |
| `python3 examples/visualize_array.py` | 实时热力图（需 GUI） |
| `python3 examples/timeseries_plot.py` | 时序录制，生成 CSV + PNG |
