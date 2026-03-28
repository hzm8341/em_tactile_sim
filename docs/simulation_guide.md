# em_tactile_sim 仿真测试操作说明

> 版本：Phase 2（MuJoCo 完整，Isaac Sim 已实现）
> 最后更新：2026-03-27

---

## 一、环境准备

### 1.1 依赖版本（已验证）

| 包 | 版本 |
|----|------|
| Python | 3.10 |
| mujoco | 3.6.0 |
| numpy | 2.2.6 |
| matplotlib | 3.10.8 |

### 1.2 创建 Conda 环境

```bash
# 创建专用环境（Python 3.10）
conda create -n tactisim python=3.10 -y

# 激活环境
conda activate tactisim
```

### 1.3 安装包

```bash
cd /media/hzm/data_disk/tactiSim_all/em_tactile_sim

# 开发模式安装（含测试依赖）
pip install -e ".[dev]"

# 验证安装
python3 -c "import em_tactile_sim; print('OK')"
```

> **注意**：后续所有命令均在 `conda activate tactisim` 环境下执行。

---

## 二、MuJoCo 仿真测试（Phase 1，已可用）

所有命令均在 `em_tactile_sim/` 目录下执行。

### 2.1 自动化测试（无需 GPU，约 1s）

```bash
# 全部测试（57 个）
python3 -m pytest tests/ -v

# 只跑 core 层（最快，无 MuJoCo 依赖）
python3 -m pytest tests/test_sensor_config.py tests/test_contact_model.py tests/test_hall_response.py -v

# 只跑 SensorBridge 单元测试（Isaac Sim 层，无需 Isaac Sim）
python3 -m pytest tests/test_isaac_sensor_bridge.py -v

# 只跑 MuJoCo 集成测试
python3 -m pytest tests/test_integration.py -v
```

**期望输出：**
```
57 passed, 1 skipped, 1 warning
```
（1 skipped = Isaac Sim 集成测试，在无 Isaac Sim 环境下自动跳过）

---

### 2.2 手动测试：标准品（7×7）球落压力测试

```bash
python3 examples/flat_press_test.py
```

**场景**：半径 5mm、质量 10g 的球从 5cm 高自由落下，撞击 7×7 传感器垫，每 0.1s 打印一次法向力读数。

**期望输出（精确值）：**

```
Sensor: 7x7, dim=151
Timestep: 8.33 ms, sim for 5 s (600 steps)

t=0.00s  max_fn=0.0000 N  resultant=[0.000, 0.000, 0.000] N   ← 球尚未触地
t=0.10s  max_fn=0.0282 N  resultant=[0.184, 0.000, 0.000] N   ← 撞击峰值
t=0.20s  max_fn=0.0232 N  resultant=[0.096, 0.000, 0.000] N   ← 球静止压合
t=0.30s  max_fn=0.0232 N  resultant=[0.098, 0.000, 0.000] N
...（后续数值稳定在此水平）...

Done.
```

**判断标准：**
- `t=0.00s`：全零，说明传感器初始无接触 ✓
- `t=0.10s`：出现峰值，说明球撞击被正确捕获 ✓
- `t=0.20s` 之后：数值稳定（不为零），说明球静止后持续接触 ✓
- `Ftx / Fty` 始终为 `0.000`：球垂直落下，无切向力 ✓

---

### 2.3 手动测试：实时 7×7 热力图可视化

> **需要 GUI 环境（X11 / Wayland）**，无头服务器请跳至 2.4。

```bash
python3 examples/visualize_array.py
```

**期望效果：**
- 弹出 matplotlib 窗口，标题 `EM Tactile Sensor — 7×7 Normal Force`
- 窗口中央显示 7×7 彩色方格（颜色范围 0~20N，蓝→黄→红）
- 初始全蓝（无接触）
- 约 0.1s 后中心区域出现黄/橙色亮斑（球落下瞬间）
- 之后亮斑持续存在并略微扩散（球静止压合）
- 关闭窗口即退出仿真

---

### 2.4 手动测试：时序数据录制与导出

```bash
python3 examples/timeseries_plot.py
```

**期望终端输出：**

```
开始仿真...
仿真时步: 0.008330s
目标时间: 5.0s
预计总步数: 600

完成步数: 100/600
完成步数: 200/600
完成步数: 300/600
完成步数: 400/600
完成步数: 500/600
完成步数: 600/600

============================================================
仿真完成 - 数据摘要
============================================================
总帧数: 600
仿真时间: 4.998s
最大法向合力: 1.1999 N
最大法向合力时刻: 0.1000s

CSV 导出路径: .../examples/output/sensor_timeseries.csv
图表导出路径: .../examples/output/resultant_timeseries.png
============================================================
```

**期望输出文件：**

| 文件 | 内容 |
|------|------|
| `examples/output/sensor_timeseries.csv` | 600行，列：time, fn_max, resultant_fn, resultant_ftx, resultant_fty |
| `examples/output/resultant_timeseries.png` | 折线图：Fn_sum（蓝）在 t≈0.1s 出现峰值后趋于平稳；Ftx/Fty（橙/绿）始终为 0 |

**判断标准：**
- `最大法向合力 ≈ 1.20 N`（允许 ±0.1N 范围内波动）
- `最大法向合力时刻 = 0.1000s`（球落下的撞击时刻）
- CSV 中 `resultant_ftx` / `resultant_fty` 列全为 0.000

---

### 2.5 手动测试：CUSTOM1 指尖型（6×6）

```bash
python3 - <<'EOF'
from em_tactile_sim.core.sensor_config import SensorConfig, SensorVariant
from em_tactile_sim.mujoco.env import EMTactileEnv

cfg = SensorConfig(variant=SensorVariant.CUSTOM1)
env = EMTactileEnv("em_tactile_sim/mujoco/models/em_sensor_custom1.xml", cfg)
env.step()
print(f"CUSTOM1: rows={cfg.rows}, cols={cfg.cols}, dim={cfg.sensor_dim}")
print(f"tactile shape: {env.get_tactile().shape}")
print(f"flat shape:    {env.get_tactile_flat().shape}")
env.close()
EOF
```

**期望输出：**

```
CUSTOM1: rows=6, cols=6, dim=112
tactile shape: (6, 6, 3)
flat shape:    (112,)
```

**判断标准：**
- `dim=112 = 6×6×3 + 3 + 1`，shape 正确即通过 ✓

---

### 2.6 手动测试：MuJoCo Viewer（交互式调试）

> **需要 GUI 环境**，可看到传感器几何体和球的实时运动。

```bash
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

**期望效果：**
- 弹出 MuJoCo Viewer 窗口
- 场景中可见：蓝色传感器垫 + 红色球（静止在垫上）
- 传感器垫表面有 49 个橙色小点标记 7×7 测点位置
- 在 Viewer 中拖拽可旋转视角，滚轮缩放

---

## 三、Isaac Sim 集成测试（Phase 2，需 Isaac Sim 环境）

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

### 3.2 前置检查：确认 Isaac Sim 版本

```bash
python3 -c "
from em_tactile_sim.isaac import is_isaac_available, get_isaac_version
print('Available:', is_isaac_available())
print('Version:  ', get_isaac_version())
"
```

**期望输出：**
```
Available: True
Version:   4.5.0      # 或 5.0.0
```

若输出 `Available: False`，说明当前 Python 解释器不是 Isaac Sim 内置的，需改用下方方式运行。

---

### 3.3 手动测试：独立脚本（球落压力测试）

使用 Isaac Sim 自带的 Python 解释器运行：

```bash
# Isaac Sim 4.5.0
~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh examples/isaac_press_test.py

# Isaac Sim 5.0.0
~/.local/share/ov/pkg/isaac-sim-5.0.0/python.sh examples/isaac_press_test.py
```

**期望终端输出：**

```
仿真 600 步（5s @ 120Hz）...
  步数: 100/600
  步数: 200/600
  步数: 300/600
  步数: 400/600
  步数: 500/600
  步数: 600/600

总帧数: 600
最大法向合力: X.XXXX N      ← 应 > 0（PhysX 接触力非零）
CSV: .../examples/output/isaac_sensor_timeseries.csv
PNG: .../examples/output/isaac_resultant_timeseries.png
```

**判断标准：**
- 脚本无报错运行完毕 ✓
- `最大法向合力 > 0`（球落下后有接触力输出）✓
- 输出文件存在：`isaac_sensor_timeseries.csv` 和 `isaac_resultant_timeseries.png` ✓

> **注意**：PhysX 与 MuJoCo 使用不同的接触求解器，数值不会与 MuJoCo 完全一致，但合力量级应在同一数量级（0.1~2 N 范围内）。

---

### 3.4 手动测试：Extension UI（图形界面插件）

在 Isaac Sim 图形界面中加载：

1. 打开 Isaac Sim → **Window → Extensions**
2. 点击右上角 **+** → 路径填写：`/media/hzm/data_disk/tactiSim_all/em_tactile_sim/em_tactile_sim/isaac/`
3. 在列表中找到 `EMTactileExtension` → 点击启用

**期望效果：**

| 时机 | 期望现象 |
|------|----------|
| Extension 加载后 | 弹出 `EM Tactile Sensor` 面板，包含 Step / Reset 按钮、Fn_sum / Ftx_sum / Fty_sum 数值标签、7×7 FloatField 数值网格（初始全为 0.000） |
| 点击 **Step** 若干次（约 12 次 = 0.1s） | Fn_sum 出现非零读数；7×7 网格中心附近若干格出现非零数值 |
| 点击 **Reset** | Fn_sum / Ftx_sum / Fty_sum 归零；7×7 网格全部恢复 0.000 |
| 关闭 Extension（on_shutdown） | 仿真 World 正常停止，无报错 |

---

### 3.5 手动测试：接口一致性验证（Isaac Sim vs MuJoCo）

在 Isaac Sim Python 解释器中运行：

```python
from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.isaac.env import EMTactileIsaacEnv

cfg = SensorConfig()
env = EMTactileIsaacEnv("em_tactile_sim/isaac/models/em_sensor_flat.usda", cfg)
env.setup()

# 运行 120 步（约 1s）让球落下
for _ in range(120):
    env.step()

tactile   = env.get_tactile()       # 期望 shape: (7, 7, 3)
resultant = env.get_resultant()     # 期望 shape: (3,)
temp      = env.get_temperature()   # 期望值: 0.0
flat      = env.get_tactile_flat()  # 期望 shape: (151,)

print(f"tactile shape:   {tactile.shape}")    # (7, 7, 3)
print(f"resultant shape: {resultant.shape}")  # (3,)
print(f"temperature:     {temp}")             # 0.0
print(f"flat shape:      {flat.shape}")       # (151,)
print(f"Fn_sum:          {resultant[0]:.4f} N")  # 应 > 0

env.close()
```

**期望输出：**
```
tactile shape:   (7, 7, 3)
resultant shape: (3,)
temperature:     0.0
flat shape:      (151,)
Fn_sum:          X.XXXX N     ← > 0 即通过
```

---

## 四、快速参考

| 命令 | 用途 | 依赖 |
|------|------|------|
| `pytest tests/ -v` | 全部自动化测试（57个） | 无 |
| `pytest tests/test_isaac_sensor_bridge.py -v` | SensorBridge 单元测试 | 无 |
| `python3 examples/flat_press_test.py` | MuJoCo 球落压力（终端输出） | MuJoCo |
| `python3 examples/visualize_array.py` | MuJoCo 实时热力图 | MuJoCo + GUI |
| `python3 examples/timeseries_plot.py` | MuJoCo 时序录制 → CSV + PNG | MuJoCo |
| `isaac-sim/python.sh examples/isaac_press_test.py` | Isaac Sim 球落测试 | Isaac Sim |
| Isaac Sim Extension UI | 图形界面 + 7×7 热力图 | Isaac Sim + GUI |
