# em_tactile_sim

> MuJoCo simulation library for electromagnetic (Hall-effect) tactile sensors.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo 3.0+](https://img.shields.io/badge/mujoco-3.0%2B-green.svg)](https://mujoco.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 38 passed](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](tests/)

`em_tactile_sim` models the electromagnetic tactile sensors manufactured by 深圳元触科技有限公司 inside MuJoCo. It extracts contact forces at each physics step, applies Hertz contact mechanics to distribute force across the sensing grid, and applies a linear Hall-effect response model to produce a realistic sensor output vector.

---

## Features

- **Two sensor variants** — Standard 7×7 and Custom1 6×6, fully configurable via `SensorConfig`
- **Physics-accurate force distribution** — Hertz contact model distributes normal and shear forces to individual sensing cells
- **Realistic Hall response** — linear sensitivity model maps force arrays to raw sensor output vectors
- **MuJoCo callback integration** — `mjcb_sensor` callback with clean register/unregister lifecycle
- **Full output vector** — `[fn_array, resultant, temperature]` matching physical hardware protocol
- **Tested** — 38 unit and integration tests covering the full pipeline

---

## Installation

**Prerequisites:** Python 3.10+, MuJoCo 3.0+

```bash
git clone git@github.com:hzm8341/em_tactile_sim.git
cd em_tactile_sim
pip install -e .
```

To install development dependencies (pytest):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
from em_tactile_sim.core.sensor_config import SensorConfig, SensorVariant
from em_tactile_sim.mujoco.env import EMTactileEnv

# Standard 7×7 sensor
cfg = SensorConfig()

# Custom1 6×6 sensor
cfg = SensorConfig(SensorVariant.CUSTOM1)

# Load your MuJoCo model and create the environment
env = EMTactileEnv("em_tactile_sim/mujoco/models/em_sensor_flat.xml", cfg)

# Advance the simulation
env.step()

# Read sensor outputs
tactile  = env.get_tactile()       # (7, 7, 3)  — [fn, ftx, fty] per cell
resultant = env.get_resultant()    # (3,)        — total force vector
flat     = env.get_tactile_flat()  # (151,)      — full output vector
temp     = env.get_temperature()   # float       — temperature (Phase 1: 0.0)

# Clean up
env.reset()
env.close()
```

---

## Sensor Specifications

| Parameter          | Standard      | Custom1       |
|--------------------|---------------|---------------|
| Grid               | 7×7           | 6×6           |
| Dimensions         | 43×28×8 mm    | 20×17×9 mm    |
| Interface          | USB Type-C    | CAN bus       |
| Sample rate        | 120 Hz        | 120 Hz        |
| Normal force range | 0–20 N        | 0–20 N        |
| Shear force range  | ±10 N         | ±10 N         |
| Sensitivity        | 0.05 N        | 0.05 N        |
| Output dimension   | 151           | 112           |

**Output vector layout (Standard):** `[fn_array (147), resultant (3), temperature (1)]` = 151 values

---

## Architecture

```
MuJoCo Physics Step
        │
        ▼
EMSensorCallback (mjcb_sensor)
        │  raw contact forces per site
        ▼
ContactModel (Hertz contact mechanics)
        │  per-cell force distribution  (rows × cols × 3)
        ▼
compute_output() (linear Hall response)
        │  sensor output vector
        ▼
 [fn_array | resultant | temperature]
  147 values   3 values   1 value
             = 151 total (Standard)
```

### Module Map

```
em_tactile_sim/
  core/
    sensor_config.py     SensorVariant enum + SensorConfig dataclass
    contact_model.py     ContactModel: Hertz contact mechanics
    hall_response.py     compute_output(): force array → sensor vector
  mujoco/
    models/
      em_sensor_flat.xml              MJCF model (7×7 pad + falling ball)
      em_sensor_flat_with_sites.xml   with visual site markers
      gen_sites.py                    generates site XML elements
    callback.py          EMSensorCallback: registers mjcb_sensor
    env.py               EMTactileEnv: step/get_tactile/reset/close/render
  isaac/                 (placeholder — Phase 2)
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **38 passed**

| Test file                    | Tests | Coverage                              |
|------------------------------|-------|---------------------------------------|
| `test_sensor_config.py`      | 10    | SensorVariant, SensorConfig fields    |
| `test_contact_model.py`      | 11    | Hertz mechanics, force distribution   |
| `test_hall_response.py`      | 8     | Hall response, output vector layout   |
| `test_integration.py`        | 9     | Full MuJoCo pipeline                  |

---

## Examples

### Ball-drop demo (headless)

Drops a ball onto the sensor pad and prints the maximum normal force every 0.1 s.

```bash
python examples/flat_press_test.py
```

### Real-time heatmap (requires display)

Renders a live 7×7 force heatmap using matplotlib.

```bash
python examples/visualize_array.py
```

---

## API Reference

### `SensorConfig`

```python
from em_tactile_sim.core.sensor_config import SensorConfig, SensorVariant

cfg = SensorConfig()                       # Standard 7×7, 151-dim output
cfg = SensorConfig(SensorVariant.CUSTOM1)  # Custom1 6×6, 112-dim output
```

Key fields: `rows`, `cols`, `output_dim`, `hall_sensitivity`, `variant`.

### `EMTactileEnv`

```python
env = EMTactileEnv(model_path: str, config: SensorConfig)
```

| Method                 | Returns        | Description                              |
|------------------------|----------------|------------------------------------------|
| `env.step()`           | `None`         | Advance one physics step                 |
| `env.get_tactile()`    | `(R, C, 3)`    | Per-cell `[fn, ftx, fty]` array          |
| `env.get_resultant()`  | `(3,)`         | Total force vector                       |
| `env.get_tactile_flat()` | `(output_dim,)` | Full output vector                    |
| `env.get_temperature()` | `float`       | Temperature (Phase 1: always `0.0`)      |
| `env.reset()`          | `None`         | Reset simulation state                   |
| `env.close()`          | `None`         | Unregister `mjcb_sensor` callback        |

**Important:** Always call `env.close()` when done. `EMSensorCallback` registers a global `mjcb_sensor` callback; failing to unregister it will affect subsequent MuJoCo environments in the same process.

---

## Roadmap

| Phase | Status      | Description                                            |
|-------|-------------|--------------------------------------------------------|
| 1     | Complete    | MuJoCo integration, Hertz contact model, linear Hall response, 38 tests |
| 2     | Planned     | Isaac Sim integration (`em_tactile_sim/isaac/`)        |
| 3     | Planned     | Real sensor data collection and calibration            |
| 4     | Planned     | LUT / MLP Hall-effect response model                   |

---

## License

MIT © 2024

---

## Acknowledgements

Sensor specifications and hardware reference provided by **深圳元触科技有限公司 (Shenzhen Yuanchu Technology Co., Ltd.)**.
