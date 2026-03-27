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
