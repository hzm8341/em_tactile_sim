"""
Real-time 7×7 normal-force heatmap during simulation.
Run from em_tactile_sim/ root:
    python examples/visualize_array.py
Closes when matplotlib window is closed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.mujoco.env import EMTactileEnv

XML = os.path.join(
    os.path.dirname(__file__),
    "../em_tactile_sim/mujoco/models/em_sensor_flat.xml",
)


def main():
    cfg = SensorConfig()
    env = EMTactileEnv(XML, cfg)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        np.zeros((cfg.rows, cfg.cols)),
        vmin=0, vmax=cfg.fn_max,
        cmap="hot", origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Normal Force (N)")
    ax.set_title("EM Tactile Sensor — 7×7 Normal Force")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        color="white", fontsize=9)

    steps_per_frame = 2   # advance 2 physics steps per animation frame

    def update(_frame):
        for _ in range(steps_per_frame):
            env.step()
        fn_map = env.get_tactile()[:, :, 0]
        im.set_data(fn_map)
        t = env._data.time
        time_text.set_text(f"t={t:.3f}s  max={fn_map.max():.3f}N")
        return im, time_text

    ani = animation.FuncAnimation(
        fig, update, interval=16, blit=True, cache_frame_data=False)

    try:
        plt.show()
    finally:
        env.close()


if __name__ == "__main__":
    main()
