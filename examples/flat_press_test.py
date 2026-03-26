"""
Demo: ball falls onto sensor pad.
Prints max normal force every 0.1 s (every 12 steps @ 120 Hz).
Run from em_tactile_sim/ root:
    python examples/flat_press_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.mujoco.env import EMTactileEnv

XML = os.path.join(
    os.path.dirname(__file__),
    "../em_tactile_sim/mujoco/models/em_sensor_flat.xml",
)


def main():
    cfg = SensorConfig()
    env = EMTactileEnv(XML, cfg)

    print(f"Sensor: {cfg.rows}x{cfg.cols}, dim={cfg.sensor_dim}")
    print(f"Timestep: {env._model.opt.timestep*1000:.2f} ms, "
          f"sim for 5 s ({int(5/env._model.opt.timestep)} steps)\n")

    n_steps = int(5.0 / env._model.opt.timestep)   # 5 seconds
    report_every = int(0.1 / env._model.opt.timestep)  # every 0.1 s

    for frame in range(n_steps):
        env.step()
        if frame % report_every == 0:
            t       = env.get_tactile()
            r       = env.get_resultant()
            fn_max  = t[:, :, 0].max()
            t_s     = frame * env._model.opt.timestep
            print(f"t={t_s:.2f}s  max_fn={fn_max:.4f} N  "
                  f"resultant=[{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}] N")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
