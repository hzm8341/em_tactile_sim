#!/usr/bin/env python3
"""
Interactive EM Tactile Sensor Dashboard  (PaXini-style)

Layout  (matches reference video):
  Left  60%  :  7x7 per-cell time-series mini plots
                 red=Fn(normal), yellow=Ftx(shear-X), cyan=Fty(shear-Y)
  Right 40%  :
    Top  40%  :  Sensor array — circles coloured by Fn, arrows show Ft direction
    Bottom 60%:  3-D deformation surface — blue=contact (depression), red=flat

MuJoCo window (opens separately):
    Ctrl + left-drag on RED BALL  ->  push / roll the ball
    Right-drag                    ->  rotate camera
    Scroll                        ->  zoom
    Space                         ->  pause / resume
"""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Fix: system mpl_toolkits conflicts with pip matplotlib — force pip version
import site as _site
import mpl_toolkits as _mpl_tk
_mpl_tk.__path__ = [_site.getusersitepackages() + "/mpl_toolkits"]

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import mujoco
from mujoco import viewer as mjviewer
from scipy.interpolate import RectBivariateSpline

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.mujoco.env import EMTactileEnv


XML = os.path.join(os.path.dirname(__file__),
                   "../em_tactile_sim/mujoco/models/em_sensor_rolling.xml")

HISTORY_LEN = 180   # per-cell history length (~1.5 s @ 120 Hz)


# ──────────────────────────────────────────────────────────────────────────────
# Thread-safe shared state
# ──────────────────────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self, cfg: SensorConfig) -> None:
        self._lock = threading.Lock()
        self.is_running = True

        self.tactile   = np.zeros((cfg.rows, cfg.cols, 3))
        self.resultant = np.zeros(3)
        self.sim_time  = 0.0

        # Resultant rolling buffers
        self._fn_h  = np.zeros(HISTORY_LEN)
        self._ftx_h = np.zeros(HISTORY_LEN)
        self._fty_h = np.zeros(HISTORY_LEN)
        self._t_h   = np.zeros(HISTORY_LEN)

        # Per-cell rolling buffers: (rows, cols, channel, time)
        # channel 0=Fn, 1=Ftx, 2=Fty
        self._cell_h = np.zeros((cfg.rows, cfg.cols, 3, HISTORY_LEN))

    def push(self, tactile: np.ndarray, resultant: np.ndarray, t: float) -> None:
        with self._lock:
            self.tactile[:]   = tactile
            self.resultant[:] = resultant
            self.sim_time     = t

            self._fn_h  = np.roll(self._fn_h,  -1); self._fn_h[-1]  = resultant[0]
            self._ftx_h = np.roll(self._ftx_h, -1); self._ftx_h[-1] = resultant[1]
            self._fty_h = np.roll(self._fty_h, -1); self._fty_h[-1] = resultant[2]
            self._t_h   = np.roll(self._t_h,   -1); self._t_h[-1]   = t

            self._cell_h        = np.roll(self._cell_h, -1, axis=3)
            self._cell_h[:, :, 0, -1] = tactile[:, :, 0]
            self._cell_h[:, :, 1, -1] = tactile[:, :, 1]
            self._cell_h[:, :, 2, -1] = tactile[:, :, 2]

    def snapshot(self) -> tuple:
        with self._lock:
            return (
                self.tactile.copy(),
                self.resultant.copy(),
                self.sim_time,
                self._fn_h.copy(),
                self._ftx_h.copy(),
                self._fty_h.copy(),
                self._t_h.copy(),
                self._cell_h.copy(),
            )


# ──────────────────────────────────────────────────────────────────────────────
# MuJoCo simulation thread
# ──────────────────────────────────────────────────────────────────────────────
def run_mujoco(env: EMTactileEnv, state: SharedState) -> None:
    key_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_KEY, "roll")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(env._model, env._data, key_id)

    with mjviewer.launch_passive(env._model, env._data) as v:
        v.cam.azimuth   = 30
        v.cam.elevation = -22
        v.cam.distance  = 0.10
        v.cam.lookat[:] = [0, 0, 0.005]

        while v.is_running() and state.is_running:
            env.step()
            state.push(env.get_tactile(), env.get_resultant(), env._data.time)
            v.sync()

    state.is_running = False


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────
def build_and_run_dashboard(cfg: SensorConfig, state: SharedState) -> None:
    BG    = "#f0f2f5"
    PANEL = "#ffffff"
    TC    = "#1a1a2e"
    SPINE = "#b0b8cc"
    CMAP  = "jet"

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL,
        "text.color":       TC,
        "axes.labelcolor":  TC,
        "xtick.color":      TC,
        "ytick.color":      TC,
        "axes.edgecolor":   SPINE,
        "grid.color":       SPINE,
        "grid.linewidth":   0.4,
        "font.size":        7,
    })

    rows, cols = cfg.rows, cfg.cols
    cmap_fn = plt.get_cmap(CMAP)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "EM Tactile Sensor — Interactive Dashboard"
        "    [ MuJoCo: Ctrl+drag the RED BALL to push it ]",
        fontsize=11, fontweight="bold", color=TC,
    )

    # ── Main grid: left (cell plots) | right (array + 3-D) ──────────────────
    gs_main = GridSpec(
        1, 2, figure=fig,
        left=0.01, right=0.99, top=0.94, bottom=0.02,
        wspace=0.06, width_ratios=[3, 2],
    )

    # Left: 7×7 mini time-series subplots
    gs_cells = GridSpecFromSubplotSpec(
        rows, cols, subplot_spec=gs_main[0, 0],
        hspace=0.10, wspace=0.10,
    )

    # Right: array (top) + 3-D surface (bottom)
    gs_right = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_main[0, 1],
        height_ratios=[1, 1.6], hspace=0.32,
    )

    # ── Build 7×7 mini plots ─────────────────────────────────────────────────
    axes_cells: list[list] = []
    cell_lines: list[list] = []   # [r][c] = (l_fn, l_ftx, l_fty)

    for r in range(rows):
        row_ax, row_ln = [], []
        for c in range(cols):
            ax = fig.add_subplot(gs_cells[r, c])
            ax.set_facecolor(PANEL)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color(SPINE)
                sp.set_linewidth(0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.001, 0.04)
            ax.axhline(0, color=SPINE, lw=0.5)

            # Tiny cell label (top-left corner)
            ax.text(0.03, 0.92, f"R{r}C{c}",
                    transform=ax.transAxes,
                    fontsize=4, color="#445577", va="top")

            l_fn,  = ax.plot([], [], color="#ff4040", lw=0.9)
            l_ftx, = ax.plot([], [], color="#ffdd00", lw=0.6, ls="--")
            l_fty, = ax.plot([], [], color="#00dddd", lw=0.6, ls=":")

            row_ax.append(ax)
            row_ln.append((l_fn, l_ftx, l_fty))
        axes_cells.append(row_ax)
        cell_lines.append(row_ln)

    # ── Right-top: sensor array ───────────────────────────────────────────────
    ax_arr = fig.add_subplot(gs_right[0])
    ax_arr.set_facecolor(PANEL)
    ax_arr.set_title(f"{rows}x{cols} Sensor Array  (colour=Fn | arrow=Ft)",
                     fontsize=9, pad=4, color=TC)
    ax_arr.set_aspect("equal")
    ax_arr.axis("off")
    ax_arr.set_xlim(-0.7, cols - 0.3)
    ax_arr.set_ylim(-1.0, rows - 0.3)

    circles: list[list] = []
    for r in range(rows):
        row_c = []
        for c in range(cols):
            circ = plt.Circle((c, r), 0.40, color=cmap_fn(0.0), zorder=2)
            ax_arr.add_patch(circ)
            row_c.append(circ)
        circles.append(row_c)

    Cg, Rg = np.meshgrid(np.arange(cols, dtype=float),
                          np.arange(rows, dtype=float))
    quiv = ax_arr.quiver(
        Cg, Rg,
        np.zeros((rows, cols)), np.zeros((rows, cols)),
        color="black",
        scale=1, scale_units="xy",   # U=0.42 → arrow 0.42 data-units long
        width=0.025, headwidth=4, headlength=5,
        zorder=5, pivot="mid",
    )

    # Colorbar for array
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb_arr = fig.colorbar(sm, ax=ax_arr, fraction=0.03, pad=0.02, shrink=0.75)
    cb_arr.set_label("Fn (normalised)", fontsize=6, color=TC)
    cb_arr.ax.tick_params(colors=TC, labelsize=5)

    # Info text below array
    arr_info = ax_arr.text(
        cols / 2 - 0.5, -0.80, "",
        color=TC, fontsize=7, ha="center",
    )

    # Readout text (top-right corner of array panel)
    readout = ax_arr.text(
        cols - 0.3, rows - 0.3, "",
        color="#1a3a6a", fontsize=7, ha="right", va="top",
        fontfamily="monospace",
    )

    # ── Right-bottom: 3-D deformation surface ────────────────────────────────
    ax_3d = fig.add_subplot(gs_right[1], projection="3d")
    ax_3d.set_facecolor(PANEL)
    ax_3d.set_title("3D Deformation  (blue=contact / red=flat)",
                     fontsize=9, pad=4, color=TC)
    ax_3d.set_xlabel("col", fontsize=6, labelpad=3)
    ax_3d.set_ylabel("row", fontsize=6, labelpad=3)
    ax_3d.set_zlabel("height", fontsize=6, labelpad=3)
    ax_3d.tick_params(colors=TC, labelsize=5)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor(SPINE)
    ax_3d.yaxis.pane.set_edgecolor(SPINE)
    ax_3d.zaxis.pane.set_edgecolor(SPINE)
    ax_3d.view_init(elev=30, azim=-55)

    # Fine interpolation grid (5× resolution → 35×35)
    N_FINE  = 35
    _r_orig = np.arange(rows, dtype=float)
    _c_orig = np.arange(cols, dtype=float)
    _r_fine = np.linspace(0, rows - 1, N_FINE)
    _c_fine = np.linspace(0, cols - 1, N_FINE)
    Cg3, Rg3 = np.meshgrid(_c_fine, _r_fine)

    # Initial surface: flat at height=1 (all red)
    surf_holder: list = [
        ax_3d.plot_surface(
            Cg3, Rg3, np.ones((N_FINE, N_FINE)),
            cmap=CMAP, vmin=0, vmax=1,
            linewidth=0, antialiased=True, alpha=0.93,
        )
    ]
    ax_3d.set_zlim(0, 1.1)

    # ── Animation update ─────────────────────────────────────────────────────
    frame_idx   = [0]
    last_x0     = [0.0]   # track time-window start for lazy xlim updates
    last_print  = [0.0]   # for debug prints

    def update(_frame: int) -> list:
        if not state.is_running:
            return []

        (tactile, resultant, sim_t,
         fn_h, ftx_h, fty_h, t_h, cell_h) = state.snapshot()

        fn_map  = tactile[:, :, 0]
        ftx_map = tactile[:, :, 1]
        fty_map = tactile[:, :, 2]

        # Dynamic scale (always fills full colour range)
        fn_display_max = max(fn_map.max(), 0.005)
        fn_norm = np.clip(fn_map / fn_display_max, 0.0, 1.0)

        # ── 1. Per-cell mini plots ────────────────────────────────────────────
        mask = t_h > 0
        if mask.any():
            t_win = t_h[mask]
            x0, x1 = float(t_win[0]), float(t_win[-1])

            # Shared Y scale across all cells (based on global max)
            y_max_fn = max(cell_h[:, :, 0, :].max(), 0.005)
            y_min_ft = min(cell_h[:, :, 1:, :].min(), -0.001)
            y_lo = y_min_ft - y_max_fn * 0.05
            y_hi = y_max_fn * 1.20

            # Update xlim only when window shifts noticeably
            xlim_changed = abs(x0 - last_x0[0]) > 0.08
            if xlim_changed:
                last_x0[0] = x0

            for r in range(rows):
                for c in range(cols):
                    ax  = axes_cells[r][c]
                    lns = cell_lines[r][c]
                    lns[0].set_data(t_win, cell_h[r, c, 0, mask])
                    lns[1].set_data(t_win, cell_h[r, c, 1, mask])
                    lns[2].set_data(t_win, cell_h[r, c, 2, mask])
                    if xlim_changed:
                        ax.set_xlim(x0, x1 + 1e-4)
                    ax.set_ylim(y_lo, y_hi)

        # ── 2. Sensor array ───────────────────────────────────────────────────
        for r in range(rows):
            for c in range(cols):
                circles[r][c].set_facecolor(cmap_fn(fn_norm[r, c]))

        shear = np.sqrt(ftx_map**2 + fty_map**2)
        s_max = float(shear.max())
        # Pure dynamic normalisation: arrows always fill 42% of cell radius
        # whenever any shear exists; invisible when Ft==0 (ball static)
        SHEAR_THRESH = 1e-7   # N — below this treat as zero
        if s_max > SHEAR_THRESH:
            quiv.set_UVC(ftx_map / s_max * 0.42,
                         fty_map / s_max * 0.42)
        else:
            quiv.set_UVC(np.zeros_like(ftx_map), np.zeros_like(fty_map))

        arr_info.set_text(
            f"Fn scale: 0 - {fn_display_max:.4f} N  |  "
            f"Shear: {s_max*1000:.3f} mN"
            + ("  << Ctrl+drag ball for arrows >>" if s_max <= SHEAR_THRESH else "")
        )
        readout.set_text(
            f"t = {sim_t:.2f} s\n"
            f"Fn  = {resultant[0]:+.3f} N\n"
            f"Ftx = {resultant[1]*1000:+.2f} mN\n"
            f"Fty = {resultant[2]*1000:+.2f} mN"
        )

        # ── Debug: print every frame when shear is non-zero ──────────────────
        if s_max > SHEAR_THRESH:
            print(f"  [SHEAR] t={sim_t:.2f}s  s_max={s_max*1000:.3f}mN  "
                  f"Ftx={resultant[1]*1000:+.2f}mN  Fty={resultant[2]*1000:+.2f}mN  "
                  f"arrow_U={ftx_map.max()/s_max*0.42:.3f}")
        elif sim_t - last_print[0] >= 2.0:
            last_print[0] = sim_t
            print(f"  [static] t={sim_t:.1f}s  Fn={resultant[0]:.3f}N  "
                  f"Shear={s_max*1e6:.2f}uN  (drag ball to see arrows)")

        # ── 3. 3-D deformation surface (every 5 frames) ───────────────────────
        frame_idx[0] += 1
        if frame_idx[0] % 5 == 0:
            # Bowl effect: high Fn → depression → low z (blue); flat → high z (red)
            z_coarse = 1.0 - fn_norm        # (rows, cols), range [0, 1]
            # Bicubic interpolation → 35×35 smooth surface
            interp  = RectBivariateSpline(_r_orig, _c_orig, z_coarse, kx=3, ky=3)
            z_fine  = np.clip(interp(_r_fine, _c_fine), 0.0, 1.0)
            surf_holder[0].remove()
            surf_holder[0] = ax_3d.plot_surface(
                Cg3, Rg3, z_fine,
                cmap=CMAP, vmin=0, vmax=1,
                linewidth=0, antialiased=True, alpha=0.93,
            )
            ax_3d.set_zlim(max(float(z_fine.min()) - 0.05, 0), 1.05)

        return []

    ani = animation.FuncAnimation(
        fig, update, interval=50, blit=False, cache_frame_data=False,
    )

    print("\nDashboard running. Close matplotlib window to exit.\n")
    try:
        plt.show()
    finally:
        state.is_running = False


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = SensorConfig()
    env = EMTactileEnv(XML, cfg)
    state = SharedState(cfg)

    print("=" * 58)
    print("  EM Tactile Sensor — Interactive Dashboard")
    print("=" * 58)
    print(f"  Sensor  : {cfg.rows}x{cfg.cols}  dim={cfg.sensor_dim}")
    print(f"  Timestep: {env._model.opt.timestep * 1000:.2f} ms  "
          f"({1/env._model.opt.timestep:.0f} Hz)")
    print()
    print("  MuJoCo controls:")
    print("    Ctrl + left-drag on RED BALL  -> push the ball")
    print("    Right-drag                    -> rotate camera")
    print("    Scroll wheel                  -> zoom")
    print("    Space                         -> pause / resume")
    print()
    print("  NOTE: drag the RED SPHERE, not the blue flat pad!")
    print("=" * 58)

    sim_thread = threading.Thread(
        target=run_mujoco, args=(env, state), daemon=True
    )
    sim_thread.start()

    build_and_run_dashboard(cfg, state)
    env.close()
    print("Exited.")


if __name__ == "__main__":
    main()
