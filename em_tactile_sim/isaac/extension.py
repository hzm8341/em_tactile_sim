"""Isaac Sim IExt extension entry point.

This is a thin UI shell — all sensor logic lives in env.py.
UI layout:
  - [Step] [Reset] control buttons
  - Resultant force: Fn / Ftx / Fty float labels
  - 7x7 normal-force grid (49 FloatField widgets, 0~20N colour-mapped)
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
    """Isaac Sim Extension: EM tactile sensor simulation with 7x7 heatmap."""

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
        """7x7 grid of FloatField widgets showing fn per cell (N)."""
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

        # Update 7x7 heatmap
        fn_map = tactile[:, :, 0]               # (7, 7)
        for idx, field in enumerate(self._fn_fields):
            i, j = divmod(idx, self._cfg.cols)
            field.model.set_value(float(fn_map[i, j]))
