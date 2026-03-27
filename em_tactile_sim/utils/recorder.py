"""时序传感器数据记录器模块。

每步 record() 追加数据，支持 CSV 导出和合力折线图绘制。
"""

import csv
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from em_tactile_sim.core.sensor_config import SensorConfig


class DataRecorder:
    """时序传感器数据记录器。每步 record() 追加数据，支持 CSV 导出和合力折线图。"""

    def __init__(self, config: SensorConfig) -> None:
        """
        初始化记录器。

        参数:
            config: SensorConfig 实例（用于解析 array_dim/sensor_dim 维度）
        """
        self._config = config
        self._times: List[float] = []
        self._resultants: List[np.ndarray] = []  # 每帧 shape (3,)
        self._fn_max_series: List[float] = []

    def record(self, time: float, tactile_flat: np.ndarray) -> None:
        """
        记录一帧传感器数据。

        参数:
            time: 仿真时间（秒）
            tactile_flat: shape (sensor_dim,)，来自 env.get_tactile_flat()
        """
        array_dim = self._config.array_dim
        rows = self._config.rows
        cols = self._config.cols

        # 提取合力三分量：tactile_flat[array_dim:array_dim+3]
        resultant = tactile_flat[array_dim:array_dim + 3].copy()

        # 计算法向力最大值：阵列力 reshape 后取第 0 通道（法向）的最大值
        array_forces = tactile_flat[:array_dim].reshape(rows, cols, 3)
        fn_max = float(array_forces[:, :, 0].max())

        self._times.append(time)
        self._resultants.append(resultant)
        self._fn_max_series.append(fn_max)

    def save_csv(self, path: str) -> None:
        """
        导出 CSV 文件。

        列格式：time, fn_max, resultant_fn, resultant_ftx, resultant_fty
        （fn_max = 法向力阵列的最大值；resultant_* = 合力三分量）

        若没有记录数据，写一个只有标题行的空文件。
        """
        header = ["time", "fn_max", "resultant_fn", "resultant_ftx", "resultant_fty"]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, t in enumerate(self._times):
                res = self._resultants[i]
                row = [t, self._fn_max_series[i], res[0], res[1], res[2]]
                writer.writerow(row)

    def plot_resultant(self, path: str, show: bool = False) -> None:
        """
        绘制合力三分量随时间变化的折线图，保存到 path（PNG）。

        三条线：Fn_sum（蓝）、Ftx_sum（橙）、Fty_sum（绿）
        X轴：time(s)，Y轴：Force(N)
        若没有记录数据，保存一张空图（带标题）。

        参数:
            show: 若 True 则调用 plt.show()（交互模式）
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Resultant Forces over Time")
        ax.set_xlabel("time(s)")
        ax.set_ylabel("Force(N)")

        if self._times:
            times_arr = np.array(self._times)
            resultants_arr = np.array(self._resultants)  # (n_frames, 3)

            ax.plot(times_arr, resultants_arr[:, 0], color="blue", label="Fn_sum")
            ax.plot(times_arr, resultants_arr[:, 1], color="orange", label="Ftx_sum")
            ax.plot(times_arr, resultants_arr[:, 2], color="green", label="Fty_sum")
            ax.legend()

        fig.tight_layout()
        fig.savefig(path)

        if show:
            plt.show()

        plt.close(fig)

    def reset(self) -> None:
        """清空所有已记录数据。"""
        self._times.clear()
        self._resultants.clear()
        self._fn_max_series.clear()

    @property
    def n_frames(self) -> int:
        """已记录帧数。"""
        return len(self._times)

    @property
    def times(self) -> np.ndarray:
        """时间序列，shape (n_frames,)。"""
        return np.array(self._times, dtype=float)

    @property
    def resultants(self) -> np.ndarray:
        """合力时序，shape (n_frames, 3)，列为 [fn_sum, ftx_sum, fty_sum]。"""
        if not self._resultants:
            return np.empty((0, 3), dtype=float)
        return np.array(self._resultants, dtype=float)

    @property
    def fn_max_series(self) -> np.ndarray:
        """每帧法向力最大值，shape (n_frames,)。"""
        return np.array(self._fn_max_series, dtype=float)
