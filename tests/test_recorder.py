"""DataRecorder 单元测试。"""

import csv
import os
import tempfile

import numpy as np
import pytest

from em_tactile_sim.core.sensor_config import SensorConfig
from em_tactile_sim.utils.recorder import DataRecorder


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> SensorConfig:
    """返回默认 SensorConfig（7×7 STANDARD）。"""
    return SensorConfig()


@pytest.fixture
def recorder(config: SensorConfig) -> DataRecorder:
    """返回空 DataRecorder。"""
    return DataRecorder(config)


def _make_tactile_flat(config: SensorConfig,
                       fn_val: float = 1.0,
                       ft_val: float = 0.5,
                       fn_sum: float = 5.0,
                       ftx_sum: float = 1.0,
                       fty_sum: float = -1.0) -> np.ndarray:
    """构造一个合法的 tactile_flat 向量。

    布局：[阵列力(array_dim,), 合力(3,), 温度(1,)]
    阵列力通道顺序：[fn, ftx, fty] 每个 cell。
    """
    array_dim = config.array_dim            # rows*cols*3
    total_dim = array_dim + 3 + 1           # 加合力和温度

    flat = np.zeros(total_dim, dtype=float)

    # 阵列力：法向力通道（通道 0）设为 fn_val，切向设为 ft_val
    array_part = flat[:array_dim].reshape(config.rows, config.cols, 3)
    array_part[:, :, 0] = fn_val   # 法向
    array_part[:, :, 1] = ft_val   # ftx
    array_part[:, :, 2] = ft_val   # fty
    flat[:array_dim] = array_part.ravel()

    # 合力
    flat[array_dim]     = fn_sum
    flat[array_dim + 1] = ftx_sum
    flat[array_dim + 2] = fty_sum

    return flat


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_initial_state(self, recorder: DataRecorder) -> None:
        """初始状态：n_frames==0，各序列为空数组。"""
        assert recorder.n_frames == 0
        assert recorder.times.shape == (0,)
        assert recorder.resultants.shape == (0, 3)
        assert recorder.fn_max_series.shape == (0,)


class TestRecord:
    def test_record_increments_frames(self, recorder: DataRecorder,
                                      config: SensorConfig) -> None:
        """record 一帧后 n_frames==1。"""
        flat = _make_tactile_flat(config)
        recorder.record(0.1, flat)
        assert recorder.n_frames == 1

    def test_record_stores_time(self, recorder: DataRecorder,
                                config: SensorConfig) -> None:
        """times[0] 等于输入 time。"""
        flat = _make_tactile_flat(config)
        recorder.record(0.25, flat)
        assert recorder.times[0] == pytest.approx(0.25)

    def test_record_stores_resultant(self, recorder: DataRecorder,
                                     config: SensorConfig) -> None:
        """resultants[0] 等于 tactile_flat[array_dim:array_dim+3]。"""
        flat = _make_tactile_flat(config, fn_sum=7.0, ftx_sum=2.0, fty_sum=-3.0)
        recorder.record(0.0, flat)
        expected = flat[config.array_dim:config.array_dim + 3]
        np.testing.assert_allclose(recorder.resultants[0], expected)

    def test_record_stores_fn_max(self, recorder: DataRecorder,
                                  config: SensorConfig) -> None:
        """fn_max_series[0] 等于阵列法向力最大值。"""
        flat = _make_tactile_flat(config, fn_val=3.5)
        recorder.record(0.0, flat)
        expected_max = flat[:config.array_dim].reshape(
            config.rows, config.cols, 3)[:, :, 0].max()
        assert recorder.fn_max_series[0] == pytest.approx(expected_max)

    def test_record_multiple_frames(self, recorder: DataRecorder,
                                    config: SensorConfig) -> None:
        """record 多帧后 n_frames 正确累加。"""
        for i in range(5):
            flat = _make_tactile_flat(config, fn_val=float(i))
            recorder.record(i * 0.01, flat)
        assert recorder.n_frames == 5


class TestReset:
    def test_reset_clears_data(self, recorder: DataRecorder,
                               config: SensorConfig) -> None:
        """record 后 reset，n_frames==0，序列为空。"""
        flat = _make_tactile_flat(config)
        recorder.record(0.1, flat)
        assert recorder.n_frames == 1

        recorder.reset()
        assert recorder.n_frames == 0
        assert recorder.times.shape == (0,)
        assert recorder.resultants.shape == (0, 3)
        assert recorder.fn_max_series.shape == (0,)


class TestSaveCsv:
    def test_save_csv_creates_file(self, recorder: DataRecorder,
                                   config: SensorConfig) -> None:
        """save_csv 后文件存在。"""
        flat = _make_tactile_flat(config)
        recorder.record(0.0, flat)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            recorder.save_csv(csv_path)
            assert os.path.exists(csv_path)
        finally:
            os.unlink(csv_path)

    def test_save_csv_empty(self, recorder: DataRecorder) -> None:
        """无记录时 save_csv 生成只有标题行的文件。"""
        with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w") as f:
            csv_path = f.name
        try:
            recorder.save_csv(csv_path)
            with open(csv_path, newline="") as f:
                rows = list(csv.reader(f))
            # 只有 1 行（标题行）
            assert len(rows) == 1
            assert rows[0] == ["time", "fn_max", "resultant_fn",
                                "resultant_ftx", "resultant_fty"]
        finally:
            os.unlink(csv_path)

    def test_save_csv_content(self, recorder: DataRecorder,
                              config: SensorConfig) -> None:
        """record 2 帧，CSV 有 3 行（标题+2数据行），time 列正确。"""
        flat1 = _make_tactile_flat(config, fn_sum=4.0)
        flat2 = _make_tactile_flat(config, fn_sum=6.0)
        recorder.record(0.1, flat1)
        recorder.record(0.2, flat2)

        with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w") as f:
            csv_path = f.name
        try:
            recorder.save_csv(csv_path)
            with open(csv_path, newline="") as f:
                rows = list(csv.reader(f))
            assert len(rows) == 3  # 标题 + 2 数据行
            assert float(rows[1][0]) == pytest.approx(0.1)
            assert float(rows[2][0]) == pytest.approx(0.2)
        finally:
            os.unlink(csv_path)


class TestPlotResultant:
    def test_plot_resultant_creates_file(self, recorder: DataRecorder,
                                         config: SensorConfig) -> None:
        """plot_resultant 后文件存在（不 show）。"""
        flat = _make_tactile_flat(config)
        recorder.record(0.0, flat)
        recorder.record(0.1, flat)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name
        try:
            recorder.plot_resultant(png_path, show=False)
            assert os.path.exists(png_path)
            assert os.path.getsize(png_path) > 0
        finally:
            os.unlink(png_path)

    def test_plot_resultant_empty(self, recorder: DataRecorder) -> None:
        """无记录时 plot_resultant 不报错，文件存在。"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name
        try:
            recorder.plot_resultant(png_path, show=False)
            assert os.path.exists(png_path)
            assert os.path.getsize(png_path) > 0
        finally:
            os.unlink(png_path)
