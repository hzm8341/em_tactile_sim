"""
时序传感器数据采集与可视化示例

本脚本演示如何使用 EMTactileEnv 和 DataRecorder 实现：
1. 仿真球自由落下撞击传感器（5秒，600步 @ 120Hz）
2. 每步记录传感器数据
3. 导出时序数据为 CSV 文件
4. 生成合力折线图

运行方式：
    cd em_tactile_sim/
    python examples/timeseries_plot.py
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
from em_tactile_sim.mujoco.env import EMTactileEnv
from em_tactile_sim.utils.recorder import DataRecorder
from em_tactile_sim.core.sensor_config import SensorConfig


def main():
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 获取 XML 模型路径
    xml_path = os.path.join(
        os.path.dirname(__file__),
        "../em_tactile_sim/mujoco/models/em_sensor_flat.xml"
    )

    # 初始化配置和环境
    config = SensorConfig()
    env = EMTactileEnv(xml_path, config)
    recorder = DataRecorder(config)

    # 仿真参数
    timestep = env._model.opt.timestep  # 应为 1/120 秒
    target_time = 5.0  # 目标仿真时间：5秒
    n_steps = int(target_time / timestep)  # 应为 600 步

    print(f"开始仿真...")
    print(f"仿真时步: {timestep:.6f}s")
    print(f"目标时间: {target_time}s")
    print(f"预计总步数: {n_steps}")
    print()

    # 运行仿真
    for step_idx in range(n_steps):
        # 执行一步仿真
        env.step()

        # 获取当前时间和传感器数据
        current_time = env._data.time
        tactile_flat = env.get_tactile_flat()

        # 记录数据
        recorder.record(current_time, tactile_flat)

        # 进度输出（每 100 步输出一次）
        if (step_idx + 1) % 100 == 0:
            print(f"完成步数: {step_idx + 1}/{n_steps}")

    # 仿真完成
    env.close()

    # 导出结果
    csv_path = os.path.join(output_dir, "sensor_timeseries.csv")
    png_path = os.path.join(output_dir, "resultant_timeseries.png")

    recorder.save_csv(csv_path)
    recorder.plot_resultant(png_path, show=False)

    # 计算统计信息
    n_frames = recorder.n_frames
    times_arr = np.array(recorder.times)
    resultants_arr = np.array(recorder.resultants)

    # 找到合力中法向分量的最大值及其时刻
    fn_values = resultants_arr[:, 0]  # 法向力分量
    max_fn_idx = np.argmax(fn_values)
    max_fn = fn_values[max_fn_idx]
    max_fn_time = times_arr[max_fn_idx]

    # 打印摘要
    print()
    print("=" * 60)
    print("仿真完成 - 数据摘要")
    print("=" * 60)
    print(f"总帧数: {n_frames}")
    print(f"仿真时间: {times_arr[-1]:.3f}s")
    print(f"最大法向合力: {max_fn:.4f} N")
    print(f"最大法向合力时刻: {max_fn_time:.4f}s")
    print()
    print(f"CSV 导出路径: {csv_path}")
    print(f"图表导出路径: {png_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
