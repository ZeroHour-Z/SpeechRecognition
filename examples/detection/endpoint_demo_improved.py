"""
改进的端点检测演示示例
修复参数变化无效果的问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import WAVReader, DualThresholdEndpointDetector
import matplotlib.pyplot as plt
import numpy as np


def endpoint_detection_demo_improved():
    """改进的端点检测演示"""
    print("=" * 60)
    print("改进的端点检测演示示例")
    print("=" * 60)
    
    # 查找音频文件
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    audio_dir = os.path.join(project_root, "data", "audio", "samples")
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')] if os.path.exists(audio_dir) else []
    
    if not wav_files:
        print("在 data/audio/samples 目录下没有找到WAV文件")
        print("请将WAV文件放在 data/audio/samples 目录下")
        return
    
    # 显示可用的音频文件
    print("可用的音频文件:")
    for i, file in enumerate(wav_files):
        print(f"{i+1}. {file}")
    
    # 使用第一个文件进行演示
    wav_file = os.path.join(audio_dir, wav_files[0])
    print(f"\n分析文件: {wav_file}")
    
    # 读取音频文件
    reader = WAVReader(wav_file)
    audio_data, sample_rate = reader.read()
    
    # 创建端点检测器
    detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
    
    # 测试不同的参数组合
    energy_ratios = [0.05, 0.1, 0.2, 0.3]
    
    plt.figure(figsize=(15, 12))
    
    # 显示不同能量阈值比例的效果
    for i, energy_ratio in enumerate(energy_ratios):
        result = detector.detect_endpoints(audio_data, energy_ratio=energy_ratio, zcr_ratio=1.5)
        
        plt.subplot(2, 2, i+1)
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        plt.plot(time_axis, audio_data, 'b-', linewidth=1, alpha=0.7)
        
        # 显示阈值线
        thresholds = result['thresholds']
        energy = result['analysis_result']['short_time_energy']
        energy_time = result['analysis_result']['time_axis']
        
        # 绘制能量阈值线
        plt.axhline(y=thresholds['energy_high'], color='red', linestyle='--', alpha=0.7, label=f'Energy High: {thresholds["energy_high"]:.4f}')
        plt.axhline(y=thresholds['energy_low'], color='orange', linestyle='--', alpha=0.7, label=f'Energy Low: {thresholds["energy_low"]:.4f}')
        
        # 标记检测到的语音段
        for endpoint in result['endpoints']:
            seg_start = max(endpoint['start_time'], 0.0)
            seg_end = min(endpoint['end_time'], time_axis[-1])
            if seg_end > seg_start:
                plt.axvspan(seg_start, seg_end, alpha=0.3, color='yellow', label='Detected Speech')
        
        plt.title(f'Energy Ratio: {energy_ratio} ({len(result["endpoints"])} segments)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        if i == 0:  # 只在第一个子图显示图例
            plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 详细分析参数影响
    print("\n详细参数分析:")
    print("=" * 80)
    print(f"{'能量阈值比例':<12} {'高阈值':<12} {'低阈值':<12} {'语音段数量':<10} {'总语音时长(秒)':<15}")
    print("-" * 80)
    
    for energy_ratio in energy_ratios:
        result = detector.detect_endpoints(audio_data, energy_ratio=energy_ratio, zcr_ratio=1.5)
        thresholds = result['thresholds']
        total_speech_time = sum([ep['end_time'] - ep['start_time'] for ep in result['endpoints']])
        
        print(f"{energy_ratio:<12.2f} {thresholds['energy_high']:<12.4f} {thresholds['energy_low']:<12.4f} "
              f"{len(result['endpoints']):<10} {total_speech_time:<15.2f}")
    
    # 显示能量分布
    print("\n能量分布分析:")
    print("=" * 50)
    result = detector.detect_endpoints(audio_data, energy_ratio=0.1, zcr_ratio=1.5)
    energy = result['analysis_result']['short_time_energy']
    thresholds = result['thresholds']
    
    print(f"最大能量: {thresholds['max_energy']:.4f}")
    print(f"最小能量: {thresholds['min_energy']:.4f}")
    print(f"平均能量: {np.mean(energy):.4f}")
    print(f"能量标准差: {np.std(energy):.4f}")
    print(f"高阈值: {thresholds['energy_high']:.4f} ({thresholds['energy_high']/thresholds['max_energy']*100:.1f}% of max)")
    print(f"低阈值: {thresholds['energy_low']:.4f} ({thresholds['energy_low']/thresholds['max_energy']*100:.1f}% of max)")


if __name__ == "__main__":
    endpoint_detection_demo_improved()

