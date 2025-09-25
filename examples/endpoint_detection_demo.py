"""
端点检测演示示例
演示不同参数下的端点检测效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_processing import WAVReader, DualThresholdEndpointDetector
import matplotlib.pyplot as plt
import numpy as np


def endpoint_detection_demo():
    """端点检测演示"""
    print("=" * 60)
    print("端点检测演示示例")
    print("=" * 60)
    
    # 查找音频文件
    audio_dir = "../data/audio"
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')] if os.path.exists(audio_dir) else []
    
    if not wav_files:
        print("在 data/audio 目录下没有找到WAV文件")
        print("请将WAV文件放在 data/audio 目录下")
        return
    
    # 显示可用的音频文件
    print("可用的音频文件:")
    for i, file in enumerate(wav_files):
        print(f"{i+1}. {file}")
    
    # 让用户选择文件
    while True:
        try:
            choice = input(f"\n请选择要分析的文件编号 (1-{len(wav_files)}): ")
            file_index = int(choice) - 1
            if 0 <= file_index < len(wav_files):
                break
            else:
                print(f"请输入1到{len(wav_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    # 使用用户选择的文件
    wav_file = os.path.join(audio_dir, wav_files[file_index])
    print(f"分析文件: {wav_file}")
    
    # 读取音频文件
    reader = WAVReader(wav_file)
    audio_data, sample_rate = reader.read()
    
    # 创建端点检测器
    detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
    
    # 测试不同的参数组合
    energy_ratios = [0.05, 0.1, 0.2, 0.3]
    zcr_ratios = [1.2, 1.5, 2.0, 2.5]
    
    plt.figure(figsize=(15, 12))
    
    # 显示不同能量阈值比例的效果（英文标签，避免中文字体问题）
    for i, energy_ratio in enumerate(energy_ratios):
        result = detector.detect_endpoints(audio_data, energy_ratio=energy_ratio, zcr_ratio=1.5)
        
        plt.subplot(2, 2, i+1)
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        plt.plot(time_axis, audio_data, 'b-', linewidth=1, alpha=0.7)
        
        # 标记检测到的语音段（仅与当前视窗相交部分）
        for endpoint in result['endpoints']:
            seg_start = max(endpoint['start_time'], 0.0)
            seg_end = min(endpoint['end_time'], time_axis[-1])
            if seg_end > seg_start:
                plt.axvspan(seg_start, seg_end, alpha=0.15, color='yellow')
        
        plt.title(f'Energy Ratio: {energy_ratio} ({len(result["endpoints"])} segments)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 详细分析一个参数组合
    print("\n详细分析 (能量阈值比例: 0.1, 过零率比例: 1.5):")
    result = detector.detect_endpoints(audio_data, energy_ratio=0.1, zcr_ratio=1.5)
    detector.plot_endpoint_detection(audio_data, result)
    
    # 参数对比表
    print("\n不同参数下的端点检测结果对比:")
    print("=" * 80)
    print(f"{'能量阈值比例':<12} {'过零率比例':<12} {'语音段数量':<10} {'总语音时长(秒)':<15}")
    print("-" * 80)
    
    for energy_ratio in energy_ratios:
        for zcr_ratio in zcr_ratios:
            result = detector.detect_endpoints(audio_data, energy_ratio=energy_ratio, zcr_ratio=zcr_ratio)
            total_duration = sum(endpoint['duration'] for endpoint in result['endpoints'])
            print(f"{energy_ratio:<12} {zcr_ratio:<12} {len(result['endpoints']):<10} {total_duration:<15.3f}")
    
    print("=" * 80)
    
    # 提取语音段
    speech_segments = detector.extract_speech_segments(audio_data, result)
    print(f"\n提取到 {len(speech_segments)} 个语音段:")
    for i, segment in enumerate(speech_segments):
        print(f"语音段 {i+1}: {len(segment)} 采样点 ({len(segment)/sample_rate:.3f} 秒)")


if __name__ == "__main__":
    endpoint_detection_demo()
