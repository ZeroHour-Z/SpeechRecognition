"""
语音信号处理基础使用示例
演示如何使用语音处理包的基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_processing import WAVReader, FrameProcessor, TimeDomainAnalyzer, DualThresholdEndpointDetector
import matplotlib.pyplot as plt
import numpy as np


def basic_analysis_example():
    """基础分析示例"""
    print("=" * 60)
    print("语音信号处理基础使用示例")
    print("=" * 60)
    
    # 查找音频文件
    audio_dir = "../data/audio"
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')] if os.path.exists(audio_dir) else []
    
    if not wav_files:
        print("在 data/audio 目录下没有找到WAV文件")
        print("请将WAV文件放在 data/audio 目录下")
        return
    
    # 使用第一个WAV文件
    wav_file = os.path.join(audio_dir, wav_files[0])
    print(f"分析文件: {wav_file}")
    
    # 1. 读取WAV文件
    print("\n1. 读取WAV文件...")
    reader = WAVReader(wav_file)
    audio_data, sample_rate = reader.read()
    reader.print_info()
    
    # 2. 分帧处理
    print("\n2. 分帧处理...")
    processor = FrameProcessor(sample_rate, 25.0, 10.0)
    frames, windowed_frames = processor.process_signal(audio_data, 'hamming')
    print(f"分帧完成，共 {len(frames)} 帧")
    
    # 3. 时域分析
    print("\n3. 时域分析...")
    analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
    analysis_result = analyzer.analyze_signal(audio_data, 'hamming')
    print("时域特征计算完成")
    
    # 4. 端点检测
    print("\n4. 端点检测...")
    detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
    endpoint_result = detector.detect_endpoints(audio_data)
    print(f"端点检测完成，检测到 {len(endpoint_result['endpoints'])} 个语音段")
    
    # 5. 简单可视化
    print("\n5. 生成可视化结果...")
    plot_basic_analysis(audio_data, sample_rate, analysis_result, endpoint_result)
    
    print("\n基础分析完成！")


def plot_basic_analysis(audio_data, sample_rate, analysis_result, endpoint_result):
    """绘制基础分析结果"""
    plt.figure(figsize=(12, 8))
    
    # 原始信号
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    plt.plot(time_axis, audio_data, 'b-', linewidth=1)
    plt.title('原始语音信号')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)
    
    # 短时能量
    plt.subplot(3, 1, 2)
    plt.plot(analysis_result['time_axis'], analysis_result['energy'], 'r-', linewidth=2)
    plt.title('短时能量')
    plt.xlabel('时间 (秒)')
    plt.ylabel('能量')
    plt.grid(True, alpha=0.3)
    
    # 端点检测结果
    plt.subplot(3, 1, 3)
    plt.plot(analysis_result['time_axis'], endpoint_result['speech_frames'].astype(int), 'g-', linewidth=2)
    plt.fill_between(analysis_result['time_axis'], 0, endpoint_result['speech_frames'].astype(int), 
                    alpha=0.3, color='green')
    plt.title('端点检测结果')
    plt.xlabel('时间 (秒)')
    plt.ylabel('语音/静音')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    basic_analysis_example()
