"""
窗函数比较示例
演示不同窗函数的特性和应用效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_processing import WAVReader, FrameProcessor, WindowFunctions
import matplotlib.pyplot as plt
import numpy as np


def window_comparison_example():
    """窗函数比较示例"""
    print("=" * 60)
    print("窗函数比较示例")
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
    
    # 创建分帧处理器
    processor = FrameProcessor(sample_rate, 25.0, 10.0)
    
    # 比较三种窗函数
    window_types = ['rectangular', 'hamming', 'hanning']
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(15, 10))
    
    # 1. 窗函数时域特性
    plt.subplot(2, 2, 1)
    frame_length = 256
    n = np.arange(frame_length)
    
    rect_window = WindowFunctions.rectangular_window(frame_length)
    hamming_window = WindowFunctions.hamming_window(frame_length)
    hanning_window = WindowFunctions.hanning_window(frame_length)
    
    plt.plot(n, rect_window, 'b-', label='Rectangular Window', linewidth=2)
    plt.plot(n, hamming_window, 'r-', label='Hamming Window', linewidth=2)
    plt.plot(n, hanning_window, 'g-', label='Hanning Window', linewidth=2)
    plt.title('Window Function Time Domain Characteristics')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 2. 窗函数频域特性
    plt.subplot(2, 2, 2)
    freqs = np.fft.fftfreq(frame_length)
    rect_fft = np.abs(np.fft.fft(rect_window))
    hamming_fft = np.abs(np.fft.fft(hamming_window))
    hanning_fft = np.abs(np.fft.fft(hanning_window))
    
    plt.plot(freqs[:frame_length//2], rect_fft[:frame_length//2], 'b-', label='Rectangular Window', linewidth=2)
    plt.plot(freqs[:frame_length//2], hamming_fft[:frame_length//2], 'r-', label='Hamming Window', linewidth=2)
    plt.plot(freqs[:frame_length//2], hanning_fft[:frame_length//2], 'g-', label='Hanning Window', linewidth=2)
    plt.title('Window Function Frequency Domain Characteristics')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 3. 不同窗函数对语音信号的影响
    plt.subplot(2, 2, 3)
    # 取一段语音信号
    start_sample = len(audio_data) // 4
    end_sample = start_sample + frame_length
    test_signal = audio_data[start_sample:end_sample]
    
    rect_result = test_signal * rect_window
    hamming_result = test_signal * hamming_window
    hanning_result = test_signal * hanning_window
    
    time_axis = np.arange(len(test_signal)) / sample_rate
    
    plt.plot(time_axis, test_signal, 'k-', label='Original Signal', alpha=0.7)
    plt.plot(time_axis, rect_result, 'b-', label='Rectangular Window', linewidth=2)
    plt.plot(time_axis, hamming_result, 'r-', label='Hamming Window', linewidth=2)
    plt.plot(time_axis, hanning_result, 'g-', label='Hanning Window', linewidth=2)
    plt.title('Window Function Application Effects')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 4. 窗函数参数对比表
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    table_data = [
        ['Window Type', 'Main Lobe Width', 'Side Lobe Level(dB)', 'Application'],
        ['Rectangular', '0.89', '-13.3', 'Time Domain Analysis'],
        ['Hamming', '1.30', '-42.7', 'Speech Analysis'],
        ['Hanning', '1.44', '-31.5', 'Spectral Analysis']
    ]
    
    table = plt.table(cellText=table_data,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Window Function Characteristics Comparison')
    
    plt.tight_layout()
    plt.show()
    
    # 打印窗函数特性
    print("\n窗函数特性对比:")
    print("=" * 60)
    print(f"{'窗函数':<10} {'主瓣宽度':<10} {'旁瓣电平(dB)':<15} {'应用场景'}")
    print("-" * 60)
    print(f"{'矩形窗':<10} {'0.89':<10} {'-13.3':<15} {'时域分析'}")
    print(f"{'汉明窗':<10} {'1.30':<10} {'-42.7':<15} {'语音分析'}")
    print(f"{'海宁窗':<10} {'1.44':<10} {'-31.5':<15} {'频谱分析'}")
    print("=" * 60)


if __name__ == "__main__":
    window_comparison_example()
