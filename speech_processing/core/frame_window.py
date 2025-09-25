"""
语音分帧和加窗处理模块
实现语音信号的短时分析技术，包括分帧和三种窗函数处理
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class WindowFunctions:
    """窗函数类"""
    
    @staticmethod
    def rectangular_window(frame_length: int) -> np.ndarray:
        """
        矩形窗函数
        
        Args:
            frame_length: 帧长
            
        Returns:
            np.ndarray: 矩形窗函数
        """
        return np.ones(frame_length)
    
    @staticmethod
    def hamming_window(frame_length: int) -> np.ndarray:
        """
        汉明窗函数
        
        Args:
            frame_length: 帧长
            
        Returns:
            np.ndarray: 汉明窗函数
        """
        n = np.arange(frame_length)
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (frame_length - 1))
    
    @staticmethod
    def hanning_window(frame_length: int) -> np.ndarray:
        """
        海宁窗函数
        
        Args:
            frame_length: 帧长
            
        Returns:
            np.ndarray: 海宁窗函数
        """
        n = np.arange(frame_length)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (frame_length - 1)))


class FrameProcessor:
    """语音分帧处理器"""
    
    def __init__(self, sample_rate: int, frame_length_ms: float = 25.0, 
                 frame_shift_ms: float = 10.0):
        """
        初始化分帧处理器
        
        Args:
            sample_rate: 采样率
            frame_length_ms: 帧长（毫秒）
            frame_shift_ms: 帧移（毫秒）
        """
        self.sample_rate = sample_rate
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        
        # 计算帧长和帧移的采样点数
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)
        
        print(f"分帧参数:")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  帧长: {frame_length_ms} ms ({self.frame_length} 采样点)")
        print(f"  帧移: {frame_shift_ms} ms ({self.frame_shift} 采样点)")
        print(f"  重叠率: {(self.frame_length - self.frame_shift) / self.frame_length * 100:.1f}%")
    
    def frame_signal(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        对信号进行分帧处理
        
        Args:
            signal: 输入信号
            
        Returns:
            List[np.ndarray]: 分帧后的信号列表
        """
        frames = []
        signal_length = len(signal)
        
        # 计算总帧数
        num_frames = (signal_length - self.frame_length) // self.frame_shift + 1
        
        for i in range(num_frames):
            start = i * self.frame_shift
            end = start + self.frame_length
            
            if end <= signal_length:
                frame = signal[start:end]
                frames.append(frame)
            else:
                # 最后一帧如果不够长，用零填充
                frame = np.zeros(self.frame_length)
                remaining = signal_length - start
                frame[:remaining] = signal[start:]
                frames.append(frame)
        
        return frames
    
    def apply_window(self, frames: List[np.ndarray], window_type: str = 'hamming') -> List[np.ndarray]:
        """
        对分帧后的信号应用窗函数
        
        Args:
            frames: 分帧后的信号列表
            window_type: 窗函数类型 ('rectangular', 'hamming', 'hanning')
            
        Returns:
            List[np.ndarray]: 加窗后的信号列表
        """
        # 获取窗函数
        if window_type == 'rectangular':
            window = WindowFunctions.rectangular_window(self.frame_length)
        elif window_type == 'hamming':
            window = WindowFunctions.hamming_window(self.frame_length)
        elif window_type == 'hanning':
            window = WindowFunctions.hanning_window(self.frame_length)
        else:
            raise ValueError(f"不支持的窗函数类型: {window_type}")
        
        # 应用窗函数
        windowed_frames = []
        for frame in frames:
            windowed_frame = frame * window
            windowed_frames.append(windowed_frame)
        
        return windowed_frames
    
    def process_signal(self, signal: np.ndarray, window_type: str = 'hamming') -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        完整的信号处理流程：分帧 + 加窗
        
        Args:
            signal: 输入信号
            window_type: 窗函数类型
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: (原始帧, 加窗帧)
        """
        # 分帧
        frames = self.frame_signal(signal)
        
        # 加窗
        windowed_frames = self.apply_window(frames, window_type)
        
        return frames, windowed_frames


def compare_windows(frame_length: int = 256) -> None:
    """
    比较三种窗函数的特性
    
    Args:
        frame_length: 帧长
    """
    # 生成三种窗函数
    rect_window = WindowFunctions.rectangular_window(frame_length)
    hamming_window = WindowFunctions.hamming_window(frame_length)
    hanning_window = WindowFunctions.hanning_window(frame_length)
    
    # 绘制时域特性
    plt.figure(figsize=(15, 10))
    
    # 时域波形
    plt.subplot(2, 2, 1)
    n = np.arange(frame_length)
    plt.plot(n, rect_window, 'b-', label='Rectangular Window', linewidth=2)
    plt.plot(n, hamming_window, 'r-', label='Hamming Window', linewidth=2)
    plt.plot(n, hanning_window, 'g-', label='Hanning Window', linewidth=2)
    plt.title('Window Function Time Domain Characteristics')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 频域特性
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
    
    # 窗函数参数对比
    plt.subplot(2, 2, 3)
    window_names = ['Rectangular', 'Hamming', 'Hanning']
    main_lobe_widths = [0.89, 1.3, 1.44]  # 主瓣宽度（归一化）
    side_lobe_levels = [-13.3, -42.7, -31.5]  # 旁瓣电平（dB）
    
    x = np.arange(len(window_names))
    width = 0.35
    
    plt.bar(x - width/2, main_lobe_widths, width, label='Main Lobe Width', alpha=0.8)
    plt.bar(x + width/2, [abs(level) for level in side_lobe_levels], width, label='Side Lobe Level (dB)', alpha=0.8)
    
    plt.xlabel('Window Type')
    plt.ylabel('Parameter Value')
    plt.title('Window Function Parameters Comparison')
    plt.xticks(x, window_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 窗函数应用效果对比
    plt.subplot(2, 2, 4)
    # 生成测试信号（正弦波 + 噪声）
    t = np.linspace(0, 1, frame_length)
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(frame_length)
    
    rect_result = test_signal * rect_window
    hamming_result = test_signal * hamming_window
    hanning_result = test_signal * hanning_window
    
    plt.plot(t, test_signal, 'k-', label='Original Signal', alpha=0.7)
    plt.plot(t, rect_result, 'b-', label='Rectangular Window', linewidth=2)
    plt.plot(t, hamming_result, 'r-', label='Hamming Window', linewidth=2)
    plt.plot(t, hanning_result, 'g-', label='Hanning Window', linewidth=2)
    plt.title('Window Function Application Effects')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 打印窗函数特性
    print("\nWindow Function Characteristics Comparison:")
    print("=" * 60)
    print(f"{'Window Type':<15} {'Main Lobe Width':<15} {'Side Lobe Level (dB)':<20} {'Application'}")
    print("-" * 60)
    print(f"{'Rectangular':<15} {'0.89':<15} {'-13.3':<20} {'Time Domain Analysis'}")
    print(f"{'Hamming':<15} {'1.30':<15} {'-42.7':<20} {'Speech Analysis'}")
    print(f"{'Hanning':<15} {'1.44':<15} {'-31.5':<20} {'Spectral Analysis'}")
    print("=" * 60)


def visualize_framing(signal: np.ndarray, sample_rate: int, 
                     frame_length_ms: float = 25.0, frame_shift_ms: float = 10.0,
                     start_time: float = 0.0, duration: float = 0.5) -> None:
    """
    可视化分帧过程
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        frame_length_ms: 帧长（毫秒）
        frame_shift_ms: 帧移（毫秒）
        start_time: 显示开始时间（秒）
        duration: 显示时长（秒）
    """
    # 创建分帧处理器
    processor = FrameProcessor(sample_rate, frame_length_ms, frame_shift_ms)
    
    # 计算显示范围
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    display_signal = signal[start_sample:end_sample]
    display_time = np.linspace(start_time, start_time + duration, len(display_signal))
    
    # 分帧
    frames = processor.frame_signal(display_signal)
    
    # 绘制分帧结果
    plt.figure(figsize=(15, 8))
    
    # 原始信号
    plt.subplot(2, 1, 1)
    plt.plot(display_time, display_signal, 'b-', linewidth=1, label='原始信号')
    
    # 标记帧边界
    frame_length_samples = int(sample_rate * frame_length_ms / 1000)
    frame_shift_samples = int(sample_rate * frame_shift_ms / 1000)
    
    for i, frame in enumerate(frames):
        frame_start = start_time + i * frame_shift_ms / 1000
        frame_end = frame_start + frame_length_ms / 1000
        
        # 绘制帧边界
        plt.axvline(x=frame_start, color='r', linestyle='--', alpha=0.7)
        if i == len(frames) - 1:
            plt.axvline(x=frame_end, color='r', linestyle='--', alpha=0.7)
        
        # 标记帧编号
        if i % 3 == 0:  # 每3帧标记一次
            plt.text(frame_start + frame_length_ms/2000, max(display_signal) * 0.8, 
                    f'帧{i}', fontsize=8, ha='center')
    
    plt.title(f'Speech Signal Framing (Frame Length: {frame_length_ms}ms, Frame Shift: {frame_shift_ms}ms)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 显示前几帧的详细内容
    plt.subplot(2, 1, 2)
    num_frames_to_show = min(5, len(frames))
    
    for i in range(num_frames_to_show):
        frame_start = i * frame_shift_samples
        frame_end = frame_start + frame_length_samples
        frame_time = np.linspace(frame_start/sample_rate, frame_end/sample_rate, len(frames[i]))
        
        plt.plot(frame_time, frames[i], label=f'帧 {i}', linewidth=2)
    
    plt.title('Detailed Content of First Few Frames')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n分帧统计信息:")
    print(f"总帧数: {len(frames)}")
    print(f"每帧长度: {len(frames[0])} 采样点")
    print(f"帧长: {frame_length_ms} ms")
    print(f"帧移: {frame_shift_ms} ms")
    print(f"重叠率: {(frame_length_samples - frame_shift_samples) / frame_length_samples * 100:.1f}%")


if __name__ == "__main__":
    # 测试窗函数比较
    print("比较三种窗函数的特性...")
    compare_windows()
    
    # 如果有WAV文件，测试分帧功能
    import os
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if wav_files:
        from wav_reader import WAVReader
        
        test_file = wav_files[0]
        print(f"\n使用文件 {test_file} 测试分帧功能...")
        
        reader = WAVReader(test_file)
        audio_data, sample_rate = reader.read()
        
        # 可视化分帧过程
        visualize_framing(audio_data, sample_rate)
        
        # 测试不同窗函数的效果
        processor = FrameProcessor(sample_rate, 25.0, 10.0)
        frames, windowed_frames = processor.process_signal(audio_data, 'hamming')
        
        print(f"分帧完成，共 {len(frames)} 帧")
    else:
        print("\n当前目录下没有WAV文件，无法测试分帧功能")
        print("请将WAV文件放在当前目录下进行测试")
