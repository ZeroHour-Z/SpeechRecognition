"""
短时时域特性分析模块
实现语音信号的短时能量、短时平均幅度和短时过零率特征参数提取
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from .frame_window import FrameProcessor


class TimeDomainAnalyzer:
    """短时时域特性分析器"""
    
    def __init__(self, sample_rate: int, frame_length_ms: float = 25.0, 
                 frame_shift_ms: float = 10.0):
        """
        初始化时域分析器
        
        Args:
            sample_rate: 采样率
            frame_length_ms: 帧长（毫秒）
            frame_shift_ms: 帧移（毫秒）
        """
        self.sample_rate = sample_rate
        self.frame_processor = FrameProcessor(sample_rate, frame_length_ms, frame_shift_ms)
        
    def short_time_energy(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        计算短时能量
        
        短时能量定义为：E(n) = Σ[x(m)]²
        其中 x(m) 是第n帧的第m个采样点
        
        Args:
            frames: 分帧后的信号列表
            
        Returns:
            np.ndarray: 每帧的短时能量
        """
        energy = []
        for frame in frames:
            # 计算每帧的能量
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
        
        return np.array(energy)
    
    def short_time_average_amplitude(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        计算短时平均幅度
        
        短时平均幅度定义为：M(n) = Σ|x(m)|
        其中 x(m) 是第n帧的第m个采样点
        
        Args:
            frames: 分帧后的信号列表
            
        Returns:
            np.ndarray: 每帧的短时平均幅度
        """
        amplitude = []
        for frame in frames:
            # 计算每帧的平均幅度
            frame_amplitude = np.sum(np.abs(frame))
            amplitude.append(frame_amplitude)
        
        return np.array(amplitude)
    
    def short_time_zero_crossing_rate(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        计算短时过零率
        
        短时过零率定义为：Z(n) = (1/2N) * Σ|sgn[x(m)] - sgn[x(m-1)]|
        其中 sgn[x] 是符号函数，N是帧长
        
        Args:
            frames: 分帧后的信号列表
            
        Returns:
            np.ndarray: 每帧的短时过零率
        """
        zcr = []
        for frame in frames:
            # 计算符号函数
            sign_frame = np.sign(frame)
            
            # 计算相邻采样点符号变化的次数
            sign_changes = np.sum(np.abs(np.diff(sign_frame)))
            
            # 归一化（除以2倍帧长）
            frame_zcr = sign_changes / (2 * len(frame))
            zcr.append(frame_zcr)
        
        return np.array(zcr)
    
    def analyze_signal(self, signal: np.ndarray, window_type: str = 'hamming') -> dict:
        """
        完整的时域分析流程
        
        Args:
            signal: 输入信号
            window_type: 窗函数类型
            
        Returns:
            dict: 包含所有时域特征的字典
        """
        # 分帧和加窗
        frames, windowed_frames = self.frame_processor.process_signal(signal, window_type)
        
        # 计算时域特征
        energy = self.short_time_energy(windowed_frames)
        amplitude = self.short_time_average_amplitude(windowed_frames)
        zcr = self.short_time_zero_crossing_rate(windowed_frames)
        
        # 计算时间轴
        frame_shift_samples = self.frame_processor.frame_shift
        time_axis = np.arange(len(energy)) * frame_shift_samples / self.sample_rate
        
        return {
            'frames': frames,
            'windowed_frames': windowed_frames,
            'energy': energy,
            'amplitude': amplitude,
            'zcr': zcr,
            'time_axis': time_axis,
            'frame_length': len(frames[0]) if frames else 0,
            'num_frames': len(frames)
        }
    
    def plot_time_domain_features(self, signal: np.ndarray, analysis_result: dict, 
                                window_type: str = 'hamming', 
                                start_time: float = 0.0, duration: float = 2.0) -> None:
        """
        绘制时域特征图
        
        Args:
            signal: 原始信号
            analysis_result: 分析结果
            window_type: 窗函数类型
            start_time: 显示开始时间
            duration: 显示时长
        """
        # 计算显示范围
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        display_signal = signal[start_sample:end_sample]
        display_time = np.linspace(start_time, start_time + duration, len(display_signal))
        
        # 计算显示范围内的特征
        start_frame = int(start_time * self.sample_rate / self.frame_processor.frame_shift)
        end_frame = int((start_time + duration) * self.sample_rate / self.frame_processor.frame_shift)
        
        display_energy = analysis_result['energy'][start_frame:end_frame]
        display_amplitude = analysis_result['amplitude'][start_frame:end_frame]
        display_zcr = analysis_result['zcr'][start_frame:end_frame]
        display_feature_time = analysis_result['time_axis'][start_frame:end_frame]
        
        # 创建图形
        plt.figure(figsize=(15, 12))
        
        # 原始信号
        plt.subplot(4, 1, 1)
        plt.plot(display_time, display_signal, 'b-', linewidth=1)
        plt.title(f'Original Speech Signal ({window_type} Window)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # 短时能量
        plt.subplot(4, 1, 2)
        plt.plot(display_feature_time, display_energy, 'r-', linewidth=2)
        plt.title('Short-time Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.grid(True, alpha=0.3)
        
        # 短时平均幅度
        plt.subplot(4, 1, 3)
        plt.plot(display_feature_time, display_amplitude, 'g-', linewidth=2)
        plt.title('Short-time Average Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Average Magnitude')
        plt.grid(True, alpha=0.3)
        
        # 短时过零率
        plt.subplot(4, 1, 4)
        plt.plot(display_feature_time, display_zcr, 'm-', linewidth=2)
        plt.title('Short-time Zero Crossing Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Zero Crossing Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\n时域特征统计信息 ({window_type}窗):")
        print("=" * 50)
        print(f"总帧数: {analysis_result['num_frames']}")
        print(f"帧长: {analysis_result['frame_length']} 采样点")
        print(f"短时能量 - 最大值: {analysis_result['energy'].max():.6f}, 最小值: {analysis_result['energy'].min():.6f}")
        print(f"短时平均幅度 - 最大值: {analysis_result['amplitude'].max():.6f}, 最小值: {analysis_result['amplitude'].min():.6f}")
        print(f"短时过零率 - 最大值: {analysis_result['zcr'].max():.4f}, 最小值: {analysis_result['zcr'].min():.4f}")
        print("=" * 50)


def compare_window_effects(signal: np.ndarray, sample_rate: int) -> None:
    """
    比较不同窗函数对时域特征的影响
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
    """
    window_types = ['rectangular', 'hamming', 'hanning']
    colors = ['b-', 'r-', 'g-']
    
    plt.figure(figsize=(15, 10))
    
    for i, window_type in enumerate(window_types):
        analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
        result = analyzer.analyze_signal(signal, window_type)
        
        # 短时能量
        plt.subplot(2, 2, 1)
        plt.plot(result['time_axis'], result['energy'], colors[i], 
                linewidth=2, label=f'{window_type}窗')
        plt.title('Short-time Energy Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 短时平均幅度
        plt.subplot(2, 2, 2)
        plt.plot(result['time_axis'], result['amplitude'], colors[i], 
                linewidth=2, label=f'{window_type}窗')
        plt.title('Short-time Average Magnitude Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Average Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 短时过零率
        plt.subplot(2, 2, 3)
        plt.plot(result['time_axis'], result['zcr'], colors[i], 
                linewidth=2, label=f'{window_type}窗')
        plt.title('Short-time Zero Crossing Rate Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Zero Crossing Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 特征对比表
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 创建对比表格
    table_data = []
    for window_type in window_types:
        analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
        result = analyzer.analyze_signal(signal, window_type)
        
        energy_std = np.std(result['energy'])
        amplitude_std = np.std(result['amplitude'])
        zcr_std = np.std(result['zcr'])
        
        table_data.append([
            window_type,
            f"{energy_std:.4f}",
            f"{amplitude_std:.4f}",
            f"{zcr_std:.4f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['窗函数', '能量标准差', '幅度标准差', '过零率标准差'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Feature Stability Comparison of Different Windows')
    
    plt.tight_layout()
    plt.show()


def analyze_voiced_unvoiced_regions(signal: np.ndarray, sample_rate: int) -> None:
    """
    分析语音信号的浊音和清音区域
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
    """
    analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
    result = analyzer.analyze_signal(signal, 'hamming')
    
    # 计算阈值
    energy_threshold = np.mean(result['energy']) * 0.1
    zcr_threshold = np.mean(result['zcr']) * 1.5
    
    # 分类语音段
    voiced_frames = (result['energy'] > energy_threshold) & (result['zcr'] < zcr_threshold)
    unvoiced_frames = (result['energy'] > energy_threshold) & (result['zcr'] >= zcr_threshold)
    silence_frames = result['energy'] <= energy_threshold
    
    # 绘制分析结果
    plt.figure(figsize=(15, 8))
    
    # 原始信号
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(signal) / sample_rate, len(signal))
    plt.plot(time_axis, signal, 'b-', linewidth=1)
    plt.title('Original Speech Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 短时能量和过零率
    plt.subplot(3, 1, 2)
    plt.plot(result['time_axis'], result['energy'], 'r-', linewidth=2, label='短时能量')
    plt.axhline(y=energy_threshold, color='r', linestyle='--', alpha=0.7, label='能量阈值')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(result['time_axis'], result['zcr'], 'g-', linewidth=2, label='短时过零率')
    plt.axhline(y=zcr_threshold, color='g', linestyle='--', alpha=0.7, label='过零率阈值')
    plt.xlabel('Time (s)')
    plt.ylabel('Zero Crossing Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 统计信息
    total_frames = len(result['energy'])
    voiced_count = np.sum(voiced_frames)
    unvoiced_count = np.sum(unvoiced_frames)
    silence_count = np.sum(silence_frames)
    
    print(f"\n语音段分类结果:")
    print("=" * 40)
    print(f"总帧数: {total_frames}")
    print(f"浊音帧: {voiced_count} ({voiced_count/total_frames*100:.1f}%)")
    print(f"清音帧: {unvoiced_count} ({unvoiced_count/total_frames*100:.1f}%)")
    print(f"静音帧: {silence_count} ({silence_count/total_frames*100:.1f}%)")
    print(f"能量阈值: {energy_threshold:.6f}")
    print(f"过零率阈值: {zcr_threshold:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    # 测试代码
    import os
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if wav_files:
        from wav_reader import WAVReader
        
        test_file = wav_files[0]
        print(f"使用文件 {test_file} 测试时域分析功能...")
        
        reader = WAVReader(test_file)
        audio_data, sample_rate = reader.read()
        
        # 创建分析器
        analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
        
        # 分析信号
        result = analyzer.analyze_signal(audio_data, 'hamming')
        
        # 绘制时域特征
        analyzer.plot_time_domain_features(audio_data, result, 'hamming')
        
        # 比较不同窗函数的效果
        compare_window_effects(audio_data, sample_rate)
        
        # 分析浊音和清音区域
        analyze_voiced_unvoiced_regions(audio_data, sample_rate)
        
    else:
        print("当前目录下没有WAV文件，无法测试时域分析功能")
        print("请将WAV文件放在当前目录下进行测试")
