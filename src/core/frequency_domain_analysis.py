"""
频域分析模块 - 实验二
实现短时傅里叶变换（STFT）和 Mel 频率倒谱系数（MFCC）特征提取
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy.fft import dct
from .frame_window import FrameProcessor

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FrequencyDomainAnalyzer:
    """频域分析器 - 实现 STFT 和 MFCC 特征提取"""
    
    def __init__(self, sample_rate: int, frame_length_ms: float = 25.0,
                 frame_shift_ms: float = 10.0, n_fft: int = None,
                 n_mels: int = 26, n_mfcc: int = 13):
        """
        初始化频域分析器
        
        Args:
            sample_rate: 采样率
            frame_length_ms: 帧长（毫秒）
            frame_shift_ms: 帧移（毫秒）
            n_fft: FFT 点数，如果为 None 则使用帧长
            n_mels: Mel 滤波器数量
            n_mfcc: MFCC 系数数量
        """
        self.sample_rate = sample_rate
        self.frame_processor = FrameProcessor(sample_rate, frame_length_ms, frame_shift_ms)
        
        # FFT 参数
        if n_fft is None:
            self.n_fft = self.frame_processor.frame_length
        else:
            self.n_fft = n_fft
        
        # MFCC 参数
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # 预加重系数（通常为 0.97）
        self.pre_emphasis_coeff = 0.97
        
        # 构建 Mel 滤波器组
        self.mel_filters = self._create_mel_filterbank()
        
        print(f"频域分析器初始化:")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  帧长: {frame_length_ms} ms ({self.frame_processor.frame_length} 采样点)")
        print(f"  帧移: {frame_shift_ms} ms ({self.frame_processor.frame_shift} 采样点)")
        print(f"  FFT 点数: {self.n_fft}")
        print(f"  Mel 滤波器数: {self.n_mels}")
        print(f"  MFCC 系数数: {self.n_mfcc}")
    
    def pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        """
        预加重处理
        
        预加重用于补偿高频分量的衰减，提高高频部分的能量
        公式: y(n) = x(n) - α * x(n-1)
        其中 α 通常取 0.97
        
        Args:
            signal: 输入信号
            
        Returns:
            np.ndarray: 预加重后的信号
        """
        if len(signal) < 2:
            return signal
        
        # 使用一阶差分滤波器实现预加重
        emphasized = np.zeros_like(signal)
        emphasized[0] = signal[0]
        emphasized[1:] = signal[1:] - self.pre_emphasis_coeff * signal[:-1]
        
        return emphasized
    
    def stft(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        短时傅里叶变换（STFT）
        
        对每一帧信号进行 FFT，得到频谱
        使用 numpy 的 FFT 实现，但展示原理
        
        Args:
            frames: 分帧后的信号列表（已加窗）
            
        Returns:
            np.ndarray: 频谱矩阵，形状为 (n_frames, n_fft//2 + 1)
        """
        spectra = []
        
        for frame in frames:
            # 如果帧长度小于 n_fft，进行零填充
            if len(frame) < self.n_fft:
                padded_frame = np.zeros(self.n_fft)
                padded_frame[:len(frame)] = frame
            else:
                padded_frame = frame[:self.n_fft]
            
            # 执行 FFT
            # FFT 原理：将时域信号转换为频域表示
            # X(k) = Σ[n=0 to N-1] x(n) * e^(-j*2π*k*n/N)
            fft_result = np.fft.fft(padded_frame, n=self.n_fft)
            
            # 取前 n_fft//2 + 1 个点（正频率部分）
            magnitude_spectrum = np.abs(fft_result[:self.n_fft//2 + 1])
            spectra.append(magnitude_spectrum)
        
        return np.array(spectra)
    
    def compute_power_spectrum(self, spectra: np.ndarray) -> np.ndarray:
        """
        计算功率谱（谱线能量）
        
        功率谱 = |X(k)|²，表示每个频率分量的能量
        
        Args:
            spectra: 频谱幅度，形状为 (n_frames, n_fft//2 + 1)
            
        Returns:
            np.ndarray: 功率谱，形状与 spectra 相同
        """
        # 功率谱 = 幅度谱的平方
        power_spectrum = spectra ** 2
        
        return power_spectrum
    
    def _hz_to_mel(self, hz: float) -> float:
        """
        将频率从 Hz 转换为 Mel 刻度
        
        Mel 刻度是人耳对音高的感知尺度
        公式: mel = 2595 * log10(1 + hz / 700)
        
        Args:
            hz: 频率（Hz）
            
        Returns:
            float: Mel 频率
        """
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    
    def _mel_to_hz(self, mel: float) -> float:
        """
        将频率从 Mel 刻度转换为 Hz
        
        Args:
            mel: Mel 频率
            
        Returns:
            float: 频率（Hz）
        """
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """
        创建 Mel 滤波器组
        
        Mel 滤波器组用于模拟人耳对频率的感知特性
        在低频区域分辨率高，在高频区域分辨率低
        
        Returns:
            np.ndarray: Mel 滤波器组，形状为 (n_mels, n_fft//2 + 1)
        """
        # 计算 Mel 频率范围
        low_freq_mel = self._hz_to_mel(0)
        high_freq_mel = self._hz_to_mel(self.sample_rate / 2)
        
        # 在 Mel 刻度上均匀分布滤波器中心频率
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        
        # 转换回 Hz
        hz_points = self._mel_to_hz(mel_points)
        
        # 转换为 FFT bin 索引
        fft_bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # 创建滤波器组
        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        
        for i in range(self.n_mels):
            # 滤波器的起始、中心、结束频率
            f_start = fft_bins[i]
            f_center = fft_bins[i + 1]
            f_end = fft_bins[i + 2]
            
            # 上升沿（从 f_start 到 f_center）
            if f_start < len(filters[i]):
                for k in range(f_start, min(f_center, len(filters[i]))):
                    filters[i, k] = (k - f_start) / (f_center - f_start) if f_center > f_start else 0
            
            # 下降沿（从 f_center 到 f_end）
            if f_center < len(filters[i]):
                for k in range(f_center, min(f_end, len(filters[i]))):
                    filters[i, k] = (f_end - k) / (f_end - f_center) if f_end > f_center else 0
        
        return filters
    
    def apply_mel_filterbank(self, power_spectrum: np.ndarray) -> np.ndarray:
        """
        应用 Mel 滤波器组，计算 Mel 频率能量
        
        将功率谱通过 Mel 滤波器组，得到每个 Mel 滤波器的能量
        
        Args:
            power_spectrum: 功率谱，形状为 (n_frames, n_fft//2 + 1)
            
        Returns:
            np.ndarray: Mel 频率能量，形状为 (n_frames, n_mels)
        """
        # 对每一帧应用 Mel 滤波器组
        mel_energies = []
        
        for frame_power in power_spectrum:
            # 每个滤波器的能量 = Σ(功率谱 * 滤波器响应)
            mel_energy = np.dot(self.mel_filters, frame_power)
            mel_energies.append(mel_energy)
        
        return np.array(mel_energies)
    
    def compute_mfcc(self, mel_energies: np.ndarray) -> np.ndarray:
        """
        计算 MFCC 系数
        
        通过离散余弦变换（DCT）将 Mel 频率能量转换为倒谱系数
        DCT 用于去相关，提取主要特征
        
        Args:
            mel_energies: Mel 频率能量，形状为 (n_frames, n_mels)
            
        Returns:
            np.ndarray: MFCC 系数，形状为 (n_frames, n_mfcc)
        """
        # 为了避免 log(0)，添加小的常数
        mel_energies = np.maximum(mel_energies, 1e-10)
        
        # 取对数（将乘法关系转换为加法关系）
        log_mel_energies = np.log(mel_energies)
        
        # 离散余弦变换（DCT）
        # DCT 用于去相关，提取主要特征
        # 使用 scipy 的 DCT 实现（类型 II）
        mfcc = dct(log_mel_energies, type=2, axis=1, norm='ortho')
        
        # 只取前 n_mfcc 个系数（通常为 13）
        # 第 0 个系数是直流分量，通常被丢弃或保留
        mfcc = mfcc[:, :self.n_mfcc]
        
        return mfcc
    
    def extract_mfcc(self, signal: np.ndarray, window_type: str = 'hamming') -> Dict:
        """
        完整的 MFCC 特征提取流程
        
        按照实验要求实现：
        1. 预处理：预加重，分帧，加窗
        2. FFT
        3. 计算谱线能量
        4. 计算 Mel 滤波器能量
        5. 经离散余弦变换 (DCT) 得到 MFCC 系数
        
        Args:
            signal: 输入语音信号
            window_type: 窗函数类型
            
        Returns:
            dict: 包含所有中间结果和最终 MFCC 特征的字典
        """
        # 步骤 1: 预处理
        # 1.1 预加重
        emphasized_signal = self.pre_emphasis(signal)
        
        # 1.2 分帧和加窗
        frames, windowed_frames = self.frame_processor.process_signal(
            emphasized_signal, window_type
        )
        
        # 步骤 2: FFT
        spectra = self.stft(windowed_frames)
        
        # 步骤 3: 计算谱线能量（功率谱）
        power_spectrum = self.compute_power_spectrum(spectra)
        
        # 步骤 4: 计算 Mel 滤波器能量
        mel_energies = self.apply_mel_filterbank(power_spectrum)
        
        # 步骤 5: 经离散余弦变换 (DCT) 得到 MFCC 系数
        mfcc = self.compute_mfcc(mel_energies)
        
        # 计算频率轴和时间轴
        frequencies = np.fft.fftfreq(self.n_fft, 1.0 / self.sample_rate)[:self.n_fft//2 + 1]
        frame_shift_samples = self.frame_processor.frame_shift
        time_axis = np.arange(len(frames)) * frame_shift_samples / self.sample_rate
        
        return {
            'original_signal': signal,
            'emphasized_signal': emphasized_signal,
            'frames': frames,
            'windowed_frames': windowed_frames,
            'spectra': spectra,
            'power_spectrum': power_spectrum,
            'mel_energies': mel_energies,
            'mfcc': mfcc,
            'frequencies': frequencies,
            'time_axis': time_axis,
            'num_frames': len(frames),
            'frame_length': len(frames[0]) if frames else 0
        }
    
    def plot_mfcc_extraction_process(self, result: Dict, 
                                     start_frame: int = 0, 
                                     num_frames: int = 10) -> None:
        """
        可视化 MFCC 提取过程
        
        Args:
            result: extract_mfcc 返回的结果字典
            start_frame: 起始帧索引
            num_frames: 显示的帧数
        """
        end_frame = min(start_frame + num_frames, result['num_frames'])
        
        plt.figure(figsize=(16, 12))
        
        # 1. 原始信号和预加重信号
        plt.subplot(3, 3, 1)
        time_orig = np.linspace(0, len(result['original_signal']) / self.sample_rate,
                               len(result['original_signal']))
        plt.plot(time_orig, result['original_signal'], 'b-', linewidth=1, label='原始信号')
        plt.plot(time_orig, result['emphasized_signal'], 'r-', linewidth=1, alpha=0.7, label='预加重信号')
        plt.title('预处理：预加重')
        plt.xlabel('时间 (s)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 分帧和加窗（显示第一帧）
        plt.subplot(3, 3, 2)
        if result['frames']:
            frame_time = np.arange(len(result['frames'][0])) / self.sample_rate
            plt.plot(frame_time, result['frames'][0], 'b-', linewidth=1, label='原始帧')
            plt.plot(frame_time, result['windowed_frames'][0], 'r-', linewidth=1, label='加窗帧')
            plt.title('预处理：分帧和加窗（第1帧）')
            plt.xlabel('时间 (s)')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. 频谱（显示第一帧）
        plt.subplot(3, 3, 3)
        if len(result['spectra']) > 0:
            plt.plot(result['frequencies'], result['spectra'][0], 'g-', linewidth=1)
            plt.title('FFT：幅度频谱（第1帧）')
            plt.xlabel('频率 (Hz)')
            plt.ylabel('幅度')
            plt.grid(True, alpha=0.3)
        
        # 4. 功率谱（显示第一帧）
        plt.subplot(3, 3, 4)
        if len(result['power_spectrum']) > 0:
            plt.plot(result['frequencies'], result['power_spectrum'][0], 'm-', linewidth=1)
            plt.title('谱线能量：功率谱（第1帧）')
            plt.xlabel('频率 (Hz)')
            plt.ylabel('能量')
            plt.grid(True, alpha=0.3)
        
        # 5. Mel 滤波器组
        plt.subplot(3, 3, 5)
        for i in range(min(5, self.n_mels)):
            plt.plot(result['frequencies'], self.mel_filters[i], linewidth=1, 
                    label=f'滤波器 {i+1}')
        plt.title('Mel 滤波器组（前5个）')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('增益')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Mel 频率能量（显示第一帧）
        plt.subplot(3, 3, 6)
        if len(result['mel_energies']) > 0:
            mel_indices = np.arange(self.n_mels)
            plt.bar(mel_indices, result['mel_energies'][0], alpha=0.7)
            plt.title('Mel 滤波器能量（第1帧）')
            plt.xlabel('Mel 滤波器索引')
            plt.ylabel('能量')
            plt.grid(True, alpha=0.3)
        
        # 7. MFCC 系数（显示前几帧）
        plt.subplot(3, 3, 7)
        mfcc_display = result['mfcc'][start_frame:end_frame].T
        plt.imshow(mfcc_display, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='MFCC 系数值')
        plt.title(f'MFCC 系数（帧 {start_frame} 到 {end_frame-1}）')
        plt.xlabel('帧索引')
        plt.ylabel('MFCC 系数索引')
        
        # 8. MFCC 系数随时间变化（显示前几个系数）
        plt.subplot(3, 3, 8)
        for i in range(min(5, self.n_mfcc)):
            plt.plot(result['time_axis'], result['mfcc'][:, i], 
                    linewidth=1, label=f'MFCC {i}')
        plt.title('MFCC 系数随时间变化（前5个）')
        plt.xlabel('时间 (s)')
        plt.ylabel('MFCC 系数值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. MFCC 特征图（所有帧）
        plt.subplot(3, 3, 9)
        mfcc_all = result['mfcc'].T
        plt.imshow(mfcc_all, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='MFCC 系数值')
        plt.title('MFCC 特征图（所有帧）')
        plt.xlabel('帧索引')
        plt.ylabel('MFCC 系数索引')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\nMFCC 特征提取统计信息:")
        print("=" * 50)
        print(f"总帧数: {result['num_frames']}")
        print(f"帧长: {result['frame_length']} 采样点")
        print(f"FFT 点数: {self.n_fft}")
        print(f"Mel 滤波器数: {self.n_mels}")
        print(f"MFCC 系数数: {self.n_mfcc}")
        print(f"MFCC 特征形状: {result['mfcc'].shape}")
        print(f"MFCC 系数范围: [{result['mfcc'].min():.4f}, {result['mfcc'].max():.4f}]")
        print("=" * 50)


def compare_mfcc_parameters(signal: np.ndarray, sample_rate: int) -> None:
    """
    比较不同参数设置对 MFCC 特征的影响
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
    """
    # 不同的参数组合
    param_sets = [
        {'n_mels': 13, 'n_mfcc': 13, 'label': '标准参数'},
        {'n_mels': 26, 'n_mfcc': 13, 'label': '更多滤波器'},
        {'n_mels': 26, 'n_mfcc': 20, 'label': '更多系数'},
    ]
    
    plt.figure(figsize=(15, 10))
    
    for idx, params in enumerate(param_sets):
        analyzer = FrequencyDomainAnalyzer(
            sample_rate,
            n_mels=params['n_mels'],
            n_mfcc=params['n_mfcc']
        )
        result = analyzer.extract_mfcc(signal)
        
        # 显示 MFCC 特征图
        plt.subplot(2, 2, idx + 1)
        mfcc_display = result['mfcc'].T
        plt.imshow(mfcc_display, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='MFCC 系数值')
        plt.title(f"{params['label']} (Mel={params['n_mels']}, MFCC={params['n_mfcc']})")
        plt.xlabel('帧索引')
        plt.ylabel('MFCC 系数索引')
    
    # 比较第一个 MFCC 系数
    plt.subplot(2, 2, 4)
    for params in param_sets:
        analyzer = FrequencyDomainAnalyzer(
            sample_rate,
            n_mels=params['n_mels'],
            n_mfcc=params['n_mfcc']
        )
        result = analyzer.extract_mfcc(signal)
        plt.plot(result['time_axis'], result['mfcc'][:, 0], 
                linewidth=2, label=params['label'])
    plt.title('第一个 MFCC 系数对比')
    plt.xlabel('时间 (s)')
    plt.ylabel('MFCC 系数值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 测试代码
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from .wav_reader import WAVReader
    
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if wav_files:
        test_file = wav_files[0]
        print(f"使用文件 {test_file} 测试频域分析功能...")
        
        reader = WAVReader(test_file)
        audio_data, sample_rate = reader.read()
        
        # 创建频域分析器
        analyzer = FrequencyDomainAnalyzer(sample_rate, n_mels=26, n_mfcc=13)
        
        # 提取 MFCC 特征
        result = analyzer.extract_mfcc(audio_data)
        
        # 可视化提取过程
        analyzer.plot_mfcc_extraction_process(result)
        
        # 比较不同参数
        compare_mfcc_parameters(audio_data, sample_rate)
        
    else:
        print("当前目录下没有WAV文件，无法测试频域分析功能")
        print("请将WAV文件放在当前目录下进行测试")

