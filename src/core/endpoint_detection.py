"""
基于双门限法的端点检测模块
实现语音信号的端点检测，用于识别语音段的开始和结束位置
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from .time_domain_analysis import TimeDomainAnalyzer


class DualThresholdEndpointDetector:
    """双门限法端点检测器"""
    
    def __init__(self, sample_rate: int, frame_length_ms: float = 25.0, 
                 frame_shift_ms: float = 10.0):
        """
        初始化端点检测器
        
        Args:
            sample_rate: 采样率
            frame_length_ms: 帧长（毫秒）
            frame_shift_ms: 帧移（毫秒）
        """
        self.sample_rate = sample_rate
        self.analyzer = TimeDomainAnalyzer(sample_rate, frame_length_ms, frame_shift_ms)
        
    def calculate_thresholds(self, energy: np.ndarray, zcr: np.ndarray, 
                           energy_ratio: float = 0.1, zcr_ratio: float = 1.5) -> Dict[str, float]:
        """
        计算双门限法的阈值
        
        Args:
            energy: 短时能量序列
            zcr: 短时过零率序列
            energy_ratio: 能量阈值比例
            zcr_ratio: 过零率阈值比例
            
        Returns:
            Dict[str, float]: 包含各种阈值的字典
        """
        # 计算能量阈值
        max_energy = np.max(energy)
        min_energy = np.min(energy)
        energy_threshold_high = max_energy * energy_ratio
        energy_threshold_low = min_energy + (max_energy - min_energy) * energy_ratio * 0.1
        
        # 计算过零率阈值
        mean_zcr = np.mean(zcr)
        zcr_threshold = mean_zcr * zcr_ratio
        
        return {
            'energy_high': energy_threshold_high,
            'energy_low': energy_threshold_low,
            'zcr_threshold': zcr_threshold,
            'max_energy': max_energy,
            'min_energy': min_energy,
            'mean_zcr': mean_zcr
        }
    
    def detect_endpoints(self, signal: np.ndarray, 
                        energy_ratio: float = 0.1, zcr_ratio: float = 1.5,
                        min_speech_frames: int = 3, min_silence_frames: int = 3) -> Dict:
        """
        基于双门限法进行端点检测
        
        Args:
            signal: 输入信号
            energy_ratio: 能量阈值比例
            zcr_ratio: 过零率阈值比例
            min_speech_frames: 最小语音帧数
            min_silence_frames: 最小静音帧数
            
        Returns:
            Dict: 端点检测结果
        """
        # 进行时域分析
        analysis_result = self.analyzer.analyze_signal(signal, 'hamming')
        energy = analysis_result['short_time_energy']
        zcr = analysis_result['zero_crossing_rate']
        time_axis = analysis_result['time_axis']
        
        # 计算阈值
        thresholds = self.calculate_thresholds(energy, zcr, energy_ratio, zcr_ratio)
        
        # 第一级检测：基于能量高阈值
        high_energy_frames = energy > thresholds['energy_high']
        
        # 第二级检测：基于能量低阈值和过零率
        low_energy_frames = energy > thresholds['energy_low']
        low_zcr_frames = zcr < thresholds['zcr_threshold']
        
        # 组合条件：能量低阈值 OR (能量高阈值 AND 过零率低)
        speech_frames = low_energy_frames | (high_energy_frames & low_zcr_frames)
        
        # 平滑处理：去除过短的语音段和静音段
        speech_frames = self._smooth_detection(speech_frames, min_speech_frames, min_silence_frames)
        
        # 找到语音段的开始和结束点
        endpoints = self._find_endpoints(speech_frames, time_axis)
        
        return {
            'analysis_result': analysis_result,
            'thresholds': thresholds,
            'speech_frames': speech_frames,
            'endpoints': endpoints,
            'high_energy_frames': high_energy_frames,
            'low_energy_frames': low_energy_frames,
            'low_zcr_frames': low_zcr_frames
        }
    
    def _smooth_detection(self, speech_frames: np.ndarray, 
                         min_speech_frames: int, min_silence_frames: int) -> np.ndarray:
        """
        平滑端点检测结果，去除过短的语音段和静音段
        
        Args:
            speech_frames: 语音帧标记
            min_speech_frames: 最小语音帧数
            min_silence_frames: 最小静音帧数
            
        Returns:
            np.ndarray: 平滑后的语音帧标记
        """
        smoothed = speech_frames.copy()
        
        # 去除过短的语音段
        i = 0
        while i < len(smoothed):
            if smoothed[i]:  # 找到语音段的开始
                # 计算语音段长度
                speech_length = 0
                j = i
                while j < len(smoothed) and smoothed[j]:
                    speech_length += 1
                    j += 1
                
                # 如果语音段太短，标记为静音
                if speech_length < min_speech_frames:
                    smoothed[i:j] = False
                
                i = j
            else:
                i += 1
        
        # 去除过短的静音段
        i = 0
        while i < len(smoothed):
            if not smoothed[i]:  # 找到静音段的开始
                # 计算静音段长度
                silence_length = 0
                j = i
                while j < len(smoothed) and not smoothed[j]:
                    silence_length += 1
                    j += 1
                
                # 如果静音段太短，标记为语音
                if silence_length < min_silence_frames:
                    smoothed[i:j] = True
                
                i = j
            else:
                i += 1
        
        return smoothed
    
    def _find_endpoints(self, speech_frames: np.ndarray, time_axis: np.ndarray) -> List[Dict]:
        """
        找到语音段的端点
        
        Args:
            speech_frames: 语音帧标记
            time_axis: 时间轴
            
        Returns:
            List[Dict]: 端点列表
        """
        endpoints = []
        in_speech = False
        start_time = None
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # 语音段开始
                start_time = time_axis[i]
                in_speech = True
            elif not is_speech and in_speech:
                # 语音段结束
                end_time = time_axis[i-1] if i > 0 else time_axis[i]
                duration = end_time - start_time
                
                endpoints.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'start_frame': i - int(duration * self.sample_rate / self.analyzer.frame_processor.frame_shift),
                    'end_frame': i - 1
                })
                in_speech = False
        
        # 处理最后一个语音段
        if in_speech:
            end_time = time_axis[-1]
            duration = end_time - start_time
            endpoints.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_frame': len(speech_frames) - int(duration * self.sample_rate / self.analyzer.frame_processor.frame_shift),
                'end_frame': len(speech_frames) - 1
            })
        
        return endpoints
    
    def plot_endpoint_detection(self, signal: np.ndarray, detection_result: Dict, 
                              start_time: float = 0.0, duration: float = 3.0) -> None:
        """
        绘制端点检测结果
        
        Args:
            signal: 原始信号
            detection_result: 检测结果
            start_time: 显示开始时间
            duration: 显示时长
        """
        analysis_result = detection_result['analysis_result']
        thresholds = detection_result['thresholds']
        speech_frames = detection_result['speech_frames']
        endpoints = detection_result['endpoints']
        
        # 计算显示范围
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        display_signal = signal[start_sample:end_sample]
        display_time = np.linspace(start_time, start_time + duration, len(display_signal))
        
        # 计算显示范围内的特征
        start_frame = int(start_time * self.sample_rate / self.analyzer.frame_processor.frame_shift)
        end_frame = int((start_time + duration) * self.sample_rate / self.analyzer.frame_processor.frame_shift)
        
        display_energy = analysis_result['energy'][start_frame:end_frame]
        display_zcr = analysis_result['zcr'][start_frame:end_frame]
        display_speech = speech_frames[start_frame:end_frame]
        display_feature_time = analysis_result['time_axis'][start_frame:end_frame]
        
        # 创建图形
        plt.figure(figsize=(15, 12))
        
        # 原始信号和端点标记
        plt.subplot(4, 1, 1)
        plt.plot(display_time, display_signal, 'b-', linewidth=1, label='Original Signal')
        
        # 标记语音段（仅在可视范围内，并避免重复图例）
        shaded_label_used = False
        for endpoint in endpoints:
            # 与显示区间求交
            seg_start = max(endpoint['start_time'], start_time)
            seg_end = min(endpoint['end_time'], start_time + duration)
            if seg_end <= seg_start:
                continue
            label = None if shaded_label_used else 'Detected Speech Segment'
            shaded_label_used = True
            plt.axvspan(seg_start, seg_end, alpha=0.15, color='yellow', label=label, zorder=-1)
        
        plt.title('Endpoint Detection (Waveform)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 短时能量和阈值
        plt.subplot(4, 1, 2)
        plt.plot(display_feature_time, display_energy, 'r-', linewidth=2, label='Short-time Energy')
        plt.axhline(y=thresholds['energy_high'], color='r', linestyle='--', 
                   alpha=0.7, label=f'High Threshold: {thresholds["energy_high"]:.6f}')
        plt.axhline(y=thresholds['energy_low'], color='orange', linestyle='--', 
                   alpha=0.7, label=f'Low Threshold: {thresholds["energy_low"]:.6f}')
        plt.title('Short-time Energy and Thresholds')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 短时过零率和阈值
        plt.subplot(4, 1, 3)
        plt.plot(display_feature_time, display_zcr, 'g-', linewidth=2, label='Zero Crossing Rate')
        plt.axhline(y=thresholds['zcr_threshold'], color='g', linestyle='--', 
                   alpha=0.7, label=f'Threshold: {thresholds["zcr_threshold"]:.4f}')
        plt.title('Zero Crossing Rate and Threshold')
        plt.xlabel('Time (s)')
        plt.ylabel('ZCR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 语音段检测结果
        plt.subplot(4, 1, 4)
        plt.plot(display_feature_time, display_speech.astype(int), 'm-', linewidth=2, label='Speech Flag')
        plt.fill_between(display_feature_time, 0, display_speech.astype(int), 
                        alpha=0.25, color='magenta')
        plt.title('Speech/Silence Decision')
        plt.xlabel('Time (s)')
        plt.ylabel('Speech/Silence')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印检测结果
        print(f"\n端点检测结果:")
        print("=" * 60)
        print(f"检测到的语音段数量: {len(endpoints)}")
        print(f"能量高阈值: {thresholds['energy_high']:.6f}")
        print(f"能量低阈值: {thresholds['energy_low']:.6f}")
        print(f"过零率阈值: {thresholds['zcr_threshold']:.4f}")
        print("-" * 60)
        
        for i, endpoint in enumerate(endpoints):
            print(f"语音段 {i+1}:")
            print(f"  开始时间: {endpoint['start_time']:.3f} 秒")
            print(f"  结束时间: {endpoint['end_time']:.3f} 秒")
            print(f"  持续时间: {endpoint['duration']:.3f} 秒")
            print(f"  开始帧: {endpoint['start_frame']}")
            print(f"  结束帧: {endpoint['end_frame']}")
        
        print("=" * 60)
    
    def extract_speech_segments(self, signal: np.ndarray, detection_result: Dict) -> List[np.ndarray]:
        """
        提取检测到的语音段
        
        Args:
            signal: 原始信号
            detection_result: 检测结果
            
        Returns:
            List[np.ndarray]: 语音段列表
        """
        speech_segments = []
        endpoints = detection_result['endpoints']
        
        for endpoint in endpoints:
            start_sample = int(endpoint['start_time'] * self.sample_rate)
            end_sample = int(endpoint['end_time'] * self.sample_rate)
            segment = signal[start_sample:end_sample]
            speech_segments.append(segment)
        
        return speech_segments


def test_endpoint_detection_with_parameters(signal: np.ndarray, sample_rate: int) -> None:
    """
    测试不同参数下的端点检测效果
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
    """
    # 测试不同的能量阈值比例
    energy_ratios = [0.05, 0.1, 0.2, 0.3]
    zcr_ratios = [1.2, 1.5, 2.0, 2.5]
    
    plt.figure(figsize=(15, 10))
    
    for i, energy_ratio in enumerate(energy_ratios):
        detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
        result = detector.detect_endpoints(signal, energy_ratio=energy_ratio, zcr_ratio=1.5)
        
        plt.subplot(2, 2, i+1)
        time_axis = np.linspace(0, len(signal) / sample_rate, len(signal))
        plt.plot(time_axis, signal, 'b-', linewidth=1, alpha=0.7)
        
        # 标记语音段
        for endpoint in result['endpoints']:
            plt.axvspan(endpoint['start_time'], endpoint['end_time'], 
                       alpha=0.3, color='yellow')
        
        plt.title(f'Energy Ratio: {energy_ratio} ({len(result["endpoints"])} segments)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印参数对比
    print("\n不同参数下的端点检测结果对比:")
    print("=" * 80)
    print(f"{'能量阈值比例':<12} {'过零率比例':<12} {'语音段数量':<10} {'总语音时长(秒)':<15}")
    print("-" * 80)
    
    for energy_ratio in energy_ratios:
        for zcr_ratio in zcr_ratios:
            detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
            result = detector.detect_endpoints(signal, energy_ratio=energy_ratio, zcr_ratio=zcr_ratio)
            
            total_duration = sum(endpoint['duration'] for endpoint in result['endpoints'])
            print(f"{energy_ratio:<12} {zcr_ratio:<12} {len(result['endpoints']):<10} {total_duration:<15.3f}")
    
    print("=" * 80)


if __name__ == "__main__":
    # 测试代码
    import os
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if wav_files:
        from wav_reader import WAVReader
        
        test_file = wav_files[0]
        print(f"使用文件 {test_file} 测试端点检测功能...")
        
        reader = WAVReader(test_file)
        audio_data, sample_rate = reader.read()
        
        # 创建端点检测器
        detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
        
        # 进行端点检测
        result = detector.detect_endpoints(audio_data, energy_ratio=0.1, zcr_ratio=1.5)
        
        # 绘制检测结果
        detector.plot_endpoint_detection(audio_data, result)
        
        # 测试不同参数的效果
        test_endpoint_detection_with_parameters(audio_data, sample_rate)
        
        # 提取语音段
        speech_segments = detector.extract_speech_segments(audio_data, result)
        print(f"\n提取到 {len(speech_segments)} 个语音段")
        
        for i, segment in enumerate(speech_segments):
            print(f"语音段 {i+1} 长度: {len(segment)} 采样点 ({len(segment)/sample_rate:.3f} 秒)")
        
    else:
        print("当前目录下没有WAV文件，无法测试端点检测功能")
        print("请将WAV文件放在当前目录下进行测试")
