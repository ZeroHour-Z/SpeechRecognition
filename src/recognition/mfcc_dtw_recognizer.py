"""
基于 MFCC 特征和 DTW 的语音识别器 - 实验三
实现频域语音识别，使用实验二的 MFCC 特征和 DTW 算法进行模板匹配
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from ..core.wav_reader import WAVReader
from ..core.frequency_domain_analysis import FrequencyDomainAnalyzer
from ..core.endpoint_detection import DualThresholdEndpointDetector
from .dtw import DTW


class MFCCDTWRecognizer:
    """
    基于 MFCC 特征和 DTW 的语音识别器
    
    实现流程：
    1. 语音输入
    2. 去噪（可选）
    3. 端点检测
    4. 逐帧进行特征提取（使用实验二的 MFCC）
    5. 模版匹配（使用 DTW）
    6. 输出识别结果
    """
    
    def __init__(self, sample_rate: int = 16000, 
                 frame_length_ms: float = 25.0,
                 frame_shift_ms: float = 10.0,
                 n_mels: int = 26,
                 n_mfcc: int = 13,
                 dtw_window: Optional[int] = None):
        """
        初始化识别器
        
        Args:
            sample_rate: 采样率
            frame_length_ms: 帧长（毫秒）
            frame_shift_ms: 帧移（毫秒）
            n_mels: Mel 滤波器数量
            n_mfcc: MFCC 系数数量
            dtw_window: DTW 搜索窗口大小（None 表示不限制）
        """
        self.sample_rate = sample_rate
        self.frequency_analyzer = FrequencyDomainAnalyzer(
            sample_rate=sample_rate,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms,
            n_mels=n_mels,
            n_mfcc=n_mfcc
        )
        self.endpoint_detector = DualThresholdEndpointDetector(
            sample_rate=sample_rate,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms
        )
        self.dtw = DTW(distance_metric='euclidean')
        self.dtw_window = dtw_window
        
        # 存储训练模板（每个数字的 MFCC 特征序列列表）
        self.templates = {}  # {digit: [mfcc_sequence1, mfcc_sequence2, ...]}
        
        print(f"MFCC-DTW 识别器初始化:")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  帧长: {frame_length_ms} ms")
        print(f"  帧移: {frame_shift_ms} ms")
        print(f"  Mel 滤波器数: {n_mels}")
        print(f"  MFCC 系数数: {n_mfcc}")
        print(f"  DTW 窗口: {dtw_window if dtw_window else '无限制'}")
    
    def extract_mfcc_features(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """
        提取语音信号的 MFCC 特征
        
        Args:
            signal: 输入信号
            
        Returns:
            Optional[np.ndarray]: MFCC 特征序列，形状为 (n_frames, n_mfcc)，如果失败返回 None
        """
        try:
            # 端点检测
            detection_result = self.endpoint_detector.detect_endpoints(signal)
            
            if not detection_result['endpoints']:
                return None
            
            # 提取语音段
            speech_segments = self.endpoint_detector.extract_speech_segments(
                signal, detection_result
            )
            
            if not speech_segments:
                return None
            
            # 使用最长的语音段
            main_segment = max(speech_segments, key=len)
            
            # 提取 MFCC 特征
            mfcc_result = self.frequency_analyzer.extract_mfcc(main_segment)
            mfcc_features = mfcc_result['mfcc']  # 形状: (n_frames, n_mfcc)
            
            return mfcc_features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def train(self, training_data_dir: str) -> None:
        """
        训练识别器
        
        读取训练数据，提取 MFCC 特征，存储为模板
        
        Args:
            training_data_dir: 训练数据目录，应包含 digit_0, digit_1, ..., digit_9 子目录
        """
        print("开始训练 MFCC-DTW 语音识别器...")
        print("=" * 60)
        
        self.templates = {}
        
        # 扫描训练数据
        for digit in range(10):
            digit_dir = os.path.join(training_data_dir, f"digit_{digit}")
            if not os.path.exists(digit_dir):
                print(f"警告: 未找到数字 {digit} 的训练数据目录: {digit_dir}")
                continue
            
            # 读取该数字的所有训练样本
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            if not wav_files:
                print(f"警告: 数字 {digit} 目录下没有WAV文件")
                continue
            
            templates_list = []
            for wav_file in wav_files:
                wav_path = os.path.join(digit_dir, wav_file)
                try:
                    reader = WAVReader(wav_path)
                    audio_data, file_sample_rate = reader.read()
                    
                    # 如果采样率不匹配，进行重采样（简单实现）
                    if file_sample_rate != self.sample_rate:
                        # 简单的线性插值重采样
                        from scipy import signal as scipy_signal
                        num_samples = int(len(audio_data) * self.sample_rate / file_sample_rate)
                        audio_data = scipy_signal.resample(audio_data, num_samples)
                    
                    # 提取 MFCC 特征
                    mfcc_features = self.extract_mfcc_features(audio_data)
                    
                    if mfcc_features is not None and len(mfcc_features) > 0:
                        templates_list.append(mfcc_features)
                        print(f"  ✓ {wav_file}: 提取成功，{len(mfcc_features)} 帧")
                    else:
                        print(f"  ✗ {wav_file}: 特征提取失败")
                        
                except Exception as e:
                    print(f"  ✗ {wav_file}: 错误 - {e}")
            
            if templates_list:
                self.templates[digit] = templates_list
                print(f"数字 {digit}: 训练样本 {len(templates_list)} 个，模板创建成功")
            else:
                print(f"数字 {digit}: 没有有效的训练样本")
        
        print("=" * 60)
        print(f"训练完成，共训练 {len(self.templates)} 个数字")
        for digit, templates in self.templates.items():
            print(f"  数字 {digit}: {len(templates)} 个模板")
    
    def recognize(self, signal: np.ndarray) -> Tuple[int, float, Dict[int, float]]:
        """
        识别语音信号
        
        使用 DTW 算法计算测试信号与所有模板的距离，选择距离最小的数字
        
        Args:
            signal: 输入信号
            
        Returns:
            Tuple[int, float, Dict[int, float]]:
                - 识别结果（数字 0-9，-1 表示识别失败）
                - 置信度（归一化的相似度分数）
                - 所有数字的距离字典
        """
        if not self.templates:
            raise ValueError("识别器尚未训练，请先调用train()方法")
        
        # 提取 MFCC 特征
        mfcc_features = self.extract_mfcc_features(signal)
        
        if mfcc_features is None or len(mfcc_features) == 0:
            return -1, 0.0, {}
        
        # 计算与每个数字的所有模板的最小 DTW 距离
        distances = {}
        
        for digit, templates in self.templates.items():
            min_distance = float('inf')
            
            # 与所有模板计算 DTW 距离，取最小值
            for template in templates:
                try:
                    distance, _, _ = self.dtw.dtw(
                        mfcc_features, 
                        template, 
                        window=self.dtw_window
                    )
                    min_distance = min(min_distance, distance)
                except Exception as e:
                    print(f"计算 DTW 距离时出错 (数字 {digit}): {e}")
                    continue
            
            if min_distance != float('inf'):
                distances[digit] = min_distance
        
        if not distances:
            return -1, 0.0, {}
        
        # 找到距离最小的数字
        best_digit = min(distances.keys(), key=lambda x: distances[x])
        min_distance = distances[best_digit]
        
        # 计算置信度（将距离转换为相似度分数）
        # 使用 softmax 归一化
        if len(distances) > 1:
            # 计算所有距离的指数（距离越小，指数越大）
            max_distance = max(distances.values())
            exp_scores = {d: np.exp(-distances[d] / (max_distance + 1e-8)) 
                         for d in distances}
            total_exp = sum(exp_scores.values())
            confidence = exp_scores[best_digit] / total_exp if total_exp > 0 else 0.0
        else:
            confidence = 1.0
        
        return best_digit, confidence, distances
    
    def recognize_with_path(self, signal: np.ndarray, 
                           digit: Optional[int] = None) -> Tuple[int, float, Dict, List]:
        """
        识别语音信号并返回 DTW 路径（用于可视化）
        
        Args:
            signal: 输入信号
            digit: 如果指定，则只计算与该数字的 DTW 路径
            
        Returns:
            Tuple[int, float, Dict, List]:
                - 识别结果
                - 置信度
                - 所有数字的距离字典
                - DTW 路径（如果 digit 指定）
        """
        if not self.templates:
            raise ValueError("识别器尚未训练，请先调用train()方法")
        
        # 提取 MFCC 特征
        mfcc_features = self.extract_mfcc_features(signal)
        
        if mfcc_features is None or len(mfcc_features) == 0:
            return -1, 0.0, {}, []
        
        # 计算与每个数字的所有模板的最小 DTW 距离
        distances = {}
        dtw_path = []
        
        if digit is not None and digit in self.templates:
            # 只计算指定数字的 DTW 路径
            min_distance = float('inf')
            best_template = None
            
            for template in self.templates[digit]:
                try:
                    distance, accumulated_cost, path = self.dtw.dtw(
                        mfcc_features, 
                        template, 
                        window=self.dtw_window
                    )
                    if distance < min_distance:
                        min_distance = distance
                        best_template = template
                        dtw_path = path
                except Exception as e:
                    continue
            
            if min_distance != float('inf'):
                distances[digit] = min_distance
        else:
            # 计算所有数字的距离
            for digit_key, templates in self.templates.items():
                min_distance = float('inf')
                
                for template in templates:
                    try:
                        distance, _, _ = self.dtw.dtw(
                            mfcc_features, 
                            template, 
                            window=self.dtw_window
                        )
                        min_distance = min(min_distance, distance)
                    except Exception as e:
                        continue
                
                if min_distance != float('inf'):
                    distances[digit_key] = min_distance
        
        if not distances:
            return -1, 0.0, {}, []
        
        # 找到距离最小的数字
        best_digit = min(distances.keys(), key=lambda x: distances[x])
        min_distance = distances[best_digit]
        
        # 计算置信度
        if len(distances) > 1:
            max_distance = max(distances.values())
            exp_scores = {d: np.exp(-distances[d] / (max_distance + 1e-8)) 
                         for d in distances}
            total_exp = sum(exp_scores.values())
            confidence = exp_scores[best_digit] / total_exp if total_exp > 0 else 0.0
        else:
            confidence = 1.0
        
        return best_digit, confidence, distances, dtw_path
    
    def test_recognition(self, test_data_dir: str) -> Dict[str, float]:
        """
        测试识别准确率
        
        Args:
            test_data_dir: 测试数据目录
            
        Returns:
            Dict[str, float]: 测试结果
        """
        print("开始测试 MFCC-DTW 语音识别准确率...")
        print("=" * 60)
        
        total_tests = 0
        correct_predictions = 0
        digit_results = {}
        
        for digit in range(10):
            digit_dir = os.path.join(test_data_dir, f"digit_{digit}")
            if not os.path.exists(digit_dir):
                continue
            
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            digit_tests = 0
            digit_correct = 0
            
            for wav_file in wav_files:
                wav_path = os.path.join(digit_dir, wav_file)
                try:
                    reader = WAVReader(wav_path)
                    audio_data, file_sample_rate = reader.read()
                    
                    # 如果采样率不匹配，进行重采样
                    if file_sample_rate != self.sample_rate:
                        from scipy import signal as scipy_signal
                        num_samples = int(len(audio_data) * self.sample_rate / file_sample_rate)
                        audio_data = scipy_signal.resample(audio_data, num_samples)
                    
                    predicted_digit, confidence, distances = self.recognize(audio_data)
                    
                    digit_tests += 1
                    total_tests += 1
                    
                    if predicted_digit == digit:
                        digit_correct += 1
                        correct_predictions += 1
                        print(f"  ✓ {wav_file}: 预测={predicted_digit}, 实际={digit}, "
                              f"置信度={confidence:.3f}, 距离={distances[predicted_digit]:.2f}")
                    else:
                        print(f"  ✗ {wav_file}: 预测={predicted_digit}, 实际={digit}, "
                              f"置信度={confidence:.3f}, 距离={distances[predicted_digit]:.2f}")
                        
                except Exception as e:
                    print(f"  ✗ {wav_file}: 错误 - {e}")
            
            if digit_tests > 0:
                accuracy = digit_correct / digit_tests
                digit_results[f"digit_{digit}"] = accuracy
                print(f"数字 {digit}: 准确率 {accuracy:.2%} ({digit_correct}/{digit_tests})")
        
        overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0.0
        digit_results['overall'] = overall_accuracy
        
        print("=" * 60)
        print(f"总体准确率: {overall_accuracy:.2%} ({correct_predictions}/{total_tests})")
        
        return digit_results


if __name__ == "__main__":
    # 测试代码
    recognizer = MFCCDTWRecognizer(sample_rate=16000, n_mels=26, n_mfcc=13)
    
    # 如果有训练数据，可以取消注释以下代码
    # recognizer.train("data/train")
    # results = recognizer.test_recognition("data/test")

