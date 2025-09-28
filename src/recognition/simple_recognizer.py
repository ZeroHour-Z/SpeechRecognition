"""
基于时域特征的简单语音识别器
使用短时能量、平均幅度、过零率等特征进行数字识别
"""

import numpy as np
import os
from typing import Dict, List, Tuple
from ..core.time_domain_analysis import TimeDomainAnalyzer
from ..core.endpoint_detection import DualThresholdEndpointDetector


class SimpleDigitRecognizer:
    """基于时域特征的简单数字识别器"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        初始化识别器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        self.time_analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
        self.endpoint_detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
        
        # 存储训练数据
        self.training_data = {}
        self.feature_templates = {}
        
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        提取语音信号的时域特征
        
        Args:
            signal: 输入信号
            
        Returns:
            Dict[str, float]: 特征字典
        """
        # 端点检测
        detection_result = self.endpoint_detector.detect_endpoints(signal)
        
        if not detection_result['endpoints']:
            return {}
        
        # 提取语音段
        speech_segments = self.endpoint_detector.extract_speech_segments(signal, detection_result)
        
        if not speech_segments:
            return {}
        
        # 使用最长的语音段
        main_segment = max(speech_segments, key=len)
        
        # 时域分析
        analysis_result = self.time_analyzer.analyze_signal(main_segment, 'hamming')
        
        # 计算统计特征
        energy = analysis_result['energy']
        amplitude = analysis_result['amplitude']
        zcr = analysis_result['zcr']
        
        features = {
            'max_energy': np.max(energy),
            'mean_energy': np.mean(energy),
            'energy_std': np.std(energy),
            'energy_range': np.max(energy) - np.min(energy),
            
            'max_amplitude': np.max(amplitude),
            'mean_amplitude': np.mean(amplitude),
            'amplitude_std': np.std(amplitude),
            'amplitude_range': np.max(amplitude) - np.min(amplitude),
            
            'max_zcr': np.max(zcr),
            'mean_zcr': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'zcr_range': np.max(zcr) - np.min(zcr),
            
            'duration': len(main_segment) / self.sample_rate,
            'energy_ratio': np.max(energy) / (np.mean(energy) + 1e-8),
            'amplitude_ratio': np.max(amplitude) / (np.mean(amplitude) + 1e-8)
        }
        
        return features
    
    def train(self, training_data_dir: str) -> None:
        """
        训练识别器
        
        Args:
            training_data_dir: 训练数据目录
        """
        print("开始训练语音识别器...")
        print("=" * 60)
        
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
            
            features_list = []
            for wav_file in wav_files:
                wav_path = os.path.join(digit_dir, wav_file)
                try:
                    from ..core.wav_reader import WAVReader
                    reader = WAVReader(wav_path)
                    audio_data, _ = reader.read()
                    
                    features = self.extract_features(audio_data)
                    if features:
                        features_list.append(features)
                        print(f"  处理: {wav_file} - 特征提取成功")
                    else:
                        print(f"  跳过: {wav_file} - 特征提取失败")
                        
                except Exception as e:
                    print(f"  错误: {wav_file} - {e}")
            
            if features_list:
                # 计算平均特征作为模板
                template = {}
                for key in features_list[0].keys():
                    values = [f[key] for f in features_list]
                    template[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                
                self.feature_templates[digit] = template
                print(f"数字 {digit}: 训练样本 {len(features_list)} 个，模板创建成功")
            else:
                print(f"数字 {digit}: 没有有效的训练样本")
        
        print("=" * 60)
        print(f"训练完成，共训练 {len(self.feature_templates)} 个数字")
    
    def recognize(self, signal: np.ndarray) -> Tuple[int, float]:
        """
        识别语音信号
        
        Args:
            signal: 输入信号
            
        Returns:
            Tuple[int, float]: (识别结果, 置信度)
        """
        if not self.feature_templates:
            raise ValueError("识别器尚未训练，请先调用train()方法")
        
        # 提取特征
        features = self.extract_features(signal)
        if not features:
            return -1, 0.0
        
        # 计算与每个模板的相似度
        similarities = {}
        for digit, template in self.feature_templates.items():
            similarity = self._calculate_similarity(features, template)
            similarities[digit] = similarity
        
        # 找到最相似的数字
        best_digit = max(similarities.keys(), key=lambda x: similarities[x])
        confidence = similarities[best_digit]
        
        return best_digit, confidence
    
    def _calculate_similarity(self, features: Dict[str, float], template: Dict[str, Dict[str, float]]) -> float:
        """
        计算特征与模板的相似度
        
        Args:
            features: 输入特征
            template: 模板特征
            
        Returns:
            float: 相似度分数
        """
        total_similarity = 0.0
        weight_sum = 0.0
        
        # 定义特征权重
        weights = {
            'max_energy': 0.15,
            'mean_energy': 0.10,
            'energy_std': 0.05,
            'max_amplitude': 0.15,
            'mean_amplitude': 0.10,
            'amplitude_std': 0.05,
            'max_zcr': 0.10,
            'mean_zcr': 0.10,
            'zcr_std': 0.05,
            'duration': 0.10,
            'energy_ratio': 0.05
        }
        
        for feature_name, weight in weights.items():
            if feature_name in features and feature_name in template:
                value = features[feature_name]
                template_mean = template[feature_name]['mean']
                template_std = template[feature_name]['std']
                
                # 使用高斯相似度
                if template_std > 0:
                    similarity = np.exp(-0.5 * ((value - template_mean) / template_std) ** 2)
                else:
                    similarity = 1.0 if abs(value - template_mean) < 1e-6 else 0.0
                
                total_similarity += weight * similarity
                weight_sum += weight
        
        return total_similarity / weight_sum if weight_sum > 0 else 0.0
    
    def test_recognition(self, test_data_dir: str) -> Dict[str, float]:
        """
        测试识别准确率
        
        Args:
            test_data_dir: 测试数据目录
            
        Returns:
            Dict[str, float]: 测试结果
        """
        print("开始测试语音识别准确率...")
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
                    from ..core.wav_reader import WAVReader
                    reader = WAVReader(wav_path)
                    audio_data, _ = reader.read()
                    
                    predicted_digit, confidence = self.recognize(audio_data)
                    
                    digit_tests += 1
                    total_tests += 1
                    
                    if predicted_digit == digit:
                        digit_correct += 1
                        correct_predictions += 1
                        print(f"  ✓ {wav_file}: 预测={predicted_digit}, 实际={digit}, 置信度={confidence:.3f}")
                    else:
                        print(f"  ✗ {wav_file}: 预测={predicted_digit}, 实际={digit}, 置信度={confidence:.3f}")
                        
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


def create_training_data_structure():
    """创建训练数据结构"""
    print("创建训练数据结构...")
    
    # 创建训练和测试目录
    for split in ['train', 'test']:
        for digit in range(10):
            digit_dir = f"data/{split}/digit_{digit}"
            os.makedirs(digit_dir, exist_ok=True)
    
    print("目录结构创建完成:")
    print("data/")
    print("├── train/")
    print("│   ├── digit_0/")
    print("│   ├── digit_1/")
    print("│   └── ...")
    print("└── test/")
    print("    ├── digit_0/")
    print("    ├── digit_1/")
    print("    └── ...")
    print("\n请将对应的WAV文件放入相应目录进行训练和测试")


if __name__ == "__main__":
    # 创建训练数据结构
    create_training_data_structure()
    
    # 示例：训练和测试识别器
    recognizer = SimpleDigitRecognizer()
    
    # 如果有训练数据，可以取消注释以下代码
    # recognizer.train("data/train")
    # results = recognizer.test_recognition("data/test")
