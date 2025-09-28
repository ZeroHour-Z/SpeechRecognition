"""
高级语音识别器
支持多种分类器选择和对比分析
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
from .simple_recognizer import SimpleDigitRecognizer
from .classifiers import (
    ClassifierComparison, 
    NaiveBayesianClassifier,
    FisherLinearDiscriminantClassifier,
    DecisionTreeClassifier,
    SupportVectorMachineClassifier,
    KNearestNeighborsClassifier
)


class AdvancedDigitRecognizer(SimpleDigitRecognizer):
    """高级数字识别器，支持多种分类器"""
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__(sample_rate)
        self.classifier_comparison = ClassifierComparison()
        self.selected_classifier = None
        self.training_features = None
        self.training_labels = None
        
    def train_with_classifiers(self, training_data_dir: str, 
                             selected_classifiers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        使用多种分类器训练识别器
        
        Args:
            training_data_dir: 训练数据目录
            selected_classifiers: 选择的分类器列表，None表示使用所有分类器
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        print("开始高级语音识别器训练...")
        print("=" * 80)
        
        # 收集训练数据
        features_list = []
        labels_list = []
        
        for digit in range(10):
            digit_dir = os.path.join(training_data_dir, f"digit_{digit}")
            if not os.path.exists(digit_dir):
                print(f"警告: 未找到数字 {digit} 的训练数据目录: {digit_dir}")
                continue
            
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            if not wav_files:
                print(f"警告: 数字 {digit} 目录下没有WAV文件")
                continue
            
            digit_features = []
            for wav_file in wav_files:
                wav_path = os.path.join(digit_dir, wav_file)
                try:
                    from ..core.wav_reader import WAVReader
                    reader = WAVReader(wav_path)
                    audio_data, _ = reader.read()
                    
                    features = self.extract_features(audio_data)
                    if features:
                        digit_features.append(features)
                        print(f"  处理: {wav_file} - 特征提取成功")
                    else:
                        print(f"  跳过: {wav_file} - 特征提取失败")
                        
                except Exception as e:
                    print(f"  错误: {wav_file} - {e}")
            
            if digit_features:
                # 转换为numpy数组
                digit_features_array = np.array(digit_features)
                features_list.append(digit_features_array)
                labels_list.extend([digit] * len(digit_features))
                print(f"数字 {digit}: 训练样本 {len(digit_features)} 个")
            else:
                print(f"数字 {digit}: 没有有效的训练样本")
        
        if not features_list:
            raise ValueError("没有找到有效的训练数据")
        
        # 合并所有特征
        self.training_features = np.vstack(features_list)
        self.training_labels = np.array(labels_list)
        
        print(f"\n训练数据统计:")
        print(f"总样本数: {len(self.training_labels)}")
        print(f"特征维度: {self.training_features.shape[1]}")
        for digit in range(10):
            count = np.sum(self.training_labels == digit)
            print(f"数字 {digit}: {count} 个样本")
        
        # 选择要使用的分类器
        if selected_classifiers is None:
            selected_classifiers = list(self.classifier_comparison.classifiers.keys())
        
        # 过滤分类器
        filtered_classifiers = {
            name: classifier for name, classifier in self.classifier_comparison.classifiers.items()
            if name in selected_classifiers
        }
        self.classifier_comparison.classifiers = filtered_classifiers
        
        # 训练所有选定的分类器
        self.classifier_comparison.train_all(self.training_features, self.training_labels)
        
        # 返回训练结果
        training_result = {
            'total_samples': len(self.training_labels),
            'feature_dimension': self.training_features.shape[1],
            'class_distribution': {f'digit_{i}': int(np.sum(self.training_labels == i)) 
                                 for i in range(10)},
            'trained_classifiers': list(filtered_classifiers.keys())
        }
        
        print("=" * 80)
        print("训练完成！")
        return training_result
    
    def evaluate_classifiers(self, test_data_dir: str) -> Dict[str, Any]:
        """
        评估所有分类器的性能
        
        Args:
            test_data_dir: 测试数据目录
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        print("开始分类器性能评估...")
        print("=" * 80)
        
        # 收集测试数据
        test_features_list = []
        test_labels_list = []
        
        for digit in range(10):
            digit_dir = os.path.join(test_data_dir, f"digit_{digit}")
            if not os.path.exists(digit_dir):
                continue
            
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            if not wav_files:
                continue
            
            digit_features = []
            for wav_file in wav_files:
                wav_path = os.path.join(digit_dir, wav_file)
                try:
                    from ..core.wav_reader import WAVReader
                    reader = WAVReader(wav_path)
                    audio_data, _ = reader.read()
                    
                    features = self.extract_features(audio_data)
                    if features:
                        digit_features.append(features)
                        
                except Exception as e:
                    print(f"  错误: {wav_file} - {e}")
            
            if digit_features:
                digit_features_array = np.array(digit_features)
                test_features_list.append(digit_features_array)
                test_labels_list.extend([digit] * len(digit_features))
        
        if not test_features_list:
            raise ValueError("没有找到有效的测试数据")
        
        # 合并测试特征
        test_features = np.vstack(test_features_list)
        test_labels = np.array(test_labels_list)
        
        print(f"测试数据统计:")
        print(f"总样本数: {len(test_labels)}")
        for digit in range(10):
            count = np.sum(test_labels == digit)
            if count > 0:
                print(f"数字 {digit}: {count} 个样本")
        
        # 评估所有分类器
        evaluation_results = self.classifier_comparison.evaluate_all(test_features, test_labels)
        
        # 生成对比报告
        comparison_report = self.classifier_comparison.generate_comparison_report()
        recommendation = self.classifier_comparison.get_recommendation()
        
        print("\n" + comparison_report)
        print("\n" + recommendation)
        
        return {
            'evaluation_results': evaluation_results,
            'comparison_report': comparison_report,
            'recommendation': recommendation,
            'test_samples': len(test_labels)
        }
    
    def select_best_classifier(self) -> str:
        """
        选择最佳分类器
        
        Returns:
            str: 最佳分类器名称
        """
        if not self.classifier_comparison.results:
            raise ValueError("没有评估结果，请先运行evaluate_classifiers")
        
        # 找到准确率最高的分类器
        best_classifier = None
        best_accuracy = -1
        
        for name, result in self.classifier_comparison.results.items():
            if 'accuracy' in result and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_classifier = name
        
        if best_classifier is None:
            raise ValueError("没有找到有效的分类器")
        
        self.selected_classifier = best_classifier
        print(f"选择最佳分类器: {self.classifier_comparison.classifiers[best_classifier].name}")
        print(f"准确率: {best_accuracy:.3f}")
        
        return best_classifier
    
    def recognize_with_classifier(self, signal: np.ndarray, 
                                classifier_name: Optional[str] = None) -> Tuple[int, float]:
        """
        使用指定分类器进行识别
        
        Args:
            signal: 输入信号
            classifier_name: 分类器名称，None表示使用最佳分类器
            
        Returns:
            Tuple[int, float]: (识别结果, 置信度)
        """
        if classifier_name is None:
            if self.selected_classifier is None:
                self.select_best_classifier()
            classifier_name = self.selected_classifier
        
        if classifier_name not in self.classifier_comparison.classifiers:
            raise ValueError(f"分类器 {classifier_name} 不存在")
        
        classifier = self.classifier_comparison.classifiers[classifier_name]
        
        if not classifier.is_trained:
            raise ValueError(f"分类器 {classifier_name} 尚未训练")
        
        # 提取特征
        features = self.extract_features(signal)
        if not features:
            return -1, 0.0
        
        # 转换为numpy数组
        features_array = np.array([list(features.values())])
        
        # 预测
        prediction = classifier.predict(features_array)[0]
        probabilities = classifier.predict_proba(features_array)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def compare_classifier_performance(self) -> str:
        """
        生成详细的分类器性能对比报告
        
        Returns:
            str: 对比报告
        """
        if not self.classifier_comparison.results:
            return "没有评估结果"
        
        report = []
        report.append("详细分类器性能对比分析")
        report.append("=" * 100)
        
        # 总体性能对比
        report.append("\n1. 总体性能对比")
        report.append("-" * 60)
        report.append(f"{'分类器':<20} {'准确率':<10} {'置信度':<10} {'状态':<10}")
        report.append("-" * 60)
        
        sorted_results = sorted(
            [(name, result) for name, result in self.classifier_comparison.results.items() 
             if 'accuracy' in result],
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        for name, result in sorted_results:
            report.append(f"{result['classifier_name']:<20} "
                        f"{result['accuracy']:<10.3f} "
                        f"{result['avg_confidence']:<10.3f} "
                        f"{'正常':<10}")
        
        # 各数字识别准确率对比
        report.append("\n2. 各数字识别准确率对比")
        report.append("-" * 60)
        report.append(f"{'数字':<6}", end="")
        for name, result in sorted_results:
            report.append(f"{result['classifier_name']:<15}", end="")
        report.append("")
        report.append("-" * 60)
        
        for digit in range(10):
            report.append(f"{digit:<6}", end="")
            for name, result in sorted_results:
                digit_acc = result['class_accuracies'].get(f'digit_{digit}', 0)
                report.append(f"{digit_acc:<15.3f}", end="")
            report.append("")
        
        # 分类器特点分析
        report.append("\n3. 分类器特点分析")
        report.append("-" * 60)
        
        classifier_analysis = {
            'naive_bayes': {
                'name': '朴素贝叶斯',
                'advantages': ['简单快速', '适合小样本', '对噪声鲁棒'],
                'disadvantages': ['假设特征独立', '可能欠拟合'],
                'best_for': '快速原型和小数据集'
            },
            'fisher_lda': {
                'name': 'Fisher线性判别',
                'advantages': ['计算效率高', '降维效果好', '线性分类'],
                'disadvantages': ['假设高斯分布', '线性边界限制'],
                'best_for': '线性可分的数据'
            },
            'decision_tree': {
                'name': '决策树',
                'advantages': ['可解释性强', '处理非线性', '无需特征缩放'],
                'disadvantages': ['容易过拟合', '对噪声敏感'],
                'best_for': '需要解释性的场景'
            },
            'svm': {
                'name': '支持向量机',
                'advantages': ['泛化能力强', '适合高维数据', '内存效率高'],
                'disadvantages': ['参数敏感', '大数据集慢'],
                'best_for': '高维特征和复杂边界'
            },
            'knn': {
                'name': 'K近邻',
                'advantages': ['非参数方法', '适合复杂边界', '简单直观'],
                'disadvantages': ['计算复杂度高', '对维度敏感'],
                'best_for': '小数据集和复杂模式'
            }
        }
        
        for name, result in sorted_results:
            if name in classifier_analysis:
                analysis = classifier_analysis[name]
                report.append(f"\n{analysis['name']}:")
                report.append(f"  优点: {', '.join(analysis['advantages'])}")
                report.append(f"  缺点: {', '.join(analysis['disadvantages'])}")
                report.append(f"  适用场景: {analysis['best_for']}")
        
        # 选择建议
        report.append("\n4. 分类器选择建议")
        report.append("-" * 60)
        
        if sorted_results:
            best_result = sorted_results[0][1]
            best_accuracy = best_result['accuracy']
            
            if best_accuracy > 0.9:
                report.append(f"✓ 强烈推荐: {best_result['classifier_name']}")
                report.append(f"  理由: 准确率优秀 ({best_accuracy:.3f})")
            elif best_accuracy > 0.8:
                report.append(f"✓ 推荐使用: {best_result['classifier_name']}")
                report.append(f"  理由: 准确率良好 ({best_accuracy:.3f})")
            elif best_accuracy > 0.7:
                report.append(f"⚠ 可考虑: {best_result['classifier_name']}")
                report.append(f"  理由: 准确率一般 ({best_accuracy:.3f})")
                report.append("  建议: 改进特征提取或增加训练数据")
            else:
                report.append("⚠ 所有分类器性能不佳，建议:")
                report.append("  1. 增加训练样本数量")
                report.append("  2. 改进特征提取方法")
                report.append("  3. 检查数据质量和标注")
                report.append("  4. 尝试特征工程和预处理")
        
        return "\n".join(report)


def create_classifier_selection_guide() -> str:
    """创建分类器选择指南"""
    guide = """
分类器选择指南
==============

1. 朴素贝叶斯 (Naive Bayesian)
   - 适用场景: 小数据集、快速原型、特征独立假设成立
   - 优点: 训练快速、对噪声鲁棒、内存占用小
   - 缺点: 假设特征独立、可能欠拟合
   - 推荐指数: ⭐⭐⭐⭐

2. Fisher线性判别 (Fisher Linear Discriminant)
   - 适用场景: 线性可分数据、需要降维、计算资源有限
   - 优点: 计算效率高、降维效果好、理论成熟
   - 缺点: 假设高斯分布、只能处理线性边界
   - 推荐指数: ⭐⭐⭐

3. 决策树 (Decision Tree)
   - 适用场景: 需要可解释性、处理非线性关系
   - 优点: 可解释性强、处理非线性、无需特征缩放
   - 缺点: 容易过拟合、对噪声敏感、不稳定
   - 推荐指数: ⭐⭐⭐

4. 支持向量机 (Support Vector Machine)
   - 适用场景: 高维特征、复杂边界、需要强泛化能力
   - 优点: 泛化能力强、适合高维数据、内存效率高
   - 缺点: 参数敏感、大数据集训练慢、黑盒模型
   - 推荐指数: ⭐⭐⭐⭐⭐

5. K近邻 (K-Nearest Neighbors)
   - 适用场景: 小数据集、复杂模式、非参数方法
   - 优点: 非参数方法、适合复杂边界、简单直观
   - 缺点: 计算复杂度高、对维度敏感、需要大量内存
   - 推荐指数: ⭐⭐⭐

选择建议:
- 如果追求最高准确率: 支持向量机
- 如果需要快速训练: 朴素贝叶斯
- 如果需要可解释性: 决策树
- 如果数据线性可分: Fisher线性判别
- 如果数据量小且模式复杂: K近邻
"""
    return guide
