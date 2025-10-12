"""
消融实验模块 - 研究不同特征组合的影响
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

from ..core.wav_reader import WAVReader
from ..core.frame_window import FrameProcessor
from ..core.time_domain_analysis import TimeDomainAnalyzer
from ..core.endpoint_detection import DualThresholdEndpointDetector
from .evaluation import ExperimentEvaluator


class AblationStudy:
    """消融实验 - 测试不同特征组合的效果"""
    
    def __init__(self, train_dir: str, test_dir: str = None):
        """
        初始化消融实验
        
        参数:
            train_dir: 训练数据目录
            test_dir: 测试数据目录
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.evaluator = ExperimentEvaluator()
        
        # 特征组合定义
        self.feature_combinations = {
            'energy_only': ['energy'],
            'zcr_only': ['zcr'],
            'magnitude_only': ['magnitude'],
            'energy_zcr': ['energy', 'zcr'],
            'energy_magnitude': ['energy', 'magnitude'],
            'zcr_magnitude': ['zcr', 'magnitude'],
            'all_features': ['energy', 'zcr', 'magnitude']
        }
        
        self.results = {}
        
    def extract_features(self, file_path: str, 
                        features: List[str]) -> np.ndarray:
        """
        提取指定的特征
        
        参数:
            file_path: 音频文件路径
            features: 要提取的特征列表
            
        返回:
            特征向量
        """
        # 读取WAV文件
        reader = WAVReader()
        reader.read(file_path)
        
        # 分帧
        frame_processor = FrameProcessor()
        frames = frame_processor.frame_signal(
            reader.audio_data, 
            reader.sample_rate
        )
        
        # 端点检测
        detector = DualThresholdEndpointDetector()
        start_frame, end_frame = detector.detect_endpoints(
            reader.audio_data,
            reader.sample_rate
        )
        
        # 提取有效帧
        if start_frame < end_frame and end_frame <= len(frames):
            valid_frames = frames[start_frame:end_frame]
        else:
            valid_frames = frames
        
        # 计算时域特征
        analyzer = TimeDomainAnalyzer()
        feature_vector = []
        
        if 'energy' in features:
            energy = analyzer.short_time_energy(valid_frames)
            feature_vector.extend([
                np.mean(energy),
                np.std(energy),
                np.max(energy),
                np.min(energy)
            ])
        
        if 'zcr' in features:
            zcr = analyzer.zero_crossing_rate(valid_frames)
            feature_vector.extend([
                np.mean(zcr),
                np.std(zcr),
                np.max(zcr),
                np.min(zcr)
            ])
        
        if 'magnitude' in features:
            magnitude = analyzer.short_time_magnitude(valid_frames)
            feature_vector.extend([
                np.mean(magnitude),
                np.std(magnitude),
                np.max(magnitude),
                np.min(magnitude)
            ])
        
        return np.array(feature_vector)
    
    def train_with_features(self, features: List[str]) -> Dict:
        """
        使用指定特征训练模型
        
        参数:
            features: 特征列表
            
        返回:
            训练好的模板字典
        """
        templates = {}
        
        for digit in range(10):
            digit_dir = os.path.join(self.train_dir, f'digit_{digit}')
            if not os.path.exists(digit_dir):
                continue
            
            digit_features = []
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                file_path = os.path.join(digit_dir, wav_file)
                try:
                    feature_vector = self.extract_features(file_path, features)
                    digit_features.append(feature_vector)
                except Exception as e:
                    print(f"⚠️ 警告: 提取特征失败 {file_path}: {str(e)}")
                    continue
            
            if digit_features:
                # 计算平均特征向量作为模板
                templates[digit] = np.mean(digit_features, axis=0)
        
        return templates
    
    def test_with_features(self, templates: Dict, 
                          features: List[str]) -> Tuple[List, List]:
        """
        使用指定特征测试模型
        
        参数:
            templates: 模板字典
            features: 特征列表
            
        返回:
            (y_true, y_pred)
        """
        y_true = []
        y_pred = []
        
        # 如果没有测试集，使用训练集的一部分
        test_dir = self.test_dir if self.test_dir else self.train_dir
        
        for digit in range(10):
            digit_dir = os.path.join(test_dir, f'digit_{digit}')
            if not os.path.exists(digit_dir):
                continue
            
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            # 如果是训练集，只使用一部分进行测试
            if self.test_dir is None:
                wav_files = wav_files[:max(1, len(wav_files)//2)]
            
            for wav_file in wav_files:
                file_path = os.path.join(digit_dir, wav_file)
                try:
                    feature_vector = self.extract_features(file_path, features)
                    
                    # 使用高斯相似度分类
                    max_similarity = -np.inf
                    predicted_digit = -1
                    
                    for template_digit, template in templates.items():
                        # 计算欧氏距离
                        distance = np.linalg.norm(feature_vector - template)
                        similarity = -distance  # 距离越小，相似度越高
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            predicted_digit = template_digit
                    
                    y_true.append(digit)
                    y_pred.append(predicted_digit)
                    
                except Exception as e:
                    print(f"⚠️ 警告: 测试失败 {file_path}: {str(e)}")
                    continue
        
        return y_true, y_pred
    
    def run_ablation_study(self) -> Dict:
        """
        运行消融实验
        
        返回:
            包含所有特征组合结果的字典
        """
        print("\n" + "="*80)
        print("🔬 开始消融实验 | Ablation Study")
        print("="*80)
        
        for combination_name, features in self.feature_combinations.items():
            print(f"\n📊 测试特征组合: {combination_name}")
            print(f"   特征: {', '.join(features)}")
            print("-" * 60)
            
            try:
                # 训练
                start_time = time.time()
                templates = self.train_with_features(features)
                train_time = time.time() - start_time
                
                # 测试
                start_time = time.time()
                y_true, y_pred = self.test_with_features(templates, features)
                test_time = time.time() - start_time
                
                if len(y_true) == 0:
                    print(f"❌ 没有测试数据")
                    continue
                
                # 计算指标
                metrics = self.evaluator.calculate_metrics(y_true, y_pred)
                
                # 保存结果
                self.results[combination_name] = {
                    'features': features,
                    'metrics': metrics,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'train_time': train_time,
                    'test_time': test_time
                }
                
                # 打印结果
                print(f"✅ 训练完成，用时: {train_time:.3f}s")
                print(f"✅ 测试完成，用时: {test_time:.3f}s")
                print(f"📈 准确率: {metrics['accuracy']:.4f}")
                print(f"📈 F1分数: {metrics['f1_macro']:.4f}")
                
            except Exception as e:
                print(f"❌ 特征组合 {combination_name} 测试失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "="*80)
        print("✅ 消融实验完成！")
        print("="*80 + "\n")
        
        return self.results
    
    def visualize_results(self, output_dir: str = 'data/results/ablation'):
        """
        可视化消融实验结果
        
        参数:
            output_dir: 输出目录
        """
        if not self.results:
            print("❌ 没有实验结果可以可视化")
            return
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("📊 生成消融实验可视化结果")
        print("="*80 + "\n")
        
        # 1. 特征组合性能对比
        self._plot_feature_comparison(
            save_path=os.path.join(output_dir, 'feature_comparison.png')
        )
        
        # 2. 打印详细结果表格
        self._print_ablation_table()
        
        # 3. 特征重要性分析
        self._plot_feature_importance(
            save_path=os.path.join(output_dir, 'feature_importance.png')
        )
        
        print(f"\n✅ 所有消融实验结果已保存到: {output_dir}\n")
    
    def _plot_feature_comparison(self, save_path: str):
        """绘制特征组合性能对比图"""
        combination_names = list(self.results.keys())
        accuracies = [self.results[name]['metrics']['accuracy'] for name in combination_names]
        f1_scores = [self.results[name]['metrics']['f1_macro'] for name in combination_names]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(combination_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', 
                      color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', 
                      color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Feature Combination', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Ablation Study: Feature Combination Comparison', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(combination_names, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 特征组合对比图已保存到: {save_path}")
    
    def _plot_feature_importance(self, save_path: str):
        """绘制特征重要性分析图"""
        # 计算每个特征的重要性
        feature_importance = {
            'energy': 0,
            'zcr': 0,
            'magnitude': 0
        }
        
        # 基于包含该特征的组合的平均性能来计算重要性
        for combination_name, result in self.results.items():
            features = result['features']
            accuracy = result['metrics']['accuracy']
            
            for feature in features:
                feature_importance[feature] += accuracy / len(features)
        
        # 归一化
        total = sum(feature_importance.values())
        if total > 0:
            for key in feature_importance:
                feature_importance[key] /= total
        
        # 绘图
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(features, importance, color=colors, alpha=0.8)
        
        ax.set_xlabel('Feature', fontsize=14, fontweight='bold')
        ax.set_ylabel('Relative Importance', fontsize=14, fontweight='bold')
        ax.set_title('Feature Importance Analysis', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 特征重要性分析图已保存到: {save_path}")
    
    def _print_ablation_table(self):
        """打印消融实验结果表格"""
        print("\n" + "="*100)
        print("📊 消融实验结果表 | Ablation Study Results")
        print("="*100)
        
        # 表头
        header = f"{'Feature Combination':<30} {'Features':<25} {'Accuracy':>12} {'F1-Score':>12}"
        print(header)
        print("-"*100)
        
        # 按F1分数排序
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['metrics']['f1_macro'], 
            reverse=True
        )
        
        # 数据行
        for combination_name, result in sorted_results:
            features_str = ', '.join(result['features'])
            metrics = result['metrics']
            row = f"{combination_name:<30} {features_str:<25} {metrics['accuracy']:>12.4f} " \
                  f"{metrics['f1_macro']:>12.4f}"
            print(row)
        
        print("="*100)
        
        # 找出最佳特征组合
        best_combination = sorted_results[0]
        print(f"\n🏆 最佳特征组合 | Best Feature Combination: {best_combination[0]}")
        print(f"   特征 | Features: {', '.join(best_combination[1]['features'])}")
        print(f"   F1分数 | F1-Score: {best_combination[1]['metrics']['f1_macro']:.4f}")
        print()

