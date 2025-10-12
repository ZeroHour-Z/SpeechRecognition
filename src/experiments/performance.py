"""
性能测试模块 - 测试系统的运行效率和资源占用
"""

import os
import time
import psutil
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path

from ..recognition.advanced_recognizer import AdvancedDigitRecognizer


class PerformanceTest:
    """性能测试类"""
    
    def __init__(self, train_dir: str, test_dir: str = None):
        """
        初始化性能测试
        
        参数:
            train_dir: 训练数据目录
            test_dir: 测试数据目录
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.results = {}
        
    def test_classifier_performance(self, classifier_type: str = 'template') -> Dict:
        """
        测试分类器性能
        
        参数:
            classifier_type: 分类器类型
            
        返回:
            性能指标字典
        """
        print(f"\n📊 测试分类器性能: {classifier_type.upper()}")
        print("-" * 60)
        
        recognizer = AdvancedDigitRecognizer(classifier_type=classifier_type)
        
        # 1. 测试训练时间和内存占用
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        recognizer.train(self.train_dir)
        train_time = time.time() - start_time
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_usage = mem_after - mem_before
        
        print(f"✅ 训练完成")
        print(f"   训练时间: {train_time:.3f}s")
        print(f"   内存占用: {mem_usage:.2f} MB")
        
        # 2. 测试预测时间
        test_dir = self.test_dir if self.test_dir else self.train_dir
        test_files = []
        
        for digit in range(10):
            digit_dir = os.path.join(test_dir, f'digit_{digit}')
            if os.path.exists(digit_dir):
                files = [os.path.join(digit_dir, f) 
                        for f in os.listdir(digit_dir) 
                        if f.endswith('.wav')]
                test_files.extend(files[:2])  # 每个数字取2个样本
        
        prediction_times = []
        
        for file_path in test_files:
            start_time = time.time()
            _ = recognizer.recognize(file_path)
            pred_time = time.time() - start_time
            prediction_times.append(pred_time)
        
        avg_pred_time = np.mean(prediction_times)
        std_pred_time = np.std(prediction_times)
        
        print(f"✅ 预测测试完成")
        print(f"   平均预测时间: {avg_pred_time*1000:.2f} ms")
        print(f"   预测时间标准差: {std_pred_time*1000:.2f} ms")
        
        # 3. 计算吞吐量
        throughput = 1.0 / avg_pred_time if avg_pred_time > 0 else 0
        
        print(f"✅ 吞吐量: {throughput:.2f} samples/second")
        
        results = {
            'classifier': classifier_type,
            'train_time': train_time,
            'memory_usage_mb': mem_usage,
            'avg_prediction_time_ms': avg_pred_time * 1000,
            'std_prediction_time_ms': std_pred_time * 1000,
            'throughput': throughput,
            'num_test_samples': len(test_files)
        }
        
        return results
    
    def run_performance_tests(self, classifiers: List[str] = None) -> Dict:
        """
        运行多个分类器的性能测试
        
        参数:
            classifiers: 要测试的分类器列表
            
        返回:
            所有分类器的性能结果
        """
        if classifiers is None:
            classifiers = ['template', 'naive_bayes', 'fisher', 
                          'decision_tree', 'svm', 'knn']
        
        print("\n" + "="*80)
        print("🔬 开始性能测试 | Performance Testing")
        print("="*80)
        
        for clf in classifiers:
            try:
                result = self.test_classifier_performance(clf)
                self.results[clf] = result
            except Exception as e:
                print(f"❌ 分类器 {clf} 性能测试失败: {str(e)}")
                continue
        
        print("\n" + "="*80)
        print("✅ 性能测试完成！")
        print("="*80 + "\n")
        
        return self.results
    
    def visualize_results(self, output_dir: str = 'data/results/performance'):
        """
        可视化性能测试结果
        
        参数:
            output_dir: 输出目录
        """
        if not self.results:
            print("❌ 没有性能测试结果可以可视化")
            return
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("📊 生成性能测试可视化结果")
        print("="*80 + "\n")
        
        # 1. 训练时间对比
        self._plot_training_time(
            save_path=os.path.join(output_dir, 'training_time.png')
        )
        
        # 2. 预测时间对比
        self._plot_prediction_time(
            save_path=os.path.join(output_dir, 'prediction_time.png')
        )
        
        # 3. 内存占用对比
        self._plot_memory_usage(
            save_path=os.path.join(output_dir, 'memory_usage.png')
        )
        
        # 4. 综合性能对比
        self._plot_comprehensive_performance(
            save_path=os.path.join(output_dir, 'comprehensive_performance.png')
        )
        
        # 5. 打印性能表格
        self._print_performance_table()
        
        print(f"\n✅ 所有性能测试结果已保存到: {output_dir}\n")
    
    def _plot_training_time(self, save_path: str):
        """绘制训练时间对比图"""
        classifiers = list(self.results.keys())
        train_times = [self.results[clf]['train_time'] for clf in classifiers]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classifiers)))
        bars = ax.bar(classifiers, train_times, color=colors, alpha=0.8)
        
        ax.set_xlabel('Classifier', fontsize=14, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('Training Time Comparison', fontsize=16, fontweight='bold')
        ax.set_xticklabels(classifiers, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}s',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练时间对比图已保存到: {save_path}")
    
    def _plot_prediction_time(self, save_path: str):
        """绘制预测时间对比图"""
        classifiers = list(self.results.keys())
        pred_times = [self.results[clf]['avg_prediction_time_ms'] for clf in classifiers]
        std_times = [self.results[clf]['std_prediction_time_ms'] for clf in classifiers]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(classifiers)))
        bars = ax.bar(classifiers, pred_times, yerr=std_times, 
                     color=colors, alpha=0.8, capsize=5)
        
        ax.set_xlabel('Classifier', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Prediction Time (milliseconds)', fontsize=14, fontweight='bold')
        ax.set_title('Prediction Time Comparison (with standard deviation)', 
                    fontsize=16, fontweight='bold')
        ax.set_xticklabels(classifiers, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 预测时间对比图已保存到: {save_path}")
    
    def _plot_memory_usage(self, save_path: str):
        """绘制内存占用对比图"""
        classifiers = list(self.results.keys())
        mem_usage = [self.results[clf]['memory_usage_mb'] for clf in classifiers]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(classifiers)))
        bars = ax.bar(classifiers, mem_usage, color=colors, alpha=0.8)
        
        ax.set_xlabel('Classifier', fontsize=14, fontweight='bold')
        ax.set_ylabel('Memory Usage (MB)', fontsize=14, fontweight='bold')
        ax.set_title('Memory Usage Comparison', fontsize=16, fontweight='bold')
        ax.set_xticklabels(classifiers, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}MB',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 内存占用对比图已保存到: {save_path}")
    
    def _plot_comprehensive_performance(self, save_path: str):
        """绘制综合性能对比图"""
        classifiers = list(self.results.keys())
        
        # 归一化数据用于雷达图
        train_times = np.array([self.results[clf]['train_time'] for clf in classifiers])
        pred_times = np.array([self.results[clf]['avg_prediction_time_ms'] for clf in classifiers])
        mem_usage = np.array([self.results[clf]['memory_usage_mb'] for clf in classifiers])
        throughput = np.array([self.results[clf]['throughput'] for clf in classifiers])
        
        # 归一化到0-1范围（越小越好的指标需要反转）
        def normalize(arr, reverse=False):
            if arr.max() == arr.min():
                return np.ones_like(arr) * 0.5
            normalized = (arr - arr.min()) / (arr.max() - arr.min())
            return 1 - normalized if reverse else normalized
        
        train_norm = normalize(train_times, reverse=True)  # 训练时间越短越好
        pred_norm = normalize(pred_times, reverse=True)    # 预测时间越短越好
        mem_norm = normalize(mem_usage, reverse=True)      # 内存占用越小越好
        throughput_norm = normalize(throughput)             # 吞吐量越大越好
        
        # 创建子图
        fig = plt.figure(figsize=(16, 6))
        
        # 左图：堆叠柱状图
        ax1 = plt.subplot(1, 2, 1)
        
        width = 0.6
        x = np.arange(len(classifiers))
        
        p1 = ax1.bar(x, train_norm, width, label='Training Speed', color='#3498db', alpha=0.8)
        p2 = ax1.bar(x, pred_norm, width, bottom=train_norm, 
                    label='Prediction Speed', color='#e74c3c', alpha=0.8)
        p3 = ax1.bar(x, mem_norm, width, bottom=train_norm+pred_norm, 
                    label='Memory Efficiency', color='#2ecc71', alpha=0.8)
        p4 = ax1.bar(x, throughput_norm, width, bottom=train_norm+pred_norm+mem_norm, 
                    label='Throughput', color='#f39c12', alpha=0.8)
        
        ax1.set_xlabel('Classifier', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
        ax1.set_title('Comprehensive Performance Score\n(higher is better)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classifiers, rotation=45, ha='right')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 右图：性能雷达图（只显示前3个分类器，避免过于拥挤）
        ax2 = plt.subplot(1, 2, 2, projection='polar')
        
        categories = ['Training\nSpeed', 'Prediction\nSpeed', 'Memory\nEfficiency', 'Throughput']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        colors_radar = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, clf in enumerate(classifiers[:3]):  # 只显示前3个
            values = [train_norm[i], pred_norm[i], mem_norm[i], throughput_norm[i]]
            values += values[:1]  # 闭合图形
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=clf, 
                    color=colors_radar[i % len(colors_radar)], alpha=0.7)
            ax2.fill(angles, values, alpha=0.15, color=colors_radar[i % len(colors_radar)])
        
        ax2.set_title('Performance Radar Chart\n(Top 3 Classifiers)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 综合性能对比图已保存到: {save_path}")
    
    def _print_performance_table(self):
        """打印性能测试结果表格"""
        print("\n" + "="*120)
        print("📊 性能测试结果表 | Performance Test Results")
        print("="*120)
        
        # 表头
        header = f"{'Classifier':<15} {'Train Time':>12} {'Pred Time':>12} " \
                 f"{'Memory':>12} {'Throughput':>15} {'Samples':>10}"
        print(header)
        print("-"*120)
        
        # 数据行
        for clf, result in self.results.items():
            row = f"{clf:<15} {result['train_time']:>11.3f}s " \
                  f"{result['avg_prediction_time_ms']:>10.2f}ms " \
                  f"{result['memory_usage_mb']:>10.2f}MB " \
                  f"{result['throughput']:>12.2f} s/s " \
                  f"{result['num_test_samples']:>10d}"
            print(row)
        
        print("="*120)
        
        # 找出最快的分类器
        fastest_train = min(self.results.items(), key=lambda x: x[1]['train_time'])
        fastest_pred = min(self.results.items(), key=lambda x: x[1]['avg_prediction_time_ms'])
        most_efficient = min(self.results.items(), key=lambda x: x[1]['memory_usage_mb'])
        
        print(f"\n🏆 最快训练 | Fastest Training: {fastest_train[0]} ({fastest_train[1]['train_time']:.3f}s)")
        print(f"🏆 最快预测 | Fastest Prediction: {fastest_pred[0]} ({fastest_pred[1]['avg_prediction_time_ms']:.2f}ms)")
        print(f"🏆 最省内存 | Most Memory Efficient: {most_efficient[0]} ({most_efficient[1]['memory_usage_mb']:.2f}MB)")
        print()

