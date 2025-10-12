"""
评价指标计算模块
"""

import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)


class ExperimentEvaluator:
    """实验评估器 - 计算各种评价指标"""
    
    def __init__(self):
        """初始化评估器"""
        self.metrics = {}
        
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        计算所有评价指标
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            
        返回:
            包含所有指标的字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        self.metrics = metrics
        return metrics
    
    def get_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> np.ndarray:
        """
        计算混淆矩阵
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            
        返回:
            混淆矩阵
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true: List[int], y_pred: List[int], 
                                 target_names: List[str] = None) -> str:
        """
        生成分类报告
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 类别名称列表
            
        返回:
            分类报告字符串
        """
        if target_names is None:
            target_names = [f'Digit {i}' for i in range(10)]
            
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                             save_path: str = None, show: bool = True) -> None:
        """
        绘制混淆矩阵热图
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
            show: 是否显示图表
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10),
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 混淆矩阵已保存到: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]], 
                               save_path: str = None, show: bool = True) -> None:
        """
        绘制多个模型的指标对比图
        
        参数:
            metrics_dict: 字典，键为模型名称，值为指标字典
            save_path: 保存路径
            show: 是否显示图表
        """
        # 提取数据
        model_names = list(metrics_dict.keys())
        metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        metric_display_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # 准备数据
        data = []
        for metric in metric_names:
            data.append([metrics_dict[model][metric] for model in model_names])
        
        # 绘图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(model_names))
        width = 0.2
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (metric_data, metric_name, color) in enumerate(zip(data, metric_display_names, colors)):
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, metric_data, width, label=metric_name, color=color, alpha=0.8)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Classifier', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Classifier Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='lower right', fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 指标对比图已保存到: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def print_metrics_table(self, metrics_dict: Dict[str, Dict[str, float]]) -> None:
        """
        打印指标对比表格
        
        参数:
            metrics_dict: 字典，键为模型名称，值为指标字典
        """
        print("\n" + "="*100)
        print("📊 模型性能指标对比表 | Model Performance Metrics Comparison")
        print("="*100)
        
        # 表头
        header = f"{'Model Name':<25} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}"
        print(header)
        print("-"*100)
        
        # 数据行
        for model_name, metrics in metrics_dict.items():
            row = f"{model_name:<25} {metrics['accuracy']:>12.4f} {metrics['precision_macro']:>12.4f} " \
                  f"{metrics['recall_macro']:>12.4f} {metrics['f1_macro']:>12.4f}"
            print(row)
        
        print("="*100)
        
        # 找出最佳模型
        best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1_macro'])
        print(f"\n🏆 最佳模型 | Best Model: {best_model[0]} (F1-Score: {best_model[1]['f1_macro']:.4f})")
        print()

