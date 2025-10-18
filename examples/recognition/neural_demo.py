#!/usr/bin/env python3
"""
神经网络分类器演示程序
展示如何使用新添加的神经网络分类器进行语音识别
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recognition.classifiers import NeuralNetworkClassifier, ClassifierComparison

def generate_synthetic_data(n_samples=1000, n_features=11):
    """生成合成数据用于测试"""
    print("生成合成测试数据...")
    
    # 生成随机特征数据
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    
    # 为不同数字添加不同的特征模式
    labels = []
    for i in range(n_samples):
        # 根据特征值生成标签
        digit = int(np.sum(features[i]) * 2) % 10
        labels.append(digit)
    
    labels = np.array(labels)
    
    # 添加一些噪声
    features += np.random.randn(n_samples, n_features) * 0.1
    
    return features, labels

def test_neural_network_classifier():
    """测试神经网络分类器"""
    print("=" * 60)
    print("神经网络分类器测试")
    print("=" * 60)
    
    # 生成测试数据
    features, labels = generate_synthetic_data(1000, 11)
    
    # 分割训练和测试数据
    split_idx = int(0.8 * len(features))
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    test_features = features[split_idx:]
    test_labels = labels[split_idx:]
    
    print(f"训练样本数: {len(train_features)}")
    print(f"测试样本数: {len(test_features)}")
    print(f"特征维度: {train_features.shape[1]}")
    
    # 创建神经网络分类器
    nn_classifier = NeuralNetworkClassifier(
        hidden_layer_sizes=(100, 50),  # 两层隐藏层
        max_iter=1000
    )
    
    # 训练分类器
    print("\n训练神经网络...")
    nn_classifier.train(train_features, train_labels)
    
    # 测试分类器
    print("\n测试神经网络...")
    predictions = nn_classifier.predict(test_features)
    probabilities = nn_classifier.predict_proba(test_features)
    
    # 计算准确率
    accuracy = np.mean(predictions == test_labels)
    print(f"神经网络准确率: {accuracy:.4f}")
    
    # 显示分类器信息
    info = nn_classifier.get_info()
    print(f"\n分类器信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return nn_classifier, accuracy

def compare_all_classifiers():
    """对比所有分类器性能"""
    print("\n" + "=" * 60)
    print("所有分类器性能对比")
    print("=" * 60)
    
    # 生成测试数据
    features, labels = generate_synthetic_data(800, 11)
    
    # 分割数据
    split_idx = int(0.8 * len(features))
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    test_features = features[split_idx:]
    test_labels = labels[split_idx:]
    
    # 创建分类器对比
    comparison = ClassifierComparison()
    
    # 训练所有分类器
    comparison.train_all(train_features, train_labels)
    
    # 评估所有分类器
    results = comparison.evaluate_all(test_features, test_labels)
    
    # 显示结果
    print("\n分类器性能对比:")
    print("-" * 80)
    print(f"{'分类器':<20} {'准确率':<10} {'训练时间':<10} {'预测时间':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        classifier_name = result.get('classifier_name', name)
        accuracy = result.get('accuracy', 0.0)
        print(f"{classifier_name:<20} {accuracy:<10.4f} {'N/A':<10} {'N/A':<10}")
    
    return results

def plot_performance_comparison(results):
    """绘制性能对比图"""
    if not results:
        return
    
    classifiers = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in classifiers]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classifiers, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray'])
    plt.title('Classifier Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Classifier', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("神经网络分类器演示程序")
    print("=" * 60)
    
    try:
        # 测试神经网络分类器
        nn_classifier, nn_accuracy = test_neural_network_classifier()
        
        # 对比所有分类器
        results = compare_all_classifiers()
        
        # 绘制性能对比图
        plot_performance_comparison(results)
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
