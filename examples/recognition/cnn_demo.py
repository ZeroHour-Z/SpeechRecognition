#!/usr/bin/env python3
"""
CNN分类器演示程序
展示如何使用PyTorch CNN进行语音识别
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recognition.classifiers import CNNClassifier, ClassifierComparison

def generate_synthetic_data(n_samples=1000, n_features=11):
    """生成合成数据用于测试"""
    print("生成合成测试数据...")
    
    # 生成随机特征数据
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    
    # 为不同数字添加不同的特征模式
    labels = []
    for i in range(n_samples):
        # 根据特征值生成标签，添加一些模式
        digit = int(np.sum(features[i] * np.sin(np.arange(n_features))) * 2) % 10
        labels.append(digit)
    
    labels = np.array(labels)
    
    # 添加一些噪声
    features += np.random.randn(n_samples, n_features) * 0.1
    
    return features, labels

def test_cnn_classifier():
    """测试CNN分类器"""
    print("=" * 60)
    print("CNN分类器测试")
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
    
    # 创建CNN分类器
    cnn_classifier = CNNClassifier(
        input_length=11,
        num_classes=10,
        learning_rate=0.001
    )
    
    # 训练分类器
    print("\n训练CNN...")
    start_time = time.time()
    cnn_classifier.train(train_features, train_labels)
    train_time = time.time() - start_time
    
    # 测试分类器
    print("\n测试CNN...")
    start_time = time.time()
    predictions = cnn_classifier.predict(test_features)
    predict_time = time.time() - start_time
    probabilities = cnn_classifier.predict_proba(test_features)
    
    # 计算准确率
    accuracy = np.mean(predictions == test_labels)
    print(f"CNN准确率: {accuracy:.4f}")
    print(f"训练时间: {train_time:.2f}秒")
    print(f"预测时间: {predict_time:.2f}秒")
    
    # 显示分类器信息
    info = cnn_classifier.get_info()
    print(f"\n分类器信息:")
    for key, value in info.items():
        if key != 'model_architecture':  # 跳过冗长的架构信息
            print(f"  {key}: {value}")
    
    return cnn_classifier, accuracy, train_time, predict_time

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
    print(f"{'分类器':<20} {'准确率':<10} {'平均置信度':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        classifier_name = result.get('classifier_name', name)
        accuracy = result.get('accuracy', 0.0)
        confidence = result.get('avg_confidence', 0.0)
        print(f"{classifier_name:<20} {accuracy:<10.4f} {confidence:<12.4f}")
    
    return results

def plot_performance_comparison(results):
    """绘制性能对比图"""
    if not results:
        return
    
    classifiers = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in classifiers]
    confidences = [results[name]['avg_confidence'] for name in classifiers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准确率对比
    bars1 = ax1.bar(classifiers, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray'])
    ax1.set_title('Classifier Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classifier', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 置信度对比
    bars2 = ax2.bar(classifiers, confidences, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray'])
    ax2.set_title('Classifier Confidence Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Classifier', fontsize=12)
    ax2.set_ylabel('Average Confidence', fontsize=12)
    ax2.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, conf in zip(bars2, confidences):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_cnn_performance(cnn_classifier, test_features, test_labels):
    """分析CNN性能"""
    print("\n" + "=" * 60)
    print("CNN性能分析")
    print("=" * 60)
    
    predictions = cnn_classifier.predict(test_features)
    probabilities = cnn_classifier.predict_proba(test_features)
    
    # 计算每个类别的准确率
    print("\n各类别准确率:")
    for digit in range(10):
        mask = test_labels == digit
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == test_labels[mask])
            print(f"  数字 {digit}: {class_acc:.4f} ({np.sum(mask)} 个样本)")
    
    # 分析预测置信度
    max_probs = np.max(probabilities, axis=1)
    print(f"\n预测置信度统计:")
    print(f"  平均置信度: {np.mean(max_probs):.4f}")
    print(f"  最高置信度: {np.max(max_probs):.4f}")
    print(f"  最低置信度: {np.min(max_probs):.4f}")
    print(f"  置信度标准差: {np.std(max_probs):.4f}")
    
    # 分析错误预测
    wrong_predictions = predictions != test_labels
    if np.sum(wrong_predictions) > 0:
        print(f"\n错误预测分析:")
        print(f"  错误预测数量: {np.sum(wrong_predictions)}")
        print(f"  错误预测平均置信度: {np.mean(max_probs[wrong_predictions]):.4f}")
        print(f"  正确预测平均置信度: {np.mean(max_probs[~wrong_predictions]):.4f}")

def main():
    """主函数"""
    print("CNN分类器演示程序")
    print("=" * 60)
    
    try:
        # 测试CNN分类器
        cnn_classifier, cnn_accuracy, train_time, predict_time = test_cnn_classifier()
        
        # 对比所有分类器
        results = compare_all_classifiers()
        
        # 分析CNN性能
        features, labels = generate_synthetic_data(200, 11)
        analyze_cnn_performance(cnn_classifier, features, labels)
        
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
