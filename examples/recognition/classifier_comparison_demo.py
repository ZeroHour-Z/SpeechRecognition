"""
分类器对比演示程序
演示多种分类器的性能对比和选择
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import AdvancedDigitRecognizer, create_training_data_structure
import numpy as np


def classifier_comparison_demo():
    """分类器对比演示"""
    print("=" * 80)
    print("语音识别分类器对比演示")
    print("=" * 80)
    
    # 1. 创建训练数据结构
    print("\n1. 创建训练数据结构...")
    create_training_data_structure()
    
    # 2. 检查训练数据
    train_dir = "data/train"
    test_dir = "data/test"
    
    has_training_data = False
    for digit in range(10):
        digit_dir = os.path.join(train_dir, f"digit_{digit}")
        if os.path.exists(digit_dir) and os.listdir(digit_dir):
            has_training_data = True
            break
    
    if not has_training_data:
        print("\n未找到训练数据，将进行理论演示...")
        theoretical_demo()
        return
    
    # 3. 创建高级识别器
    print("\n2. 创建高级识别器...")
    recognizer = AdvancedDigitRecognizer()
    
    # 4. 训练所有分类器
    print("\n3. 训练所有分类器...")
    try:
        training_result = recognizer.train_with_classifiers(train_dir)
        print(f"训练完成，共训练 {len(training_result['trained_classifiers'])} 个分类器")
    except Exception as e:
        print(f"训练失败: {e}")
        return
    
    # 5. 评估分类器性能
    print("\n4. 评估分类器性能...")
    try:
        evaluation_result = recognizer.evaluate_classifiers(test_dir)
        print("评估完成")
    except Exception as e:
        print(f"评估失败: {e}")
        return
    
    # 6. 生成详细对比报告
    print("\n5. 生成详细对比报告...")
    detailed_report = recognizer.compare_classifier_performance()
    print(detailed_report)
    
    # 7. 选择最佳分类器
    print("\n6. 选择最佳分类器...")
    try:
        best_classifier = recognizer.select_best_classifier()
        print(f"最佳分类器: {best_classifier}")
    except Exception as e:
        print(f"选择最佳分类器失败: {e}")
    
    # 8. 交互式识别测试
    print("\n7. 交互式识别测试...")
    interactive_recognition_test(recognizer)


def theoretical_demo():
    """理论演示 - 当没有训练数据时"""
    print("\n分类器理论对比分析")
    print("=" * 60)
    
    # 显示分类器选择指南
    from src.recognition.advanced_recognizer import create_classifier_selection_guide
    guide = create_classifier_selection_guide()
    print(guide)
    
    # 模拟性能对比
    print("\n模拟性能对比结果:")
    print("-" * 60)
    print(f"{'分类器':<20} {'准确率':<10} {'训练时间':<10} {'内存占用'}")
    print("-" * 60)
    print(f"{'朴素贝叶斯':<20} {'0.85':<10} {'0.1s':<10} {'低'}")
    print(f"{'Fisher线性判别':<20} {'0.82':<10} {'0.2s':<10} {'低'}")
    print(f"{'决策树':<20} {'0.78':<10} {'0.3s':<10} {'中'}")
    print(f"{'支持向量机':<20} {'0.92':<10} {'2.0s':<10} {'中'}")
    print(f"{'K近邻':<20} {'0.88':<10} {'0.1s':<10} {'高'}")
    
    print("\n推荐选择: 支持向量机")
    print("理由: 准确率最高，适合语音识别任务")
    
    print("\n要进行实际训练和测试，请:")
    print("1. 准备训练数据: data/train/digit_X/")
    print("2. 准备测试数据: data/test/digit_X/")
    print("3. 重新运行此程序")


def interactive_recognition_test(recognizer):
    """交互式识别测试"""
    print("\n交互式识别测试")
    print("=" * 40)
    
    # 列出可用的分类器
    available_classifiers = list(recognizer.classifier_comparison.classifiers.keys())
    print("可用的分类器:")
    for i, name in enumerate(available_classifiers):
        classifier = recognizer.classifier_comparison.classifiers[name]
        print(f"{i+1}. {classifier.name}")
    
    # 选择分类器
    try:
        choice = int(input("\n请选择要使用的分类器编号: ")) - 1
        if 0 <= choice < len(available_classifiers):
            selected_classifier = available_classifiers[choice]
            classifier_name = recognizer.classifier_comparison.classifiers[selected_classifier].name
            print(f"已选择: {classifier_name}")
        else:
            print("无效选择，使用最佳分类器")
            selected_classifier = recognizer.selected_classifier
    except ValueError:
        print("无效输入，使用最佳分类器")
        selected_classifier = recognizer.selected_classifier
    
    # 选择测试文件
    audio_dir = "data/audio"
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')] if os.path.exists(audio_dir) else []
    
    if not wav_files:
        print("没有找到测试音频文件")
        return
    
    print(f"\n可用的测试文件:")
    for i, wav_file in enumerate(wav_files):
        print(f"{i+1}. {wav_file}")
    
    try:
        file_choice = int(input("\n请选择测试文件编号: ")) - 1
        if 0 <= file_choice < len(wav_files):
            wav_file = wav_files[file_choice]
            wav_path = os.path.join(audio_dir, wav_file)
            
            print(f"\n正在使用 {classifier_name} 识别: {wav_file}")
            print("-" * 50)
            
            # 读取音频
            from src import WAVReader
            reader = WAVReader(wav_path)
            audio_data, sample_rate = reader.read()
            
            # 识别
            predicted_digit, confidence = recognizer.recognize_with_classifier(
                audio_data, selected_classifier
            )
            
            print(f"识别结果: {predicted_digit}")
            print(f"置信度: {confidence:.3f}")
            
            # 置信度评估
            if confidence > 0.8:
                print("识别结果: 高置信度")
            elif confidence > 0.6:
                print("识别结果: 中等置信度")
            else:
                print("识别结果: 低置信度，可能不准确")
            
            # 显示所有分类器的预测结果
            print(f"\n所有分类器的预测结果:")
            print("-" * 50)
            for name in available_classifiers:
                try:
                    pred, conf = recognizer.recognize_with_classifier(audio_data, name)
                    classifier = recognizer.classifier_comparison.classifiers[name]
                    print(f"{classifier.name}: {pred} (置信度: {conf:.3f})")
                except Exception as e:
                    print(f"{classifier.name}: 识别失败 - {e}")
                    
        else:
            print("无效选择")
            
    except ValueError:
        print("无效输入")


def batch_test_all_classifiers():
    """批量测试所有分类器"""
    print("\n批量测试所有分类器")
    print("=" * 50)
    
    recognizer = AdvancedDigitRecognizer()
    
    # 检查是否有训练数据
    train_dir = "data/train"
    test_dir = "data/test"
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("没有找到训练或测试数据")
        return
    
    try:
        # 训练
        print("训练所有分类器...")
        recognizer.train_with_classifiers(train_dir)
        
        # 评估
        print("评估所有分类器...")
        recognizer.evaluate_classifiers(test_dir)
        
        # 生成报告
        report = recognizer.compare_classifier_performance()
        print(report)
        
        # 保存报告
        report_file = "data/results/classifier_comparison_report.txt"
        os.makedirs("data/results", exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"批量测试失败: {e}")


if __name__ == "__main__":
    classifier_comparison_demo()
