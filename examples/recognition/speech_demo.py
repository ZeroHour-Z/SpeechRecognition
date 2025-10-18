"""
语音识别演示程序
演示如何使用时域特征进行简单的数字识别
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import WAVReader, SimpleDigitRecognizer, create_training_data_structure
import numpy as np


def speech_recognition_demo():
    """语音识别演示"""
    print("=" * 80)
    print("语音识别演示程序")
    print("=" * 80)
    
    # 1. 创建训练数据结构
    print("\n1. 创建训练数据结构...")
    create_training_data_structure()
    
    # 2. 检查是否有训练数据
    train_dir = "data/train"
    test_dir = "data/test"
    
    has_training_data = False
    for digit in range(10):
        digit_dir = os.path.join(train_dir, f"digit_{digit}")
        if os.path.exists(digit_dir) and os.listdir(digit_dir):
            has_training_data = True
            break
    
    if not has_training_data:
        print("\n未找到训练数据，将使用现有音频文件进行演示...")
        demo_with_existing_files()
        return
    
    # 3. 训练识别器
    print("\n2. 训练识别器...")
    recognizer = SimpleDigitRecognizer()
    recognizer.train(train_dir)
    
    # 4. 测试识别器
    print("\n3. 测试识别器...")
    results = recognizer.test_recognition(test_dir)
    
    # 5. 交互式识别
    print("\n4. 交互式识别...")
    interactive_recognition(recognizer)


def demo_with_existing_files():
    """使用现有文件进行演示"""
    print("\n使用现有音频文件进行特征提取演示...")
    
    # 检查data/audio/samples目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    audio_dir = os.path.join(project_root, "data", "audio", "samples")
    if not os.path.exists(audio_dir):
        print(f"音频目录 {audio_dir} 不存在")
        return
    
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not wav_files:
        print("没有找到WAV文件")
        return
    
    # 创建识别器
    recognizer = SimpleDigitRecognizer()
    
    print(f"\n找到 {len(wav_files)} 个WAV文件，进行特征提取演示:")
    print("-" * 60)
    
    for wav_file in wav_files:
        wav_path = os.path.join(audio_dir, wav_file)
        try:
            reader = WAVReader(wav_path)
            audio_data, sample_rate = reader.read()
            
            # 提取特征
            features = recognizer.extract_features(audio_data)
            
            if features:
                print(f"\n文件: {wav_file}")
                print(f"  时长: {features['duration']:.3f} 秒")
                print(f"  最大能量: {features['max_energy']:.6f}")
                print(f"  平均能量: {features['mean_energy']:.6f}")
                print(f"  最大幅度: {features['max_amplitude']:.3f}")
                print(f"  平均幅度: {features['mean_amplitude']:.3f}")
                print(f"  最大过零率: {features['max_zcr']:.4f}")
                print(f"  平均过零率: {features['mean_zcr']:.4f}")
                print(f"  能量比: {features['energy_ratio']:.3f}")
            else:
                print(f"\n文件: {wav_file} - 特征提取失败")
                
        except Exception as e:
            print(f"\n文件: {wav_file} - 错误: {e}")
    
    print("\n" + "=" * 60)
    print("特征提取演示完成")
    print("\n要进行完整的语音识别，请:")
    print("1. 将训练数据按数字分类放入 data/train/digit_X/ 目录")
    print("2. 将测试数据按数字分类放入 data/test/digit_X/ 目录")
    print("3. 重新运行此程序")


def interactive_recognition(recognizer):
    """交互式识别"""
    print("\n交互式语音识别")
    print("=" * 40)
    print("请选择要识别的音频文件:")
    
    # 列出可用的音频文件
    audio_dir = "data/audio"
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')] if os.path.exists(audio_dir) else []
    
    if not wav_files:
        print("没有找到音频文件")
        return
    
    for i, wav_file in enumerate(wav_files):
        print(f"{i+1}. {wav_file}")
    
    try:
        choice = int(input("\n请选择文件编号: ")) - 1
        if 0 <= choice < len(wav_files):
            wav_file = wav_files[choice]
            wav_path = os.path.join(audio_dir, wav_file)
            
            print(f"\n正在识别: {wav_file}")
            print("-" * 40)
            
            # 读取音频
            reader = WAVReader(wav_path)
            audio_data, sample_rate = reader.read()
            
            # 识别
            predicted_digit, confidence = recognizer.recognize(audio_data)
            
            print(f"识别结果: {predicted_digit}")
            print(f"置信度: {confidence:.3f}")
            
            if confidence > 0.7:
                print("识别结果: 高置信度")
            elif confidence > 0.5:
                print("识别结果: 中等置信度")
            else:
                print("识别结果: 低置信度，可能不准确")
                
        else:
            print("无效选择")
            
    except ValueError:
        print("无效输入")
    except Exception as e:
        print(f"识别过程中出现错误: {e}")


def create_sample_training_data():
    """创建示例训练数据"""
    print("创建示例训练数据结构...")
    
    # 创建目录
    for split in ['train', 'test']:
        for digit in range(10):
            digit_dir = f"data/{split}/digit_{digit}"
            os.makedirs(digit_dir, exist_ok=True)
    
    print("目录结构已创建:")
    print("data/")
    print("├── train/")
    print("│   ├── digit_0/  (放入数字0的多个训练样本)")
    print("│   ├── digit_1/  (放入数字1的多个训练样本)")
    print("│   ├── ...")
    print("│   └── digit_9/  (放入数字9的多个训练样本)")
    print("└── test/")
    print("    ├── digit_0/  (放入数字0的测试样本)")
    print("    ├── digit_1/  (放入数字1的测试样本)")
    print("    ├── ...")
    print("    └── digit_9/  (放入数字9的测试样本)")
    
    print("\n建议:")
    print("- 每个数字至少准备5-10个训练样本")
    print("- 每个数字准备2-3个测试样本")
    print("- 样本应该包含不同的说话人、语速、音量")
    print("- 文件命名建议: digit_0_sample1.wav, digit_0_sample2.wav, ...")


if __name__ == "__main__":
    speech_recognition_demo()
