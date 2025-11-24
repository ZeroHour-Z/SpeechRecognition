"""
DTW 和 MFCC-DTW 语音识别演示 - 实验三
演示如何使用 DTW 算法和基于 MFCC 特征的语音识别
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import WAVReader, MFCCDTWRecognizer, DTW, FrequencyDomainAnalyzer
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def dtw_algorithm_demo():
    """演示 DTW 算法的工作原理"""
    print("=" * 60)
    print("DTW 算法演示")
    print("=" * 60)
    
    # 查找音频文件
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    audio_dir = os.path.join(project_root, "data", "audio", "samples")
    
    if not os.path.exists(audio_dir):
        print(f"音频目录不存在: {audio_dir}")
        return
    
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"在 {audio_dir} 目录下没有找到WAV文件")
        return
    
    # 使用前两个文件进行演示
    if len(wav_files) < 2:
        print("需要至少2个WAV文件进行DTW演示")
        return
    
    wav_file1 = os.path.join(audio_dir, wav_files[0])
    wav_file2 = os.path.join(audio_dir, wav_files[1])
    
    print(f"文件1: {wav_files[0]}")
    print(f"文件2: {wav_files[1]}")
    
    # 读取文件
    reader1 = WAVReader(wav_file1)
    audio_data1, sample_rate1 = reader1.read()
    
    reader2 = WAVReader(wav_file2)
    audio_data2, sample_rate2 = reader2.read()
    
    # 提取 MFCC 特征
    analyzer = FrequencyDomainAnalyzer(sample_rate1, n_mels=26, n_mfcc=13)
    
    result1 = analyzer.extract_mfcc(audio_data1)
    mfcc1 = result1['mfcc']
    
    result2 = analyzer.extract_mfcc(audio_data2)
    mfcc2 = result2['mfcc']
    
    print(f"\n序列1长度: {len(mfcc1)} 帧")
    print(f"序列2长度: {len(mfcc2)} 帧")
    print(f"MFCC 特征维度: {mfcc1.shape[1]}")
    
    # 计算 DTW 距离
    dtw = DTW(distance_metric='euclidean')
    distance, accumulated_cost, path = dtw.dtw(mfcc1, mfcc2)
    
    print(f"\nDTW 距离: {distance:.4f}")
    print(f"对齐路径长度: {len(path)}")
    
    # 可视化 DTW 路径
    dtw.visualize_dtw_path(mfcc1, mfcc2, path, accumulated_cost, 
                          "DTW 算法演示：两个语音信号的 MFCC 特征对齐")
    
    print("\n" + "=" * 60)
    print("DTW 算法演示完成！")
    print("=" * 60)


def mfcc_dtw_recognition_demo():
    """MFCC-DTW 语音识别完整演示"""
    print("=" * 60)
    print("实验三：基于 MFCC 和 DTW 的语音识别演示")
    print("=" * 60)
    
    # 查找训练和测试数据
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_dir = os.path.join(project_root, "data", "train")
    test_dir = os.path.join(project_root, "data", "test")
    
    if not os.path.exists(train_dir):
        print(f"训练数据目录不存在: {train_dir}")
        print("请确保训练数据目录存在，格式为:")
        print("data/train/")
        print("  ├── digit_0/")
        print("  ├── digit_1/")
        print("  └── ...")
        return
    
    # 创建识别器
    print("\n步骤 1: 创建 MFCC-DTW 识别器...")
    recognizer = MFCCDTWRecognizer(
        sample_rate=16000,
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        n_mels=26,
        n_mfcc=13,
        dtw_window=None  # 不限制窗口
    )
    
    # 训练识别器
    print("\n步骤 2: 训练识别器...")
    print("  读取训练数据，提取 MFCC 特征，存储为模板...")
    recognizer.train(train_dir)
    
    if not recognizer.templates:
        print("训练失败：没有有效的训练模板")
        return
    
    # 测试识别
    if os.path.exists(test_dir):
        print("\n步骤 3: 测试识别准确率...")
        results = recognizer.test_recognition(test_dir)
        
        # 显示结果摘要
        print("\n识别结果摘要:")
        print("=" * 60)
        for key, value in results.items():
            if key != 'overall':
                print(f"{key}: {value:.2%}")
        print(f"总体准确率: {results.get('overall', 0):.2%}")
        print("=" * 60)
    else:
        print(f"\n测试数据目录不存在: {test_dir}")
        print("跳过测试步骤")
    
    # 交互式识别演示
    print("\n步骤 4: 交互式识别演示...")
    print("可以输入音频文件路径进行识别（输入 'q' 退出）")
    
    while True:
        try:
            audio_file = input("\n请输入音频文件路径（或 'q' 退出）: ").strip()
            
            if audio_file.lower() == 'q':
                break
            
            if not os.path.exists(audio_file):
                print(f"文件不存在: {audio_file}")
                continue
            
            # 读取并识别
            reader = WAVReader(audio_file)
            audio_data, file_sample_rate = reader.read()
            
            # 如果采样率不匹配，进行重采样
            if file_sample_rate != recognizer.sample_rate:
                from scipy import signal as scipy_signal
                num_samples = int(len(audio_data) * recognizer.sample_rate / file_sample_rate)
                audio_data = scipy_signal.resample(audio_data, num_samples)
            
            predicted_digit, confidence, distances = recognizer.recognize(audio_data)
            
            if predicted_digit >= 0:
                print(f"识别结果: {predicted_digit}")
                print(f"置信度: {confidence:.3f}")
                print(f"DTW 距离: {distances[predicted_digit]:.2f}")
                print("\n所有数字的距离:")
                for digit, dist in sorted(distances.items()):
                    print(f"  数字 {digit}: {dist:.2f}")
            else:
                print("识别失败：无法提取有效特征")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


def compare_recognition_methods():
    """比较不同识别方法"""
    print("=" * 60)
    print("比较不同识别方法")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_dir = os.path.join(project_root, "data", "train")
    test_dir = os.path.join(project_root, "data", "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("需要训练和测试数据目录")
        return
    
    # 测试不同的 DTW 窗口大小
    window_sizes = [None, 5, 10, 20]
    results = {}
    
    for window in window_sizes:
        print(f"\n测试 DTW 窗口大小: {window if window else '无限制'}")
        recognizer = MFCCDTWRecognizer(
            sample_rate=16000,
            n_mels=26,
            n_mfcc=13,
            dtw_window=window
        )
        recognizer.train(train_dir)
        
        if recognizer.templates:
            test_results = recognizer.test_recognition(test_dir)
            results[window] = test_results.get('overall', 0.0)
            print(f"准确率: {results[window]:.2%}")
    
    # 可视化比较结果
    plt.figure(figsize=(10, 6))
    window_labels = [str(w) if w else '无限制' for w in window_sizes]
    accuracies = [results.get(w, 0.0) for w in window_sizes]
    
    plt.bar(window_labels, accuracies, alpha=0.7)
    plt.xlabel('DTW 窗口大小')
    plt.ylabel('识别准确率')
    plt.title('不同 DTW 窗口大小对识别准确率的影响')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("比较完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行 DTW 算法演示
    dtw_algorithm_demo()
    
    # 运行 MFCC-DTW 识别演示
    # mfcc_dtw_recognition_demo()
    
    # 比较不同识别方法
    # compare_recognition_methods()

