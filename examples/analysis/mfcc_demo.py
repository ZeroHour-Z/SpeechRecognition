"""
MFCC 特征提取演示 - 实验二
演示如何使用频域分析模块提取 MFCC 特征
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import WAVReader, FrequencyDomainAnalyzer
import matplotlib.pyplot as plt
import numpy as np


def mfcc_extraction_demo():
    """MFCC 特征提取完整演示"""
    print("=" * 60)
    print("实验二：MFCC 特征提取演示")
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
        print("请将WAV文件放在该目录下")
        return
    
    # 显示可用的音频文件
    print("\n可用的音频文件:")
    for i, file in enumerate(wav_files):
        print(f"{i+1}. {file}")
    
    # 让用户选择文件
    while True:
        try:
            choice = input(f"\n请选择要分析的文件编号 (1-{len(wav_files)}): ")
            file_index = int(choice) - 1
            if 0 <= file_index < len(wav_files):
                break
            else:
                print(f"请输入1到{len(wav_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    # 使用用户选择的文件
    wav_file = os.path.join(audio_dir, wav_files[file_index])
    print(f"\n分析文件: {wav_file}")
    
    # 1. 读取WAV文件
    print("\n步骤 1: 读取WAV文件...")
    reader = WAVReader(wav_file)
    audio_data, sample_rate = reader.read()
    print(f"  采样率: {sample_rate} Hz")
    print(f"  数据长度: {len(audio_data)} 采样点")
    print(f"  时长: {len(audio_data) / sample_rate:.3f} 秒")
    
    # 2. 创建频域分析器
    print("\n步骤 2: 创建频域分析器...")
    analyzer = FrequencyDomainAnalyzer(
        sample_rate=sample_rate,
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        n_mels=26,
        n_mfcc=13
    )
    
    # 3. 提取 MFCC 特征
    print("\n步骤 3: 提取 MFCC 特征...")
    print("  执行完整流程：")
    print("    1. 预处理：预加重，分帧，加窗")
    print("    2. FFT")
    print("    3. 计算谱线能量")
    print("    4. 计算 Mel 滤波器能量")
    print("    5. 经离散余弦变换 (DCT) 得到 MFCC 系数")
    
    result = analyzer.extract_mfcc(audio_data, window_type='hamming')
    
    print(f"\n  ✓ MFCC 特征提取完成")
    print(f"    总帧数: {result['num_frames']}")
    print(f"    MFCC 特征形状: {result['mfcc'].shape}")
    print(f"    MFCC 系数范围: [{result['mfcc'].min():.4f}, {result['mfcc'].max():.4f}]")
    
    # 4. 可视化提取过程
    print("\n步骤 4: 可视化 MFCC 提取过程...")
    analyzer.plot_mfcc_extraction_process(result, start_frame=0, num_frames=20)
    
    # 5. 显示 MFCC 特征的统计信息
    print("\n步骤 5: MFCC 特征统计信息...")
    mfcc_mean = np.mean(result['mfcc'], axis=0)
    mfcc_std = np.std(result['mfcc'], axis=0)
    
    print("\n各 MFCC 系数的均值和标准差:")
    print(f"{'系数':<8} {'均值':<12} {'标准差':<12}")
    print("-" * 35)
    for i in range(len(mfcc_mean)):
        print(f"MFCC {i:<3} {mfcc_mean[i]:>10.4f} {mfcc_std[i]:>10.4f}")
    
    # 6. 比较不同参数设置
    print("\n步骤 6: 比较不同参数设置对 MFCC 特征的影响...")
    compare_mfcc_parameters(audio_data, sample_rate)
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


def compare_mfcc_parameters(signal: np.ndarray, sample_rate: int) -> None:
    """
    比较不同参数设置对 MFCC 特征的影响
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
    """
    # 不同的参数组合
    param_sets = [
        {'n_mels': 13, 'n_mfcc': 13, 'label': '标准参数 (Mel=13, MFCC=13)'},
        {'n_mels': 26, 'n_mfcc': 13, 'label': '更多滤波器 (Mel=26, MFCC=13)'},
        {'n_mels': 26, 'n_mfcc': 20, 'label': '更多系数 (Mel=26, MFCC=20)'},
    ]
    
    plt.figure(figsize=(15, 10))
    
    for idx, params in enumerate(param_sets):
        analyzer = FrequencyDomainAnalyzer(
            sample_rate,
            n_mels=params['n_mels'],
            n_mfcc=params['n_mfcc']
        )
        result = analyzer.extract_mfcc(signal)
        
        # 显示 MFCC 特征图
        plt.subplot(2, 2, idx + 1)
        mfcc_display = result['mfcc'].T
        plt.imshow(mfcc_display, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='MFCC 系数值')
        plt.title(f"{params['label']}")
        plt.xlabel('帧索引')
        plt.ylabel('MFCC 系数索引')
    
    # 比较第一个 MFCC 系数
    plt.subplot(2, 2, 4)
    for params in param_sets:
        analyzer = FrequencyDomainAnalyzer(
            sample_rate,
            n_mels=params['n_mels'],
            n_mfcc=params['n_mfcc']
        )
        result = analyzer.extract_mfcc(signal)
        plt.plot(result['time_axis'], result['mfcc'][:, 0], 
                linewidth=2, label=params['label'])
    plt.title('第一个 MFCC 系数对比')
    plt.xlabel('时间 (s)')
    plt.ylabel('MFCC 系数值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_individual_steps():
    """演示 MFCC 提取的各个步骤"""
    print("=" * 60)
    print("MFCC 提取步骤详细演示")
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
    
    wav_file = os.path.join(audio_dir, wav_files[0])
    print(f"使用文件: {wav_file}")
    
    # 读取文件
    reader = WAVReader(wav_file)
    audio_data, sample_rate = reader.read()
    
    # 创建分析器
    analyzer = FrequencyDomainAnalyzer(sample_rate, n_mels=26, n_mfcc=13)
    
    # 步骤 1: 预加重
    print("\n步骤 1: 预加重处理")
    emphasized = analyzer.pre_emphasis(audio_data)
    print(f"  原始信号范围: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
    print(f"  预加重后范围: [{emphasized.min():.4f}, {emphasized.max():.4f}]")
    
    # 步骤 2: 分帧和加窗
    print("\n步骤 2: 分帧和加窗")
    frames, windowed_frames = analyzer.frame_processor.process_signal(emphasized, 'hamming')
    print(f"  总帧数: {len(frames)}")
    print(f"  每帧长度: {len(frames[0])} 采样点")
    
    # 步骤 3: FFT
    print("\n步骤 3: 短时傅里叶变换 (STFT)")
    spectra = analyzer.stft(windowed_frames)
    print(f"  频谱形状: {spectra.shape}")
    print(f"  频率分辨率: {sample_rate / analyzer.n_fft:.2f} Hz")
    
    # 步骤 4: 功率谱
    print("\n步骤 4: 计算功率谱（谱线能量）")
    power_spectrum = analyzer.compute_power_spectrum(spectra)
    print(f"  功率谱形状: {power_spectrum.shape}")
    print(f"  功率范围: [{power_spectrum.min():.6f}, {power_spectrum.max():.6f}]")
    
    # 步骤 5: Mel 滤波器能量
    print("\n步骤 5: 计算 Mel 滤波器能量")
    mel_energies = analyzer.apply_mel_filterbank(power_spectrum)
    print(f"  Mel 能量形状: {mel_energies.shape}")
    print(f"  Mel 能量范围: [{mel_energies.min():.6f}, {mel_energies.max():.6f}]")
    
    # 步骤 6: MFCC
    print("\n步骤 6: 计算 MFCC 系数（DCT）")
    mfcc = analyzer.compute_mfcc(mel_energies)
    print(f"  MFCC 形状: {mfcc.shape}")
    print(f"  MFCC 范围: [{mfcc.min():.4f}, {mfcc.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("各步骤演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行完整演示
    mfcc_extraction_demo()
    
    # 如果需要，可以运行详细步骤演示
    # demonstrate_individual_steps()

