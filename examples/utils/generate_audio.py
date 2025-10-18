"""
测试音频文件生成器
生成一些测试用的WAV文件，用于演示语音处理功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import wave
from src import WAVReader


def generate_sine_wave(frequency, duration, sample_rate=16000, amplitude=0.5):
    """生成正弦波信号"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal


def generate_speech_like_signal(duration, sample_rate=16000):
    """生成类似语音的信号（多个频率的正弦波组合）"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 基础频率（类似基频）
    base_freq = 100
    signal = 0.3 * np.sin(2 * np.pi * base_freq * t)
    
    # 添加谐波
    signal += 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)
    signal += 0.15 * np.sin(2 * np.pi * base_freq * 3 * t)
    signal += 0.1 * np.sin(2 * np.pi * base_freq * 4 * t)
    
    # 添加一些噪声
    noise = 0.05 * np.random.randn(len(signal))
    signal += noise
    
    # 添加包络（模拟语音的强度变化）
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))  # 2Hz的包络
    signal *= envelope
    
    return signal


def generate_silence(duration, sample_rate=16000):
    """生成静音信号"""
    return np.zeros(int(sample_rate * duration))


def save_wav_file(signal, sample_rate, filename):
    """保存信号为WAV文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 转换为16位整数
    signal_int16 = (signal * 32767).astype(np.int16)
    
    # 保存WAV文件
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_int16.tobytes())


def generate_test_files():
    """生成测试文件"""
    print("生成测试音频文件...")
    
    sample_rate = 16000
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = os.path.join(project_root, "data", "audio")
    
    # 1. 纯音测试文件
    print("1. 生成纯音测试文件...")
    sine_440 = generate_sine_wave(440, 2.0, sample_rate)  # 440Hz，2秒
    save_wav_file(sine_440, sample_rate, f"{audio_dir}/test_440hz.wav")
    
    sine_880 = generate_sine_wave(880, 2.0, sample_rate)  # 880Hz，2秒
    save_wav_file(sine_880, sample_rate, f"{audio_dir}/test_880hz.wav")
    
    # 2. 语音模拟文件
    print("2. 生成语音模拟文件...")
    speech_like = generate_speech_like_signal(3.0, sample_rate)  # 3秒
    save_wav_file(speech_like, sample_rate, f"{audio_dir}/test_speech_like.wav")
    
    # 3. 带静音的语音文件
    print("3. 生成带静音的语音文件...")
    # 静音 + 语音 + 静音
    silence1 = generate_silence(0.5, sample_rate)
    speech = generate_speech_like_signal(2.0, sample_rate)
    silence2 = generate_silence(0.5, sample_rate)
    
    combined = np.concatenate([silence1, speech, silence2])
    save_wav_file(combined, sample_rate, f"{audio_dir}/test_speech_with_silence.wav")
    
    # 4. 数字语音模拟（0-9）
    print("4. 生成数字语音模拟文件...")
    frequencies = [261, 293, 329, 349, 392, 440, 493, 523, 587, 659]  # C4到E5
    
    for i, freq in enumerate(frequencies):
        # 每个数字包含静音+语音+静音
        silence = generate_silence(0.2, sample_rate)
        digit_signal = generate_sine_wave(freq, 0.8, sample_rate, 0.3)
        silence_end = generate_silence(0.2, sample_rate)
        
        digit_combined = np.concatenate([silence, digit_signal, silence_end])
        save_wav_file(digit_combined, sample_rate, f"{audio_dir}/digit_{i}.wav")
    
    print("测试文件生成完成！")
    print(f"文件保存在: {audio_dir}")
    
    # 列出生成的文件
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"\n生成的WAV文件 ({len(wav_files)} 个):")
    for file in sorted(wav_files):
        print(f"  - {file}")


def test_generated_files():
    """测试生成的文件"""
    print("\n测试生成的文件...")
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = os.path.join(project_root, "data", "audio")
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    for wav_file in wav_files[:3]:  # 只测试前3个文件
        file_path = os.path.join(audio_dir, wav_file)
        print(f"\n测试文件: {wav_file}")
        
        try:
            reader = WAVReader(file_path)
            audio_data, sample_rate = reader.read()
            
            print(f"  采样率: {sample_rate} Hz")
            print(f"  数据长度: {len(audio_data)} 采样点")
            print(f"  时长: {len(audio_data)/sample_rate:.3f} 秒")
            print(f"  幅度范围: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
            print("  ✓ 读取成功")
            
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")


if __name__ == "__main__":
    generate_test_files()
    test_generated_files()
    
    print("\n" + "="*60)
    print("测试文件生成完成！")
    print("现在可以运行主程序进行语音处理实验：")
    print("python main.py")
    print("="*60)
