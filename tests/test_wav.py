"""
WAV文件读取模块测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src import WAVReader


def test_wav_reader():
    """测试WAV文件读取功能"""
    print("测试WAV文件读取功能...")
    
    # 查找测试文件
    audio_dir = "../data/audio"
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')] if os.path.exists(audio_dir) else []
    
    if not wav_files:
        print("没有找到测试用的WAV文件")
        return False
    
    # 测试第一个文件
    wav_file = os.path.join(audio_dir, wav_files[0])
    print(f"测试文件: {wav_file}")
    
    try:
        # 创建读取器
        reader = WAVReader(wav_file)
        
        # 读取文件
        audio_data, sample_rate = reader.read()
        
        # 验证结果
        assert isinstance(audio_data, np.ndarray), "音频数据应该是numpy数组"
        assert isinstance(sample_rate, int), "采样率应该是整数"
        assert sample_rate > 0, "采样率应该大于0"
        assert len(audio_data) > 0, "音频数据不应该为空"
        
        # 获取文件信息
        info = reader.get_info()
        assert 'sample_rate' in info, "文件信息应该包含采样率"
        assert 'duration' in info, "文件信息应该包含时长"
        
        print("✓ WAV文件读取测试通过")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  数据长度: {len(audio_data)} 采样点")
        print(f"  时长: {info['duration']:.3f} 秒")
        
        return True
        
    except Exception as e:
        print(f"✗ WAV文件读取测试失败: {e}")
        return False


if __name__ == "__main__":
    test_wav_reader()
