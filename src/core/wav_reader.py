"""
WAV文件读取模块
实现WAV文件格式的解析和语音数据读取功能
"""

import wave
import struct
import numpy as np
from typing import Tuple, Optional


class WAVReader:
    """WAV文件读取器"""
    
    def __init__(self, file_path: str):
        """
        初始化WAV文件读取器
        
        Args:
            file_path: WAV文件路径
        """
        self.file_path = file_path
        self.sample_rate = None
        self.channels = None
        self.sample_width = None
        self.frames = None
        self.duration = None
        self.audio_data = None
        
    def read(self) -> Tuple[np.ndarray, int]:
        """
        读取WAV文件
        
        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        try:
            with wave.open(self.file_path, 'rb') as wav_file:
                # 获取WAV文件的基本信息
                self.channels = wav_file.getnchannels()  # 声道数
                self.sample_width = wav_file.getsampwidth()  # 采样宽度（字节）
                self.sample_rate = wav_file.getframerate()  # 采样率
                self.frames = wav_file.getnframes()  # 总帧数
                self.duration = self.frames / self.sample_rate  # 时长（秒）
                
                # 读取音频数据
                raw_data = wav_file.readframes(self.frames)
                
                # 根据采样宽度解析数据
                if self.sample_width == 1:
                    # 8位无符号整数
                    self.audio_data = np.frombuffer(raw_data, dtype=np.uint8)
                    self.audio_data = (self.audio_data - 128) / 128.0
                elif self.sample_width == 2:
                    # 16位有符号整数
                    self.audio_data = np.frombuffer(raw_data, dtype=np.int16)
                    self.audio_data = self.audio_data / 32768.0
                elif self.sample_width == 3:
                    # 24位有符号整数（需要特殊处理）
                    # 将3字节数据转换为4字节整数
                    extended_data = bytearray()
                    for i in range(0, len(raw_data), 3):
                        # 读取3字节，扩展为4字节
                        if i + 2 < len(raw_data):
                            byte1, byte2, byte3 = raw_data[i], raw_data[i+1], raw_data[i+2]
                            # 检查符号位（最高位）
                            if byte3 & 0x80:  # 负数
                                extended_data.extend([0xFF, byte3, byte2, byte1])
                            else:  # 正数
                                extended_data.extend([0x00, byte3, byte2, byte1])
                    self.audio_data = np.frombuffer(extended_data, dtype=np.int32)
                    self.audio_data = self.audio_data / 8388608.0  # 2^23
                elif self.sample_width == 4:
                    # 32位有符号整数
                    self.audio_data = np.frombuffer(raw_data, dtype=np.int32)
                    self.audio_data = self.audio_data / 2147483648.0
                else:
                    # 尝试使用numpy直接读取
                    print(f"警告: 不常见的采样宽度 {self.sample_width} 字节，尝试直接读取...")
                    self.audio_data = np.frombuffer(raw_data, dtype=np.uint8)
                    self.audio_data = (self.audio_data - 128) / 128.0
                
                # 如果是多声道，取第一个声道
                if self.channels > 1:
                    self.audio_data = self.audio_data[::self.channels]
                
                return self.audio_data, self.sample_rate
                
        except Exception as e:
            raise Exception(f"读取WAV文件失败: {e}")
    
    def get_info(self) -> dict:
        """
        获取WAV文件信息
        
        Returns:
            dict: 文件信息字典
        """
        if self.sample_rate is None:
            self.read()
            
        return {
            'file_path': self.file_path,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'sample_width': self.sample_width,
            'frames': self.frames,
            'duration': self.duration,
            'data_length': len(self.audio_data) if self.audio_data is not None else 0
        }
    
    def print_info(self):
        """打印WAV文件信息"""
        info = self.get_info()
        print("=" * 50)
        print("WAV文件信息:")
        print(f"文件路径: {info['file_path']}")
        print(f"采样率: {info['sample_rate']} Hz")
        print(f"声道数: {info['channels']}")
        print(f"采样宽度: {info['sample_width']} 字节")
        print(f"总帧数: {info['frames']}")
        print(f"时长: {info['duration']:.3f} 秒")
        print(f"数据长度: {info['data_length']} 采样点")
        print("=" * 50)


def read_wav_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    快速读取WAV文件的便捷函数
    
    Args:
        file_path: WAV文件路径
        
    Returns:
        Tuple[np.ndarray, int]: (音频数据, 采样率)
    """
    reader = WAVReader(file_path)
    return reader.read()


if __name__ == "__main__":
    # 测试代码
    import os
    
    # 查找当前目录下的WAV文件
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if wav_files:
        test_file = wav_files[0]
        print(f"测试文件: {test_file}")
        
        reader = WAVReader(test_file)
        audio_data, sample_rate = reader.read()
        reader.print_info()
        
        print(f"音频数据范围: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
        print(f"音频数据均值: {audio_data.mean():.6f}")
    else:
        print("当前目录下没有找到WAV文件")
        print("请将WAV文件放在当前目录下进行测试")
