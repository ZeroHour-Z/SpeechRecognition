"""
WAV文件读取模块
实现WAV文件格式的解析和语音数据读取功能
"""

import wave
import struct
import numpy as np
from typing import Tuple, Optional
import os


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
            # 首先尝试使用wave模块
            try:
                with wave.open(self.file_path, 'rb') as wav_file:
                    return self._read_with_wave(wav_file)
            except Exception as wave_error:
                print(f"wave模块读取失败: {wave_error}")
                print("尝试使用原始数据读取...")
                return self._read_raw_data()
                
        except Exception as e:
            raise Exception(f"读取WAV文件失败: {e}")
    
    def _read_with_wave(self, wav_file) -> Tuple[np.ndarray, int]:
        """使用wave模块读取"""
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
            # 32位数据，可能是整数或浮点
            try:
                # 首先尝试32位浮点
                self.audio_data = np.frombuffer(raw_data, dtype=np.float32)
                print(f"成功读取32位浮点格式")
            except:
                # 如果失败，尝试32位整数
                self.audio_data = np.frombuffer(raw_data, dtype=np.int32)
                self.audio_data = self.audio_data / 2147483648.0
                print(f"成功读取32位整数格式")
        else:
            # 尝试使用numpy直接读取
            print(f"警告: 不常见的采样宽度 {self.sample_width} 字节，尝试直接读取...")
            self.audio_data = np.frombuffer(raw_data, dtype=np.uint8)
            self.audio_data = (self.audio_data - 128) / 128.0
        
        # 如果是多声道，取第一个声道
        if self.channels > 1:
            self.audio_data = self.audio_data[::self.channels]
        
        # 如果采样率不是16000Hz，进行重采样
        if self.sample_rate != 16000:
            print(f"采样率转换: {self.sample_rate}Hz -> 16000Hz")
            self.audio_data, self.sample_rate = self._resample_to_16k(self.audio_data, self.sample_rate)
        
        return self.audio_data, self.sample_rate
    
    def _read_raw_data(self) -> Tuple[np.ndarray, int]:
        """直接读取原始数据"""
        print("使用原始数据读取方法...")
        
        with open(self.file_path, 'rb') as f:
            # 读取WAV文件头
            header = f.read(44)
            
            # 解析基本信息
            if header[:4] != b'RIFF':
                raise Exception("不是有效的WAV文件")
            
            # 跳过一些字段，直接读取数据
            f.seek(44)  # 跳过WAV头
            raw_data = f.read()
            
            # 尝试不同的格式
            try:
                # 尝试32位浮点
                self.audio_data = np.frombuffer(raw_data, dtype=np.float32)
                self.sample_rate = 44100  # 假设是44100Hz
                self.channels = 2  # 假设是立体声
                # 确保数据在合理范围内
                self.audio_data = np.clip(self.audio_data, -1.0, 1.0)
                print("成功读取为32位浮点格式")
            except:
                try:
                    # 尝试16位整数
                    self.audio_data = np.frombuffer(raw_data, dtype=np.int16)
                    self.sample_rate = 16000
                    self.channels = 1
                    self.audio_data = self.audio_data / 32768.0
                    print("成功读取为16位整数格式")
                except:
                    # 最后尝试8位
                    self.audio_data = np.frombuffer(raw_data, dtype=np.uint8)
                    self.sample_rate = 16000
                    self.channels = 1
                    self.audio_data = (self.audio_data - 128) / 128.0
                    print("成功读取为8位格式")
            
            # 如果是多声道，取第一个声道
            if self.channels > 1:
                self.audio_data = self.audio_data[::self.channels]
            
            # 如果采样率不是16000Hz，进行重采样
            if self.sample_rate != 16000:
                print(f"采样率转换: {self.sample_rate}Hz -> 16000Hz")
                self.audio_data, self.sample_rate = self._resample_to_16k(self.audio_data, self.sample_rate)
            
            return self.audio_data, self.sample_rate
    
    def _resample_to_16k(self, audio_data: np.ndarray, original_rate: int) -> Tuple[np.ndarray, int]:
        """
        将音频重采样到16000Hz
        
        Args:
            audio_data: 原始音频数据
            original_rate: 原始采样率
            
        Returns:
            Tuple[np.ndarray, int]: (重采样后的音频数据, 16000)
        """
        try:
            # 简单的线性插值重采样
            target_rate = 16000
            ratio = target_rate / original_rate
            
            # 创建新的时间轴
            original_length = len(audio_data)
            new_length = int(original_length * ratio)
            
            # 使用numpy的插值进行重采样
            original_indices = np.arange(original_length)
            new_indices = np.linspace(0, original_length - 1, new_length)
            
            # 线性插值
            resampled_data = np.interp(new_indices, original_indices, audio_data)
            
            print(f"重采样完成: {original_length} -> {new_length} 采样点")
            return resampled_data, target_rate
            
        except Exception as e:
            print(f"重采样失败: {e}，使用原始数据")
            return audio_data, original_rate
    
    def get_info(self) -> dict:
        """
        获取WAV文件信息
        
        Returns:
            dict: 文件信息字典
        """
        if self.audio_data is None:
            self.read()
        
        return {
            'file_path': self.file_path,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'sample_width': self.sample_width,
            'frames': self.frames,
            'duration': self.duration,
            'data_length': len(self.audio_data) if self.audio_data is not None else 0,
            'data_type': str(self.audio_data.dtype) if self.audio_data is not None else None
        }
    
    def save(self, output_path: str, sample_rate: int = 16000):
        """
        保存音频数据为WAV文件
        
        Args:
            output_path: 输出文件路径
            sample_rate: 采样率
        """
        if self.audio_data is None:
            raise Exception("没有音频数据可保存")
        
        # 确保数据在[-1, 1]范围内
        audio_data = np.clip(self.audio_data, -1.0, 1.0)
        
        # 转换为16位整数
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)   # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data_int16.tobytes())
        
        print(f"音频已保存到: {output_path}")