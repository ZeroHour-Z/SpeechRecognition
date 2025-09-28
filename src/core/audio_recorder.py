"""
音频录音模块
实现实时音频录制功能
"""

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError as e:
    PYAUDIO_AVAILABLE = False
    print("警告: pyaudio未安装，音频录制功能不可用")
    print(f"错误详情: {e}")
    print("解决方案:")
    print("1. Ubuntu/Debian: sudo apt install portaudio19-dev python3-pyaudio")
    print("2. CentOS/RHEL: sudo yum install portaudio-devel && pip install pyaudio")
    print("3. 或者使用conda: conda install pyaudio")
    print("4. 如果仍有问题，请检查系统音频驱动")

import numpy as np
import wave
import threading
import time
from typing import Optional, Callable, Tuple
import os


class AudioRecorder:
    """音频录音器"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 format: int = None):
        """
        初始化音频录音器
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            chunk_size: 每次读取的帧数
            format: 音频格式
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format if format is not None else (pyaudio.paInt16 if PYAUDIO_AVAILABLE else 16)
        
        # PyAudio对象
        if PYAUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = None
        self.stream = None
        
        # 录音状态
        self.is_recording = False
        self.recording_thread = None
        
        # 录音数据
        self.recording_data = []
        self.callback_function = None
        
    def initialize(self) -> bool:
        """
        初始化音频系统
        
        Returns:
            bool: 初始化是否成功
        """
        if not PYAUDIO_AVAILABLE:
            print("pyaudio不可用，无法初始化音频系统")
            return False
            
        try:
            self.audio = pyaudio.PyAudio()
            return True
        except Exception as e:
            print(f"音频系统初始化失败: {e}")
            return False
    
    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """
        开始录音
        
        Args:
            callback: 实时数据回调函数
            
        Returns:
            bool: 开始录音是否成功
        """
        if self.is_recording:
            return False
            
        if not self.audio:
            if not self.initialize():
                return False
        
        try:
            # 打开音频流
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.recording_data = []
            self.callback_function = callback
            
            # 启动录音线程
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.start()
            
            return True
            
        except Exception as e:
            print(f"开始录音失败: {e}")
            return False
    
    def stop_recording(self) -> np.ndarray:
        """
        停止录音
        
        Returns:
            np.ndarray: 录音数据
        """
        if not self.is_recording:
            return np.array([])
        
        self.is_recording = False
        
        # 等待录音线程结束
        if self.recording_thread:
            self.recording_thread.join()
        
        # 关闭音频流
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # 返回录音数据
        if self.recording_data:
            audio_data = np.concatenate(self.recording_data)
            return audio_data.astype(np.float32) / 32768.0  # 归一化到[-1, 1]
        else:
            return np.array([])
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        if self.is_recording:
            # 将字节数据转换为numpy数组
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            self.recording_data.append(audio_chunk)
            
            # 调用用户回调函数
            if self.callback_function:
                try:
                    self.callback_function(audio_chunk)
                except Exception as e:
                    print(f"回调函数执行错误: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def _recording_loop(self):
        """录音循环"""
        if self.stream:
            self.stream.start_stream()
            while self.is_recording:
                time.sleep(0.01)  # 10ms间隔
    
    def save_recording(self, audio_data: np.ndarray, filename: str) -> bool:
        """
        保存录音数据到WAV文件
        
        Args:
            audio_data: 音频数据
            filename: 文件名
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 转换为16位整数
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # 保存为WAV文件
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16位 = 2字节
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return True
            
        except Exception as e:
            print(f"保存录音失败: {e}")
            return False
    
    def get_available_devices(self) -> list:
        """
        获取可用的音频设备列表
        
        Returns:
            list: 设备信息列表
        """
        if not self.audio:
            if not self.initialize():
                return []
        
        devices = []
        try:
            device_count = self.audio.get_device_count()
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # 只显示输入设备
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
        except Exception as e:
            print(f"获取设备列表失败: {e}")
        
        return devices
    
    def cleanup(self):
        """清理资源"""
        if self.is_recording:
            self.stop_recording()
        
        if self.audio:
            self.audio.terminate()
            self.audio = None


class RealTimeAnalyzer:
    """实时音频分析器"""
    
    def __init__(self, sample_rate: int = 16000, frame_length_ms: float = 25.0):
        """
        初始化实时分析器
        
        Args:
            sample_rate: 采样率
            frame_length_ms: 帧长（毫秒）
        """
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.buffer = []
        
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        添加音频块并进行分析
        
        Args:
            audio_chunk: 音频数据块
            
        Returns:
            dict: 分析结果
        """
        # 添加到缓冲区
        self.buffer.extend(audio_chunk)
        
        # 如果缓冲区足够长，进行分析
        if len(self.buffer) >= self.frame_length:
            # 取一帧数据
            frame_data = np.array(self.buffer[:self.frame_length])
            self.buffer = self.buffer[self.frame_length:]
            
            # 计算时域特征
            energy = np.sum(frame_data ** 2)
            amplitude = np.mean(np.abs(frame_data))
            zcr = self._calculate_zcr(frame_data)
            
            return {
                'energy': energy,
                'amplitude': amplitude,
                'zcr': zcr,
                'is_speech': energy > 0.01  # 简单的语音检测阈值
            }
        
        return {}
    
    def _calculate_zcr(self, signal: np.ndarray) -> float:
        """计算过零率"""
        if len(signal) < 2:
            return 0.0
        
        sign_changes = np.sum(np.diff(np.sign(signal)) != 0)
        return sign_changes / (len(signal) - 1)
    
    def reset(self):
        """重置缓冲区"""
        self.buffer = []
