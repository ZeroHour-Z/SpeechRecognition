"""
分帧处理器测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from speech_processing import WAVReader, FrameProcessor, WindowFunctions


def test_frame_processor():
    """测试分帧处理功能"""
    print("测试分帧处理功能...")
    
    # 创建测试信号
    sample_rate = 16000
    duration = 1.0  # 1秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_signal = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
    
    try:
        # 创建分帧处理器
        processor = FrameProcessor(sample_rate, 25.0, 10.0)
        
        # 分帧处理
        frames, windowed_frames = processor.process_signal(test_signal, 'hamming')
        
        # 验证结果
        assert len(frames) > 0, "应该有分帧结果"
        assert len(windowed_frames) == len(frames), "加窗帧数应该等于原始帧数"
        assert len(frames[0]) == processor.frame_length, "帧长应该正确"
        
        print("✓ 分帧处理测试通过")
        print(f"  总帧数: {len(frames)}")
        print(f"  帧长: {len(frames[0])} 采样点")
        print(f"  帧移: {processor.frame_shift} 采样点")
        
        return True
        
    except Exception as e:
        print(f"✗ 分帧处理测试失败: {e}")
        return False


def test_window_functions():
    """测试窗函数"""
    print("测试窗函数...")
    
    try:
        frame_length = 256
        
        # 测试三种窗函数
        rect_window = WindowFunctions.rectangular_window(frame_length)
        hamming_window = WindowFunctions.hamming_window(frame_length)
        hanning_window = WindowFunctions.hanning_window(frame_length)
        
        # 验证窗函数
        assert len(rect_window) == frame_length, "矩形窗长度应该正确"
        assert len(hamming_window) == frame_length, "汉明窗长度应该正确"
        assert len(hanning_window) == frame_length, "海宁窗长度应该正确"
        
        # 验证窗函数值范围
        assert np.all(rect_window == 1.0), "矩形窗值应该全为1"
        assert np.all(hamming_window >= 0) and np.all(hamming_window <= 1), "汉明窗值应该在[0,1]范围内"
        assert np.all(hanning_window >= 0) and np.all(hanning_window <= 1), "海宁窗值应该在[0,1]范围内"
        
        print("✓ 窗函数测试通过")
        print(f"  矩形窗: 长度={len(rect_window)}, 值范围=[{rect_window.min():.3f}, {rect_window.max():.3f}]")
        print(f"  汉明窗: 长度={len(hamming_window)}, 值范围=[{hamming_window.min():.3f}, {hamming_window.max():.3f}]")
        print(f"  海宁窗: 长度={len(hanning_window)}, 值范围=[{hanning_window.min():.3f}, {hanning_window.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ 窗函数测试失败: {e}")
        return False


if __name__ == "__main__":
    test_frame_processor()
    test_window_functions()
