"""
matplotlib中文字体配置模块
解决matplotlib显示中文字符的问题
"""

import matplotlib.pyplot as plt
import matplotlib
import platform
import os


def configure_english_font():
    """配置matplotlib使用英文字体"""
    # 设置英文字体参数
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return True


def get_english_labels():
    """获取英文标签，用于中文字体不可用时的备选方案"""
    return {
        # 窗函数相关
        'window_functions': {
            'rectangular': 'Rectangular Window',
            'hamming': 'Hamming Window', 
            'hanning': 'Hanning Window'
        },
        
        # 时域分析相关
        'time_domain': {
            'original_signal': 'Original Signal',
            'short_time_energy': 'Short-time Energy',
            'short_time_amplitude': 'Short-time Amplitude',
            'zero_crossing_rate': 'Zero Crossing Rate',
            'time_axis': 'Time (s)',
            'amplitude': 'Amplitude',
            'energy': 'Energy',
            'zcr': 'ZCR'
        },
        
        # 端点检测相关
        'endpoint_detection': {
            'speech_segments': 'Speech Segments',
            'detected_segments': 'Detected Segments',
            'energy_threshold': 'Energy Threshold',
            'zcr_threshold': 'ZCR Threshold',
            'speech_silence': 'Speech/Silence'
        },
        
        # 窗函数特性
        'window_properties': {
            'main_lobe_width': 'Main Lobe Width',
            'side_lobe_level': 'Side Lobe Level (dB)',
            'application': 'Application',
            'time_domain_analysis': 'Time Domain Analysis',
            'speech_analysis': 'Speech Analysis',
            'spectral_analysis': 'Spectral Analysis'
        }
    }


def setup_plot_style():
    """设置绘图样式"""
    # 设置图形样式
    plt.style.use('default')
    
    # 设置图形参数
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # 设置网格样式
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5
    
    # 设置线条样式
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6


def initialize_plotting():
    """初始化绘图环境"""
    # 配置英文字体
    configure_english_font()
    
    # 设置绘图样式
    setup_plot_style()
    
    return True


# 自动初始化
if __name__ != "__main__":
    # 当模块被导入时自动初始化
    initialize_plotting()
