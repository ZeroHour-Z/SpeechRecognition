"""
语音信号处理系统配置文件
"""

# 默认参数配置
DEFAULT_CONFIG = {
    # 分帧参数
    'frame_length_ms': 25.0,      # 帧长（毫秒）
    'frame_shift_ms': 10.0,       # 帧移（毫秒）
    
    # 窗函数类型
    'window_type': 'hamming',     # 默认窗函数类型
    
    # 端点检测参数
    'energy_ratio': 0.1,          # 能量阈值比例
    'zcr_ratio': 1.5,             # 过零率阈值比例
    'min_speech_frames': 3,       # 最小语音帧数
    'min_silence_frames': 5,      # 最小静音帧数
    
    # 文件路径
    'audio_dir': 'data/audio',    # 音频文件目录
    'results_dir': 'data/results', # 结果保存目录
    
    # 可视化参数
    'figure_size': (15, 10),      # 图形大小
    'dpi': 100,                   # 图形分辨率
    'show_plots': True,           # 是否显示图形
}

# 支持的窗函数类型
SUPPORTED_WINDOWS = ['rectangular', 'hamming', 'hanning']

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = ['.wav']

# 采样率范围
SAMPLE_RATE_RANGE = (8000, 48000)

# 帧长范围（毫秒）
FRAME_LENGTH_RANGE = (10, 50)

# 帧移范围（毫秒）
FRAME_SHIFT_RANGE = (5, 25)
