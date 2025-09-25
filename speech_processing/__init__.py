"""
语音信号处理包
提供完整的语音信号分析、处理功能
"""

__version__ = "1.0.0"
__author__ = "DSP Lab"
__description__ = "语音信号处理实验系统"

# 首先初始化matplotlib英文字体配置
from .utils.plot_config import initialize_plotting
initialize_plotting()

# 导入核心模块
from .core.wav_reader import WAVReader, read_wav_file
from .core.frame_window import FrameProcessor, WindowFunctions
from .core.time_domain_analysis import TimeDomainAnalyzer
from .core.endpoint_detection import DualThresholdEndpointDetector

# 导入语音识别模块
from .recognition.simple_recognizer import SimpleDigitRecognizer, create_training_data_structure
from .recognition.advanced_recognizer import AdvancedDigitRecognizer

__all__ = [
    'WAVReader',
    'read_wav_file', 
    'FrameProcessor',
    'WindowFunctions',
    'TimeDomainAnalyzer',
    'DualThresholdEndpointDetector',
    'SimpleDigitRecognizer',
    'AdvancedDigitRecognizer',
    'create_training_data_structure'
]
