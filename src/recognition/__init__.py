"""
语音识别模块
"""

from .classifiers import *
from .simple_recognizer import SimpleDigitRecognizer, create_training_data_structure
from .advanced_recognizer import AdvancedDigitRecognizer
from .dtw import DTW
from .mfcc_dtw_recognizer import MFCCDTWRecognizer

__all__ = [
    'SimpleDigitRecognizer', 'AdvancedDigitRecognizer', 'AdvancedRecognizer',
    'create_training_data_structure', 'DTW', 'MFCCDTWRecognizer'
]

# 为了向后兼容，添加别名
AdvancedRecognizer = AdvancedDigitRecognizer