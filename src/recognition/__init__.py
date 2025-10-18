"""
语音识别模块
"""

from .classifiers import *
from .simple_recognizer import SimpleDigitRecognizer, create_training_data_structure
from .advanced_recognizer import AdvancedDigitRecognizer

__all__ = [
    'SimpleDigitRecognizer', 'AdvancedDigitRecognizer', 'AdvancedRecognizer',
    'create_training_data_structure'
]

# 为了向后兼容，添加别名
AdvancedRecognizer = AdvancedDigitRecognizer