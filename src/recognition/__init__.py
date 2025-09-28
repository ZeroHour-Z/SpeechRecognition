"""
语音识别模块
"""

from .classifiers import *
from .simple_recognizer import SimpleDigitRecognizer
from .advanced_recognizer import AdvancedDigitRecognizer

__all__ = [
    'SimpleDigitRecognizer', 'AdvancedDigitRecognizer'
]