"""
语音信号处理工具模块
包含可视化、报告生成等辅助功能
"""

from .plot_config import initialize_plotting, get_english_labels

# 自动初始化matplotlib英文字体
initialize_plotting()

__all__ = [
    'initialize_plotting',
    'get_english_labels'
]
