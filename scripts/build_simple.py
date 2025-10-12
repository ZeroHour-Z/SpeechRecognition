#!/usr/bin/env python3
"""
简化版EXE打包脚本
"""

import os
import sys
import subprocess

def main():
    print("🎤 语音信号处理系统 - 简化打包")
    print("=" * 40)
    
    # 检查PyInstaller
    try:
        import PyInstaller
        print(f"✅ PyInstaller已安装: {PyInstaller.__version__}")
    except ImportError:
        print("正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",  # 打包成单个exe文件
        "--windowed",  # 不显示控制台窗口
        "--name=SpeechProcessingGUI",  # 输出文件名
        "--add-data=src;src",  # 包含src目录
        "--add-data=data;data",  # 包含data目录
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtGui", 
        "--hidden-import=PyQt5.QtWidgets",
        "--hidden-import=matplotlib.backends.backend_qt5agg",
        "--hidden-import=src.core.audio_recorder",
        "--hidden-import=src.core.endpoint_detection",
        "--hidden-import=src.core.frame_window",
        "--hidden-import=src.core.time_domain_analysis",
        "--hidden-import=src.core.wav_reader",
        "--hidden-import=src.recognition.advanced_recognizer",
        "--hidden-import=src.recognition.classifiers",
        "--hidden-import=src.recognition.simple_recognizer",
        "--hidden-import=src.utils.plot_config",
        "apps/run_gui_universal.py"
    ]
    
    print("开始打包...")
    try:
        subprocess.check_call(cmd)
        print("\n🎉 打包成功！")
        print("EXE文件位置: dist/SpeechProcessingGUI.exe")
    except subprocess.CalledProcessError as e:
        print(f"❌ 打包失败: {e}")

if __name__ == "__main__":
    main()


