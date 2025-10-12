#!/usr/bin/env python3
"""
语音信号处理系统 - EXE打包脚本
使用PyInstaller将Qt GUI打包成可执行文件
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_pyinstaller():
    """检查PyInstaller是否已安装"""
    try:
        import PyInstaller
        print(f"✅ PyInstaller已安装: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("❌ PyInstaller未安装")
        return False

def install_pyinstaller():
    """安装PyInstaller"""
    print("正在安装PyInstaller...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyInstaller安装失败: {e}")
        return False

def create_spec_file():
    """创建PyInstaller spec文件"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 数据文件
datas = [
    ('data', 'data'),
    ('src', 'src'),
    ('examples', 'examples'),
    ('docs', 'docs'),
]

# 隐藏导入
hiddenimports = [
    'PyQt5.QtCore',
    'PyQt5.QtGui', 
    'PyQt5.QtWidgets',
    'matplotlib.backends.backend_qt5agg',
    'numpy',
    'scipy',
    'matplotlib',
    'pyaudio',
    'sklearn',
    'torch',
    'src.core.audio_recorder',
    'src.core.endpoint_detection',
    'src.core.frame_window',
    'src.core.time_domain_analysis',
    'src.core.wav_reader',
    'src.recognition.advanced_recognizer',
    'src.recognition.classifiers',
    'src.recognition.simple_recognizer',
    'src.utils.plot_config',
]

a = Analysis(
    ['apps/run_gui_universal.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SpeechProcessingGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 不显示控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)
'''
    
    with open('SpeechProcessingGUI.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("✅ 已创建PyInstaller spec文件")

def build_exe():
    """构建EXE文件"""
    print("开始构建EXE文件...")
    
    # 清理之前的构建
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    
    try:
        # 运行PyInstaller
        cmd = [sys.executable, "-m", "PyInstaller", "--clean", "SpeechProcessingGUI.spec"]
        subprocess.check_call(cmd)
        print("✅ EXE文件构建成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ EXE文件构建失败: {e}")
        return False

def create_installer_script():
    """创建安装脚本"""
    installer_content = '''@echo off
echo 语音信号处理系统 - 安装程序
echo ================================

echo 正在复制文件...

if not exist "%USERPROFILE%\\Desktop\\SpeechProcessing" mkdir "%USERPROFILE%\\Desktop\\SpeechProcessing"
xcopy /E /I /Y "dist\\SpeechProcessingGUI" "%USERPROFILE%\\Desktop\\SpeechProcessing\\"

echo 安装完成！
echo 可执行文件位置: %USERPROFILE%\\Desktop\\SpeechProcessing\\SpeechProcessingGUI.exe
pause
'''
    
    with open('install.bat', 'w', encoding='gbk') as f:
        f.write(installer_content)
    
    print("✅ 已创建安装脚本 install.bat")

def main():
    """主函数"""
    print("🎤 语音信号处理系统 - EXE打包工具")
    print("=" * 50)
    
    # 检查PyInstaller
    if not check_pyinstaller():
        if not install_pyinstaller():
            print("❌ 无法安装PyInstaller，请手动安装: pip install pyinstaller")
            return False
    
    # 创建spec文件
    create_spec_file()
    
    # 创建安装脚本
    create_installer_script()
    
    # 构建EXE
    if build_exe():
        print("\n🎉 打包完成！")
        print("EXE文件位置: dist/SpeechProcessingGUI/SpeechProcessingGUI.exe")
        print("运行 install.bat 可以将程序安装到桌面")
        return True
    else:
        print("\n❌ 打包失败")
        return False

if __name__ == "__main__":
    main()


