#!/usr/bin/env python3
"""
语音信号处理系统 - 跨平台启动器
支持Windows、Linux、macOS
"""

import sys
import os
import platform

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def detect_platform():
    """检测操作系统平台"""
    system = platform.system().lower()
    return system

def setup_platform_environment():
    """设置平台特定的环境"""
    system = detect_platform()
    
    if system == "windows":
        # Windows特定设置
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        print("✅ Windows环境配置完成")
        
    elif system == "linux":
        # Linux特定设置
        print("✅ Linux环境配置完成")
        
    elif system == "darwin":
        # macOS特定设置
        print("✅ macOS环境配置完成")
        
    else:
        print(f"⚠️ 未识别的操作系统: {system}")

def check_requirements():
    """检查系统要求"""
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本: {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_dependencies():
    """检查依赖包"""
    required_packages = {
        'PyQt5': 'PyQt5',
        'numpy': 'numpy', 
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'pyaudio': 'pyaudio'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} 已安装")
        except ImportError:
            print(f"❌ {package_name} 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """主函数"""
    print("🎤 语音信号处理系统 - 跨平台版本")
    print("=" * 50)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print("=" * 50)
    
    # 检查系统要求
    if not check_requirements():
        input("按回车键退出...")
        sys.exit(1)
    
    # 设置平台环境
    setup_platform_environment()
    
    # 检查依赖包
    if not check_dependencies():
        input("按回车键退出...")
        sys.exit(1)
    
    try:
        # 导入Qt界面
        from qt_interface import main as qt_main
        print("\n✅ 正在启动Qt界面...")
        qt_main()
    except ImportError as e:
        print(f"❌ 无法启动Qt界面: {e}")
        print("请安装依赖: pip install PyQt5 pyaudio")
        input("按回车键退出...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        input("按回车键退出...")
        sys.exit(1)

if __name__ == "__main__":
    main()
