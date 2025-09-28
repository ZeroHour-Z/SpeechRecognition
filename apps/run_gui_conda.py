#!/home/zerohour/.conda/envs/dsp1/bin/python3
"""
语音信号处理系统 - Conda环境启动器
使用conda环境中的Python和依赖包
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
    
    if system == "linux":
        print("✅ Linux环境配置完成")
        return True
    elif system == "windows":
        print("✅ Windows环境配置完成")
        return True
    elif system == "darwin":
        print("✅ macOS环境配置完成")
        return True
    else:
        print(f"❌ 不支持的操作系统: {system}")
        return False

def check_python_version():
    """检查Python版本"""
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
        'pyaudio': 'pyaudio',
        'sklearn': 'sklearn',
        'torch': 'torch'
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
        print("conda install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """主函数"""
    print("🎤 语音信号处理系统 - Conda版本")
    print("=" * 50)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        input("按回车键退出...")
        return
    
    # 设置平台环境
    if not setup_platform_environment():
        input("按回车键退出...")
        return
    
    # 检查依赖包
    if not check_dependencies():
        input("按回车键退出...")
        return
    
    # 检查sklearn
    try:
        import sklearn
        print("✅ sklearn 已安装")
    except ImportError:
        print("警告: sklearn未安装，将使用简化版本的分类器")
    
    print("\n✅ 正在启动Qt界面...")
    
    try:
        # 导入并启动Qt界面
        from qt_interface import SpeechProcessingGUI
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        window = SpeechProcessingGUI()
        window.show()
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()
