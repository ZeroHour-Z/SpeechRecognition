"""
语音信号处理系统主程序
提供交互式菜单，方便用户选择不同的功能
"""

import os
import sys
from speech_processing import WAVReader, FrameProcessor, TimeDomainAnalyzer, DualThresholdEndpointDetector


def print_menu():
    """打印主菜单"""
    print("\n" + "=" * 80)
    print("🎤 语音信号处理系统 | Speech Signal Processing System")
    print("=" * 80)
    print("1️⃣  基础分析演示      - Basic Analysis Demo (examples/basic_analysis_demo.py)")
    print("2️⃣  窗函数比较演示    - Window Function Comparison Demo (examples/window_comparison_demo.py)")
    print("3️⃣  端点检测演示      - Endpoint Detection Demo (examples/endpoint_detection_demo.py)")
    print("4️⃣  完整分析流程演示  - Complete Analysis Pipeline Demo (examples/speech_analysis_demo.py)")
    print("5️⃣  语音识别演示      - Speech Recognition Demo (examples/speech_recognition_demo.py)")
    print("6️⃣  分类器对比演示    - Classifier Comparison Demo (examples/classifier_comparison_demo.py)")
    print("7️⃣  运行测试          - Run Tests (tests/)")
    print("8️⃣  查看帮助          - Show Help")
    print("0️⃣  退出程序          - Exit Program")
    print("=" * 80)


def check_audio_files():
    """检查音频文件"""
    audio_dir = "data/audio"
    if not os.path.exists(audio_dir):
        print(f"音频目录 {audio_dir} 不存在，正在创建...")
        os.makedirs(audio_dir, exist_ok=True)
        return []
    
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    return wav_files


def select_audio_file(wav_files):
    """选择音频文件"""
    if not wav_files:
        print("在 data/audio 目录下没有找到WAV文件")
        print("请将WAV文件放在 data/audio 目录下")
        return None
    
    if len(wav_files) == 1:
        return os.path.join("data/audio", wav_files[0])
    
    print("找到多个WAV文件:")
    for i, file in enumerate(wav_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("请选择要分析的文件编号: ")) - 1
        if 0 <= choice < len(wav_files):
            return os.path.join("data/audio", wav_files[choice])
        else:
            print("无效选择，使用第一个文件")
            return os.path.join("data/audio", wav_files[0])
    except ValueError:
        print("无效输入，使用第一个文件")
        return os.path.join("data/audio", wav_files[0])


def basic_analysis():
    """基础分析功能 - 调用examples/basic_analysis_demo.py"""
    print("\n--- 基础分析演示 ---")
    print("正在运行基础分析示例程序...")
    
    try:
        # 添加examples目录到路径
        sys.path.append("examples")
        
        # 导入并运行基础分析示例
        from basic_analysis_demo import basic_analysis_example
        basic_analysis_example()
        
    except ImportError as e:
        print(f"无法导入基础分析示例: {e}")
        print("请确保examples/basic_analysis_demo.py文件存在")
    except Exception as e:
        print(f"运行基础分析示例时出现错误: {e}")


def window_comparison():
    """窗函数比较功能 - 调用examples/window_comparison_demo.py"""
    print("\n--- 窗函数比较演示 ---")
    print("正在运行窗函数比较示例程序...")
    
    try:
        # 运行窗函数比较示例
        sys.path.append("examples")
        from window_comparison_demo import window_comparison_example
        window_comparison_example()
        
    except ImportError as e:
        print(f"无法导入窗函数比较示例: {e}")
        print("请确保examples/window_comparison_demo.py文件存在")
    except Exception as e:
        print(f"窗函数比较过程中出现错误: {e}")


def endpoint_detection():
    """端点检测功能 - 调用examples/endpoint_detection_demo.py"""
    print("\n--- 端点检测演示 ---")
    print("正在运行端点检测示例程序...")
    
    try:
        # 运行端点检测示例
        sys.path.append("examples")
        from endpoint_detection_demo import endpoint_detection_demo
        endpoint_detection_demo()
        
    except ImportError as e:
        print(f"无法导入端点检测示例: {e}")
        print("请确保examples/endpoint_detection_demo.py文件存在")
    except Exception as e:
        print(f"端点检测过程中出现错误: {e}")


def complete_analysis():
    """完整分析流程 - 调用examples/speech_analysis_demo.py"""
    print("\n--- 完整分析流程演示 ---")
    print("正在运行完整分析示例程序...")
    
    try:
        # 运行完整分析示例
        sys.path.append("examples")
        from speech_analysis_demo import SpeechAnalysisDemo
        
        demo = SpeechAnalysisDemo()
        analysis_result = demo.run_complete_analysis()
        demo.visualize_complete_analysis(analysis_result)
        
        # 生成报告
        report = demo.generate_analysis_report(analysis_result)
        print(report)
        
        # 保存报告
        report_file = f"data/results/complete_analysis_report.txt"
        os.makedirs("data/results", exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n分析报告已保存到: {report_file}")
        
    except ImportError as e:
        print(f"无法导入完整分析示例: {e}")
        print("请确保examples/speech_analysis_demo.py文件存在")
    except Exception as e:
        print(f"完整分析过程中出现错误: {e}")


def speech_recognition():
    """语音识别功能 - 调用examples/speech_recognition_demo.py"""
    print("\n--- 语音识别演示 ---")
    print("正在运行语音识别示例程序...")
    
    try:
        # 运行语音识别演示
        sys.path.append("examples")
        from speech_recognition_demo import speech_recognition_demo
        
        speech_recognition_demo()
        
    except ImportError as e:
        print(f"无法导入语音识别示例: {e}")
        print("请确保examples/speech_recognition_demo.py文件存在")
    except Exception as e:
        print(f"语音识别过程中出现错误: {e}")


def classifier_comparison():
    """分类器对比分析功能 - 调用examples/classifier_comparison_demo.py"""
    print("\n--- 分类器对比演示 ---")
    print("正在运行分类器对比示例程序...")
    
    try:
        # 运行分类器对比演示
        sys.path.append("examples")
        from classifier_comparison_demo import classifier_comparison_demo
        
        classifier_comparison_demo()
        
    except ImportError as e:
        print(f"无法导入分类器对比示例: {e}")
        print("请确保examples/classifier_comparison_demo.py文件存在")
    except Exception as e:
        print(f"分类器对比分析过程中出现错误: {e}")


def run_tests():
    """运行测试 - 调用tests/目录下的测试文件"""
    print("\n--- 运行测试 ---")
    print("正在运行系统测试...")
    
    try:
        # 运行WAV读取测试
        sys.path.append("tests")
        from test_wav_reader import test_wav_reader
        from test_frame_processor import test_frame_processor, test_window_functions
        
        print("运行WAV读取测试...")
        test_wav_reader()
        
        print("\n运行分帧处理测试...")
        test_frame_processor()
        test_window_functions()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")


def show_help():
    """显示帮助信息"""
    print("\n--- 帮助信息 | Help Information ---")
    print("本系统提供以下功能 | This system provides the following features:")
    print("1️⃣  基础分析演示      - Basic Analysis Demo (examples/basic_analysis_demo.py)")
    print("2️⃣  窗函数比较演示    - Window Function Comparison Demo (examples/window_comparison_demo.py)")
    print("3️⃣  端点检测演示      - Endpoint Detection Demo (examples/endpoint_detection_demo.py)")
    print("4️⃣  完整分析流程演示  - Complete Analysis Pipeline Demo (examples/speech_analysis_demo.py)")
    print("5️⃣  语音识别演示      - Speech Recognition Demo (examples/speech_recognition_demo.py)")
    print("6️⃣  分类器对比演示    - Classifier Comparison Demo (examples/classifier_comparison_demo.py)")
    print("7️⃣  运行测试          - Run Tests (tests/)")
    print("8️⃣  查看帮助          - Show Help")
    print("0️⃣  退出程序          - Exit Program")
    print("\n使用说明 | Usage Instructions:")
    print("- 将WAV文件放在 data/audio 目录下 | Place WAV files in data/audio directory")
    print("- 系统会自动检测并列出可用的音频文件 | System will auto-detect and list available audio files")
    print("- 选择相应的功能进行分析 | Select corresponding function for analysis")
    print("- 分析结果会显示在屏幕上，并可保存到 data/results 目录 | Results displayed on screen and saved to data/results")
    print("- 语音识别需要按数字分类的训练数据 | Speech recognition requires training data organized by digits")


def main():
    """主函数"""
    print("🎤 欢迎使用语音信号处理系统！| Welcome to Speech Signal Processing System!")
    
    while True:
        print_menu()
        
        try:
            choice = input("请选择功能 (0-8) | Please select function (0-8): ").strip()
            
            if choice == '0':
                print("👋 感谢使用，再见！| Thank you for using, goodbye!")
                break
            elif choice == '1':
                basic_analysis()
            elif choice == '2':
                window_comparison()
            elif choice == '3':
                endpoint_detection()
            elif choice == '4':
                complete_analysis()
            elif choice == '5':
                speech_recognition()
            elif choice == '6':
                classifier_comparison()
            elif choice == '7':
                run_tests()
            elif choice == '8':
                show_help()
            else:
                print("❌ 无效选择，请重新输入 | Invalid choice, please try again")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 程序被用户中断 | Program interrupted by user")
            break
        except Exception as e:
            print(f"❌ 程序运行出错 | Program error: {e}")
        
        input("\n按回车键继续... | Press Enter to continue...")


if __name__ == "__main__":
    main()
