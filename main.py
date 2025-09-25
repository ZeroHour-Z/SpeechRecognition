"""
语音信号处理系统主程序
提供交互式菜单，方便用户选择不同的功能
"""

import os
import sys
from speech_processing import WAVReader, FrameProcessor, TimeDomainAnalyzer, DualThresholdEndpointDetector


def print_menu():
    """打印主菜单"""
    print("\n" + "=" * 60)
    print("语音信号处理系统")
    print("=" * 60)
    print("1. 基础分析演示")
    print("2. 窗函数比较")
    print("3. 端点检测演示")
    print("4. 完整分析流程")
    print("5. 语音识别演示")
    print("6. 分类器对比分析")
    print("7. 运行测试")
    print("8. 查看帮助")
    print("0. 退出程序")
    print("=" * 60)


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
    """基础分析功能"""
    print("\n--- 基础分析演示 ---")
    
    wav_files = check_audio_files()
    wav_file = select_audio_file(wav_files)
    
    if not wav_file:
        return
    
    try:
        # 读取文件
        reader = WAVReader(wav_file)
        audio_data, sample_rate = reader.read()
        reader.print_info()
        
        # 分帧处理
        processor = FrameProcessor(sample_rate, 25.0, 10.0)
        frames, windowed_frames = processor.process_signal(audio_data, 'hamming')
        print(f"分帧完成，共 {len(frames)} 帧")
        
        # 时域分析
        analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
        analysis_result = analyzer.analyze_signal(audio_data, 'hamming')
        print("时域特征计算完成")
        
        # 端点检测
        detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
        endpoint_result = detector.detect_endpoints(audio_data)
        print(f"端点检测完成，检测到 {len(endpoint_result['endpoints'])} 个语音段")
        
        print("基础分析完成！")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")


def window_comparison():
    """窗函数比较功能"""
    print("\n--- 窗函数比较 ---")
    
    wav_files = check_audio_files()
    wav_file = select_audio_file(wav_files)
    
    if not wav_file:
        return
    
    try:
        # 运行窗函数比较示例
        sys.path.append("examples")
        from window_comparison import window_comparison_example
        window_comparison_example()
        
    except Exception as e:
        print(f"窗函数比较过程中出现错误: {e}")


def endpoint_detection():
    """端点检测功能"""
    print("\n--- 端点检测演示 ---")
    
    wav_files = check_audio_files()
    wav_file = select_audio_file(wav_files)
    
    if not wav_file:
        return
    
    try:
        # 运行端点检测示例
        sys.path.append("examples")
        from endpoint_detection_demo import endpoint_detection_demo
        endpoint_detection_demo()
        
    except Exception as e:
        print(f"端点检测过程中出现错误: {e}")


def complete_analysis():
    """完整分析流程"""
    print("\n--- 完整分析流程 ---")
    
    wav_files = check_audio_files()
    wav_file = select_audio_file(wav_files)
    
    if not wav_file:
        return
    
    try:
        # 运行完整分析示例
        sys.path.append("examples")
        from speech_analysis_demo import SpeechAnalysisDemo
        
        demo = SpeechAnalysisDemo()
        analysis_result = demo.run_complete_analysis(wav_file)
        demo.visualize_complete_analysis(analysis_result)
        
        # 生成报告
        report = demo.generate_analysis_report(analysis_result)
        print(report)
        
        # 保存报告
        report_file = f"data/results/{os.path.basename(wav_file)}_analysis_report.txt"
        os.makedirs("data/results", exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n分析报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"完整分析过程中出现错误: {e}")


def speech_recognition():
    """语音识别功能"""
    print("\n--- 语音识别演示 ---")
    
    try:
        # 运行语音识别演示
        sys.path.append("examples")
        from speech_recognition_demo import speech_recognition_demo
        
        speech_recognition_demo()
        
    except Exception as e:
        print(f"语音识别过程中出现错误: {e}")


def classifier_comparison():
    """分类器对比分析功能"""
    print("\n--- 分类器对比分析 ---")
    
    try:
        # 运行分类器对比演示
        sys.path.append("examples")
        from classifier_comparison_demo import classifier_comparison_demo
        
        classifier_comparison_demo()
        
    except Exception as e:
        print(f"分类器对比分析过程中出现错误: {e}")


def run_tests():
    """运行测试"""
    print("\n--- 运行测试 ---")
    
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
    print("\n--- 帮助信息 ---")
    print("本系统提供以下功能:")
    print("1. 基础分析演示 - 展示基本的语音处理流程")
    print("2. 窗函数比较 - 比较不同窗函数的特性")
    print("3. 端点检测演示 - 演示语音端点检测功能")
    print("4. 完整分析流程 - 运行完整的分析并生成报告")
    print("5. 语音识别演示 - 基于时域特征的数字识别")
    print("6. 分类器对比分析 - 多种分类器性能对比和选择")
    print("7. 运行测试 - 运行系统测试")
    print("8. 查看帮助 - 显示此帮助信息")
    print("\n使用说明:")
    print("- 将WAV文件放在 data/audio 目录下")
    print("- 系统会自动检测并列出可用的音频文件")
    print("- 选择相应的功能进行分析")
    print("- 分析结果会显示在屏幕上，并可保存到 data/results 目录")
    print("- 语音识别需要按数字分类的训练数据")


def main():
    """主函数"""
    print("欢迎使用语音信号处理系统！")
    
    while True:
        print_menu()
        
        try:
            choice = input("请选择功能 (0-8): ").strip()
            
            if choice == '0':
                print("感谢使用，再见！")
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
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"程序运行出错: {e}")
        
        input("\n按回车键继续...")


if __name__ == "__main__":
    main()
