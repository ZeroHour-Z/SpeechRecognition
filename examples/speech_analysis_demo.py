"""
语音信号处理综合演示程序
展示完整的语音信号处理流程：WAV读取 -> 分帧加窗 -> 时域分析 -> 端点检测
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from speech_processing import WAVReader, FrameProcessor, TimeDomainAnalyzer, DualThresholdEndpointDetector
from speech_processing.core.frame_window import compare_windows, visualize_framing
from speech_processing.core.time_domain_analysis import compare_window_effects, analyze_voiced_unvoiced_regions
from speech_processing.core.endpoint_detection import test_endpoint_detection_with_parameters


class SpeechAnalysisDemo:
    """语音信号处理演示类"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        初始化演示程序
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        self.frame_processor = FrameProcessor(sample_rate, 25.0, 10.0)
        self.time_analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
        self.endpoint_detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
        
    def run_complete_analysis(self, wav_file_path: str) -> dict:
        """
        运行完整的语音分析流程
        
        Args:
            wav_file_path: WAV文件路径
            
        Returns:
            dict: 分析结果
        """
        print("=" * 80)
        print("语音信号处理综合演示")
        print("=" * 80)
        
        # 1. 读取WAV文件
        print("\n1. 读取WAV文件...")
        reader = WAVReader(wav_file_path)
        audio_data, sample_rate = reader.read()
        reader.print_info()
        
        # 更新采样率
        self.sample_rate = sample_rate
        self.frame_processor = FrameProcessor(sample_rate, 25.0, 10.0)
        self.time_analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
        self.endpoint_detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
        
        # 2. 分帧和加窗处理
        print("\n2. 分帧和加窗处理...")
        frames, windowed_frames = self.frame_processor.process_signal(audio_data, 'hamming')
        print(f"分帧完成，共 {len(frames)} 帧")
        
        # 3. 时域特性分析
        print("\n3. 时域特性分析...")
        time_analysis_result = self.time_analyzer.analyze_signal(audio_data, 'hamming')
        print("时域特征计算完成")
        
        # 4. 端点检测
        print("\n4. 端点检测...")
        endpoint_result = self.endpoint_detector.detect_endpoints(audio_data, 
                                                                 energy_ratio=0.1, 
                                                                 zcr_ratio=1.5)
        print(f"端点检测完成，检测到 {len(endpoint_result['endpoints'])} 个语音段")
        
        return {
            'audio_data': audio_data,
            'sample_rate': sample_rate,
            'frames': frames,
            'windowed_frames': windowed_frames,
            'time_analysis': time_analysis_result,
            'endpoint_detection': endpoint_result
        }
    
    def visualize_complete_analysis(self, analysis_result: dict, 
                                  start_time: float = 0.0, duration: float = 3.0) -> None:
        """
        可视化完整的分析结果
        
        Args:
            analysis_result: 分析结果
            start_time: 显示开始时间
            duration: 显示时长
        """
        audio_data = analysis_result['audio_data']
        sample_rate = analysis_result['sample_rate']
        time_analysis = analysis_result['time_analysis']
        endpoint_result = analysis_result['endpoint_detection']
        
        # 计算显示范围
        start_sample = int(start_time * sample_rate)
        end_sample = int((start_time + duration) * sample_rate)
        display_signal = audio_data[start_sample:end_sample]
        display_time = np.linspace(start_time, start_time + duration, len(display_signal))
        
        # 计算显示范围内的特征
        start_frame = int(start_time * sample_rate / self.frame_processor.frame_shift)
        end_frame = int((start_time + duration) * sample_rate / self.frame_processor.frame_shift)
        
        display_energy = time_analysis['energy'][start_frame:end_frame]
        display_amplitude = time_analysis['amplitude'][start_frame:end_frame]
        display_zcr = time_analysis['zcr'][start_frame:end_frame]
        display_feature_time = time_analysis['time_axis'][start_frame:end_frame]
        
        # 创建综合显示图
        plt.figure(figsize=(16, 12))
        
        # 原始信号和端点标记
        plt.subplot(5, 1, 1)
        plt.plot(display_time, display_signal, 'b-', linewidth=1, label='Original Signal')
        
        # 标记检测到的语音段
        for endpoint in endpoint_result['endpoints']:
            if (endpoint['start_time'] >= start_time and endpoint['start_time'] <= start_time + duration) or \
               (endpoint['end_time'] >= start_time and endpoint['end_time'] <= start_time + duration):
                plt.axvspan(endpoint['start_time'], endpoint['end_time'], 
                           alpha=0.3, color='yellow', label='Detected Speech Segment')
        
        plt.title('Speech Signal Processing Comprehensive Results')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 短时能量
        plt.subplot(5, 1, 2)
        plt.plot(display_feature_time, display_energy, 'r-', linewidth=2, label='Short-time Energy')
        plt.axhline(y=endpoint_result['thresholds']['energy_high'], color='r', 
                   linestyle='--', alpha=0.7, label='High Energy Threshold')
        plt.axhline(y=endpoint_result['thresholds']['energy_low'], color='orange', 
                   linestyle='--', alpha=0.7, label='Low Energy Threshold')
        plt.title('Short-time Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 短时平均幅度
        plt.subplot(5, 1, 3)
        plt.plot(display_feature_time, display_amplitude, 'g-', linewidth=2, label='Short-time Average Magnitude')
        plt.title('Short-time Average Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Average Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 短时过零率
        plt.subplot(5, 1, 4)
        plt.plot(display_feature_time, display_zcr, 'm-', linewidth=2, label='Short-time Zero Crossing Rate')
        plt.axhline(y=endpoint_result['thresholds']['zcr_threshold'], color='m', 
                   linestyle='--', alpha=0.7, label='ZCR Threshold')
        plt.title('Short-time Zero Crossing Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Zero Crossing Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 语音段检测结果
        plt.subplot(5, 1, 5)
        display_speech = endpoint_result['speech_frames'][start_frame:end_frame]
        plt.plot(display_feature_time, display_speech.astype(int), 'k-', linewidth=2, label='Speech Segment')
        plt.fill_between(display_feature_time, 0, display_speech.astype(int), 
                        alpha=0.3, color='black')
        plt.title('Speech Segment Detection Results')
        plt.xlabel('Time (s)')
        plt.ylabel('Speech/Silence')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_comparative_analysis(self, wav_file_path: str) -> None:
        """
        运行对比分析
        
        Args:
            wav_file_path: WAV文件路径
        """
        print("\n" + "=" * 80)
        print("对比分析演示")
        print("=" * 80)
        
        # 读取文件
        reader = WAVReader(wav_file_path)
        audio_data, sample_rate = reader.read()
        
        # 1. 窗函数比较
        print("\n1. 窗函数特性比较...")
        compare_windows(256)
        
        # 2. 分帧可视化
        print("\n2. 分帧过程可视化...")
        visualize_framing(audio_data, sample_rate, 25.0, 10.0, 0.0, 0.5)
        
        # 3. 不同窗函数的时域特征对比
        print("\n3. 不同窗函数的时域特征对比...")
        compare_window_effects(audio_data, sample_rate)
        
        # 4. 浊音清音分析
        print("\n4. 浊音清音区域分析...")
        analyze_voiced_unvoiced_regions(audio_data, sample_rate)
        
        # 5. 不同参数的端点检测对比
        print("\n5. 不同参数的端点检测对比...")
        test_endpoint_detection_with_parameters(audio_data, sample_rate)
    
    def generate_analysis_report(self, analysis_result: dict) -> str:
        """
        生成分析报告
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            str: 分析报告
        """
        audio_data = analysis_result['audio_data']
        sample_rate = analysis_result['sample_rate']
        time_analysis = analysis_result['time_analysis']
        endpoint_result = analysis_result['endpoint_detection']
        
        report = []
        report.append("=" * 80)
        report.append("语音信号处理分析报告")
        report.append("=" * 80)
        
        # 基本信息
        report.append(f"\n【基本信息】")
        report.append(f"采样率: {sample_rate} Hz")
        report.append(f"信号长度: {len(audio_data)} 采样点")
        report.append(f"信号时长: {len(audio_data) / sample_rate:.3f} 秒")
        report.append(f"信号幅度范围: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
        report.append(f"信号均值: {audio_data.mean():.6f}")
        report.append(f"信号标准差: {audio_data.std():.6f}")
        
        # 分帧信息
        report.append(f"\n【分帧信息】")
        report.append(f"帧长: {self.frame_processor.frame_length} 采样点 ({self.frame_processor.frame_length_ms} ms)")
        report.append(f"帧移: {self.frame_processor.frame_shift} 采样点 ({self.frame_processor.frame_shift_ms} ms)")
        report.append(f"总帧数: {time_analysis['num_frames']}")
        report.append(f"重叠率: {(self.frame_processor.frame_length - self.frame_processor.frame_shift) / self.frame_processor.frame_length * 100:.1f}%")
        
        # 时域特征统计
        report.append(f"\n【时域特征统计】")
        report.append(f"短时能量:")
        report.append(f"  最大值: {time_analysis['energy'].max():.6f}")
        report.append(f"  最小值: {time_analysis['energy'].min():.6f}")
        report.append(f"  均值: {time_analysis['energy'].mean():.6f}")
        report.append(f"  标准差: {time_analysis['energy'].std():.6f}")
        
        report.append(f"短时平均幅度:")
        report.append(f"  最大值: {time_analysis['amplitude'].max():.6f}")
        report.append(f"  最小值: {time_analysis['amplitude'].min():.6f}")
        report.append(f"  均值: {time_analysis['amplitude'].mean():.6f}")
        report.append(f"  标准差: {time_analysis['amplitude'].std():.6f}")
        
        report.append(f"短时过零率:")
        report.append(f"  最大值: {time_analysis['zcr'].max():.4f}")
        report.append(f"  最小值: {time_analysis['zcr'].min():.4f}")
        report.append(f"  均值: {time_analysis['zcr'].mean():.4f}")
        report.append(f"  标准差: {time_analysis['zcr'].std():.4f}")
        
        # 端点检测结果
        report.append(f"\n【端点检测结果】")
        report.append(f"检测到的语音段数量: {len(endpoint_result['endpoints'])}")
        report.append(f"能量高阈值: {endpoint_result['thresholds']['energy_high']:.6f}")
        report.append(f"能量低阈值: {endpoint_result['thresholds']['energy_low']:.6f}")
        report.append(f"过零率阈值: {endpoint_result['thresholds']['zcr_threshold']:.4f}")
        
        total_speech_duration = sum(endpoint['duration'] for endpoint in endpoint_result['endpoints'])
        report.append(f"总语音时长: {total_speech_duration:.3f} 秒")
        report.append(f"语音占比: {total_speech_duration / (len(audio_data) / sample_rate) * 100:.1f}%")
        
        report.append(f"\n各语音段详情:")
        for i, endpoint in enumerate(endpoint_result['endpoints']):
            report.append(f"  语音段 {i+1}:")
            report.append(f"    开始时间: {endpoint['start_time']:.3f} 秒")
            report.append(f"    结束时间: {endpoint['end_time']:.3f} 秒")
            report.append(f"    持续时间: {endpoint['duration']:.3f} 秒")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """主函数"""
    print("语音信号处理综合演示程序")
    print("支持的功能:")
    print("1. WAV文件读取和格式解析")
    print("2. 语音分帧和加窗处理（矩形窗、汉明窗、海宁窗）")
    print("3. 短时时域特性分析（短时能量、短时平均幅度、短时过零率）")
    print("4. 基于双门限法的端点检测")
    print("5. 综合分析和可视化")
    
    # 查找WAV文件
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if not wav_files:
        print("\n当前目录下没有找到WAV文件！")
        print("请将WAV文件放在当前目录下，然后重新运行程序。")
        return
    
    # 选择文件
    if len(wav_files) == 1:
        selected_file = wav_files[0]
        print(f"\n找到WAV文件: {selected_file}")
    else:
        print(f"\n找到多个WAV文件:")
        for i, file in enumerate(wav_files):
            print(f"{i+1}. {file}")
        
        try:
            choice = int(input("请选择要分析的文件编号: ")) - 1
            if 0 <= choice < len(wav_files):
                selected_file = wav_files[choice]
            else:
                print("无效选择，使用第一个文件")
                selected_file = wav_files[0]
        except ValueError:
            print("无效输入，使用第一个文件")
            selected_file = wav_files[0]
    
    print(f"\n开始分析文件: {selected_file}")
    
    # 创建演示程序
    demo = SpeechAnalysisDemo()
    
    # 运行完整分析
    analysis_result = demo.run_complete_analysis(selected_file)
    
    # 可视化结果
    print("\n生成可视化结果...")
    demo.visualize_complete_analysis(analysis_result, 0.0, 3.0)
    
    # 生成分析报告
    print("\n生成分析报告...")
    report = demo.generate_analysis_report(analysis_result)
    print(report)
    
    # 保存报告到文件
    report_file = f"{selected_file}_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n分析报告已保存到: {report_file}")
    
    # 询问是否运行对比分析
    try:
        run_comparison = input("\n是否运行对比分析？(y/n): ").lower().strip()
        if run_comparison in ['y', 'yes', '是']:
            demo.run_comparative_analysis(selected_file)
    except KeyboardInterrupt:
        print("\n程序结束")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
