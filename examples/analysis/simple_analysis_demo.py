"""
Simple Speech Processing Demo (English Labels)
A simple demonstration without Chinese font issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import WAVReader, FrameProcessor, TimeDomainAnalyzer, DualThresholdEndpointDetector
import matplotlib.pyplot as plt
import numpy as np


def simple_demo():
    """Simple speech processing demonstration"""
    print("=" * 60)
    print("Simple Speech Processing Demo")
    print("=" * 60)
    
    # Find audio files
    audio_dir = "../data/audio"
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')] if os.path.exists(audio_dir) else []
    
    if not wav_files:
        print("No WAV files found in data/audio directory")
        return
    
    # Use the first WAV file
    wav_file = os.path.join(audio_dir, wav_files[0])
    print(f"Analyzing file: {wav_file}")
    
    # Read audio file
    reader = WAVReader(wav_file)
    audio_data, sample_rate = reader.read()
    reader.print_info()
    
    # Frame processing
    processor = FrameProcessor(sample_rate, 25.0, 10.0)
    frames, windowed_frames = processor.process_signal(audio_data, 'hamming')
    print(f"Framing completed: {len(frames)} frames")
    
    # Time domain analysis
    analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
    analysis_result = analyzer.analyze_signal(audio_data, 'hamming')
    print("Time domain analysis completed")
    
    # Endpoint detection
    detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
    endpoint_result = detector.detect_endpoints(audio_data)
    print(f"Endpoint detection completed: {len(endpoint_result['endpoints'])} speech segments detected")
    
    # Simple visualization
    plot_simple_analysis(audio_data, sample_rate, analysis_result, endpoint_result)
    
    print("Demo completed!")


def plot_simple_analysis(audio_data, sample_rate, analysis_result, endpoint_result):
    """Plot simple analysis results with English labels"""
    plt.figure(figsize=(12, 8))
    
    # Original signal
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    plt.plot(time_axis, audio_data, 'b-', linewidth=1)
    plt.title('Original Speech Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Short-time energy
    plt.subplot(3, 1, 2)
    plt.plot(analysis_result['time_axis'], analysis_result['energy'], 'r-', linewidth=2)
    plt.title('Short-time Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.grid(True, alpha=0.3)
    
    # Endpoint detection result
    plt.subplot(3, 1, 3)
    plt.plot(analysis_result['time_axis'], endpoint_result['speech_frames'].astype(int), 'g-', linewidth=2)
    plt.fill_between(analysis_result['time_axis'], 0, endpoint_result['speech_frames'].astype(int), 
                    alpha=0.3, color='green')
    plt.title('Endpoint Detection Result')
    plt.xlabel('Time (s)')
    plt.ylabel('Speech/Silence')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simple_demo()
