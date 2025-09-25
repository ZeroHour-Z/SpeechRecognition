"""
Window Function Comparison Example (English Version)
Demonstrates the characteristics and effects of different window functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_processing import WAVReader, FrameProcessor, WindowFunctions
import matplotlib.pyplot as plt
import numpy as np


def window_comparison_english():
    """Window function comparison example in English"""
    print("=" * 60)
    print("Window Function Comparison Example")
    print("=" * 60)
    
    # Find audio files
    audio_dir = "../data/audio"
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')] if os.path.exists(audio_dir) else []
    
    if not wav_files:
        print("No WAV files found in data/audio directory")
        print("Please place WAV files in data/audio directory")
        return
    
    # Use the first WAV file
    wav_file = os.path.join(audio_dir, wav_files[0])
    print(f"Analyzing file: {wav_file}")
    
    # Read audio file
    reader = WAVReader(wav_file)
    audio_data, sample_rate = reader.read()
    
    # Create frame processor
    processor = FrameProcessor(sample_rate, 25.0, 10.0)
    
    # Compare three window functions
    window_types = ['rectangular', 'hamming', 'hanning']
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(15, 10))
    
    # 1. Window function time domain characteristics
    plt.subplot(2, 2, 1)
    frame_length = 256
    n = np.arange(frame_length)
    
    rect_window = WindowFunctions.rectangular_window(frame_length)
    hamming_window = WindowFunctions.hamming_window(frame_length)
    hanning_window = WindowFunctions.hanning_window(frame_length)
    
    plt.plot(n, rect_window, 'b-', label='Rectangular Window', linewidth=2)
    plt.plot(n, hamming_window, 'r-', label='Hamming Window', linewidth=2)
    plt.plot(n, hanning_window, 'g-', label='Hanning Window', linewidth=2)
    plt.title('Window Function Time Domain Characteristics')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 2. Window function frequency domain characteristics
    plt.subplot(2, 2, 2)
    freqs = np.fft.fftfreq(frame_length)
    rect_fft = np.abs(np.fft.fft(rect_window))
    hamming_fft = np.abs(np.fft.fft(hamming_window))
    hanning_fft = np.abs(np.fft.fft(hanning_window))
    
    plt.plot(freqs[:frame_length//2], rect_fft[:frame_length//2], 'b-', label='Rectangular Window', linewidth=2)
    plt.plot(freqs[:frame_length//2], hamming_fft[:frame_length//2], 'r-', label='Hamming Window', linewidth=2)
    plt.plot(freqs[:frame_length//2], hanning_fft[:frame_length//2], 'g-', label='Hanning Window', linewidth=2)
    plt.title('Window Function Frequency Domain Characteristics')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 3. Effect of different window functions on speech signal
    plt.subplot(2, 2, 3)
    # Take a segment of speech signal
    start_sample = len(audio_data) // 4
    end_sample = start_sample + frame_length
    test_signal = audio_data[start_sample:end_sample]
    
    rect_result = test_signal * rect_window
    hamming_result = test_signal * hamming_window
    hanning_result = test_signal * hanning_window
    
    time_axis = np.arange(len(test_signal)) / sample_rate
    
    plt.plot(time_axis, test_signal, 'k-', label='Original Signal', alpha=0.7)
    plt.plot(time_axis, rect_result, 'b-', label='Rectangular Window', linewidth=2)
    plt.plot(time_axis, hamming_result, 'r-', label='Hamming Window', linewidth=2)
    plt.plot(time_axis, hanning_result, 'g-', label='Hanning Window', linewidth=2)
    plt.title('Window Function Application Effects')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 4. Window function parameter comparison table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    table_data = [
        ['Window Type', 'Main Lobe Width', 'Side Lobe Level (dB)', 'Application'],
        ['Rectangular', '0.89', '-13.3', 'Time Domain Analysis'],
        ['Hamming', '1.30', '-42.7', 'Speech Analysis'],
        ['Hanning', '1.44', '-31.5', 'Spectral Analysis']
    ]
    
    table = plt.table(cellText=table_data,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Window Function Characteristics Comparison')
    
    plt.tight_layout()
    plt.show()
    
    # Print window function characteristics
    print("\nWindow Function Characteristics Comparison:")
    print("=" * 60)
    print(f"{'Window Type':<15} {'Main Lobe Width':<15} {'Side Lobe Level (dB)':<20} {'Application'}")
    print("-" * 60)
    print(f"{'Rectangular':<15} {'0.89':<15} {'-13.3':<20} {'Time Domain Analysis'}")
    print(f"{'Hamming':<15} {'1.30':<15} {'-42.7':<20} {'Speech Analysis'}")
    print(f"{'Hanning':<15} {'1.44':<15} {'-31.5':<20} {'Spectral Analysis'}")
    print("=" * 60)


if __name__ == "__main__":
    window_comparison_english()
