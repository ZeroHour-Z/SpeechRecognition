"""
Speech Signal Processing System - Qt GUI
Provides complete graphical operation interface
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

# 导入字体配置
from src.utils.plot_config import initialize_plotting

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QTextEdit, QFileDialog, QComboBox, QSlider, 
                            QProgressBar, QTabWidget, QGroupBox, QSpinBox,
                            QDoubleSpinBox, QCheckBox, QMessageBox, QSplitter)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon

# 导入语音处理模块
from src import (WAVReader, FrameProcessor, TimeDomainAnalyzer, 
                             DualThresholdEndpointDetector, SimpleDigitRecognizer)
from src.core.audio_recorder import AudioRecorder, RealTimeAnalyzer


class AudioAnalysisWorker(QThread):
    """Audio analysis worker thread"""
    analysis_complete = pyqtSignal(dict)
    progress_update = pyqtSignal(int)
    
    def __init__(self, audio_data, sample_rate):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        
    def run(self):
        """Execute audio analysis"""
        try:
            # Create analyzers
            analyzer = TimeDomainAnalyzer(self.sample_rate, 25.0, 10.0)
            endpoint_detector = DualThresholdEndpointDetector(self.sample_rate, 25.0, 10.0)
            
            # Execute analysis
            self.progress_update.emit(20)
            analysis_result = analyzer.analyze_signal(self.audio_data)
            
            self.progress_update.emit(40)
            endpoint_result = endpoint_detector.detect_endpoints(self.audio_data)
            
            self.progress_update.emit(60)
            # Extract speech segments
            if endpoint_result['endpoints']:
                speech_segments = endpoint_detector.extract_speech_segments(
                    self.audio_data, endpoint_result)
            else:
                speech_segments = []
            
            self.progress_update.emit(80)
            
            # Prepare results
            result = {
                'analysis': analysis_result,
                'endpoints': endpoint_result,
                'speech_segments': speech_segments,
                'sample_rate': self.sample_rate
            }
            
            self.progress_update.emit(100)
            self.analysis_complete.emit(result)
            
        except Exception as e:
            print(f"Error during analysis: {e}")


class SpeechRecognitionWorker(QThread):
    """Speech recognition worker thread"""
    recognition_complete = pyqtSignal(dict)
    progress_update = pyqtSignal(int)
    
    def __init__(self, audio_data, sample_rate, recognizer):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.recognizer = recognizer
        
    def run(self):
        """Execute speech recognition"""
        try:
            self.progress_update.emit(20)
            
            # Execute recognition
            result = self.recognizer.recognize(self.audio_data)
            
            self.progress_update.emit(100)
            self.recognition_complete.emit(result)
            
        except Exception as e:
            print(f"Error during recognition: {e}")


class MatplotlibWidget(QWidget):
    """Matplotlib graphics display component"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def plot_audio_signal(self, audio_data, sample_rate, title="Audio Signal"):
        """Plot audio signal"""
        self.figure.clear()
        
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        ax = self.figure.add_subplot(111)
        ax.plot(time_axis, audio_data)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.grid(True)
        
        self.canvas.draw()
    
    def plot_analysis_results(self, analysis_result, sample_rate):
        """Plot analysis results"""
        self.figure.clear()
        
        # Create subplots
        gs = self.figure.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Short-time energy
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax1.plot(analysis_result['short_time_energy'])
        ax1.set_title('Short-time Energy', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Energy', fontsize=9)
        ax1.grid(True)
        
        # Average amplitude
        ax2 = self.figure.add_subplot(gs[0, 1])
        ax2.plot(analysis_result['average_amplitude'])
        ax2.set_title('Average Amplitude', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Amplitude', fontsize=9)
        ax2.grid(True)
        
        # Zero crossing rate
        ax3 = self.figure.add_subplot(gs[1, :])
        ax3.plot(analysis_result['zero_crossing_rate'])
        ax3.set_title('Zero Crossing Rate', fontsize=10, fontweight='bold')
        ax3.set_ylabel('ZCR', fontsize=9)
        ax3.set_xlabel('Frame Number', fontsize=9)
        ax3.grid(True)
        
        # Endpoint detection results
        ax4 = self.figure.add_subplot(gs[2, :])
        if analysis_result.get('endpoints'):
            endpoints = analysis_result['endpoints']
            ax4.plot(analysis_result['short_time_energy'], label='Short-time Energy')
            for start, end in endpoints:
                ax4.axvspan(start, end, alpha=0.3, color='red', label='Speech Segment')
            ax4.set_title('Endpoint Detection Results', fontsize=10, fontweight='bold')
            ax4.set_xlabel('Frame Number', fontsize=9)
            ax4.set_ylabel('Energy', fontsize=9)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No speech segments detected', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Endpoint Detection Results', fontsize=10, fontweight='bold')
        
        ax4.grid(True)
        
        self.canvas.draw()


class SpeechProcessingGUI(QMainWindow):
    """Speech Processing System Main Interface"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize matplotlib font configuration
        self.setup_matplotlib_fonts()
        
        self.audio_data = None
        self.sample_rate = 16000
        self.recorder = AudioRecorder()
        self.real_time_analyzer = RealTimeAnalyzer()
        self.recognizer = SimpleDigitRecognizer()
        
        self.init_ui()
        self.setup_connections()
    
    def setup_matplotlib_fonts(self):
        """Setup matplotlib fonts"""
        try:
            # Set font parameters
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # Test if font is available
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.set_title('Test')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            plt.close(fig)
            
        except Exception as e:
            print(f"Font setup warning: {e}")
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Speech Signal Processing System - Speech Signal Processing System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left control panel
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Right display area
        display_area = self.create_display_area()
        splitter.addWidget(display_area)
        
        # Set splitter ratio
        splitter.setSizes([400, 1000])
        
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File Operations
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        self.load_button = QPushButton("Load audio file")
        self.load_button.clicked.connect(self.load_audio_file)
        file_layout.addWidget(self.load_button)
        
        self.save_button = QPushButton("Save Recording")
        self.save_button.clicked.connect(self.save_recording)
        file_layout.addWidget(self.save_button)
        
        layout.addWidget(file_group)
        
        # Recording Control
        record_group = QGroupBox("Recording Control")
        record_layout = QVBoxLayout(record_group)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        record_layout.addWidget(self.record_button)
        
        self.record_status = QLabel("Status: Not recording")
        record_layout.addWidget(self.record_status)
        
        # Recording Parameters
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("Sample Rate:"), 0, 0)
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setValue(16000)
        param_layout.addWidget(self.sample_rate_spin, 0, 1)
        
        param_layout.addWidget(QLabel("Recording Duration:"), 1, 0)
        self.record_duration_spin = QSpinBox()
        self.record_duration_spin.setRange(1, 60)
        self.record_duration_spin.setValue(5)
        param_layout.addWidget(self.record_duration_spin, 1, 1)
        
        record_layout.addLayout(param_layout)
        layout.addWidget(record_group)
        
        # Signal Analysis
        analysis_group = QGroupBox("信号分析")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analyze_button = QPushButton("Execute Analysis")
        self.analyze_button.clicked.connect(self.analyze_audio)
        analysis_layout.addWidget(self.analyze_button)
        
        self.progress_bar = QProgressBar()
        analysis_layout.addWidget(self.progress_bar)
        
        # Analysis Parameters
        analysis_param_layout = QGridLayout()
        analysis_param_layout.addWidget(QLabel("Frame Length(ms):"), 0, 0)
        self.frame_length_spin = QDoubleSpinBox()
        self.frame_length_spin.setRange(10.0, 50.0)
        self.frame_length_spin.setValue(25.0)
        analysis_param_layout.addWidget(self.frame_length_spin, 0, 1)
        
        analysis_param_layout.addWidget(QLabel("Frame Shift(ms):"), 1, 0)
        self.frame_shift_spin = QDoubleSpinBox()
        self.frame_shift_spin.setRange(5.0, 25.0)
        self.frame_shift_spin.setValue(10.0)
        analysis_param_layout.addWidget(self.frame_shift_spin, 1, 1)
        
        analysis_layout.addLayout(analysis_param_layout)
        layout.addWidget(analysis_group)
        
        # Speech Recognition
        recognition_group = QGroupBox("语音识别")
        recognition_layout = QVBoxLayout(recognition_group)
        
        self.recognize_button = QPushButton("Start Recognition")
        self.recognize_button.clicked.connect(self.recognize_speech)
        recognition_layout.addWidget(self.recognize_button)
        
        self.recognition_result = QLabel("Recognition Result: Not recognized")
        recognition_layout.addWidget(self.recognition_result)
        
        # 分类器选择
        classifier_layout = QHBoxLayout()
        classifier_layout.addWidget(QLabel("Classifier:"))
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItems(["Simple Recognizer", "Advanced Recognizer"])
        classifier_layout.addWidget(self.classifier_combo)
        recognition_layout.addLayout(classifier_layout)
        
        layout.addWidget(recognition_group)
        
        # Real-time Analysis
        realtime_group = QGroupBox("实时分析")
        realtime_layout = QVBoxLayout(realtime_group)
        
        self.realtime_checkbox = QCheckBox("Enable Real-time Analysis")
        realtime_layout.addWidget(self.realtime_checkbox)
        
        self.realtime_info = QLabel("Real-time Info: Disabled")
        realtime_layout.addWidget(self.realtime_info)
        
        layout.addWidget(realtime_group)
        
        # Add elastic space
        layout.addStretch()
        
        return panel
    
    def create_display_area(self):
        """Create display area"""
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Signal Display
        self.signal_widget = MatplotlibWidget()
        tab_widget.addTab(self.signal_widget, "信号显示")
        
        # Analysis Results
        self.analysis_widget = MatplotlibWidget()
        tab_widget.addTab(self.analysis_widget, "分析结果")
        
        # Log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        clear_log_button = QPushButton("Clear log")
        clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_button)
        
        tab_widget.addTab(log_widget, "Log")
        
        return tab_widget
    
    def setup_connections(self):
        """Setup signal connections"""
        # Create timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_realtime_info)
        
    def log_message(self, message):
        """添加Log消息"""
        self.log_text.append(f"[{self.get_current_time()}] {message}")
        
    def get_current_time(self):
        """Get current time string"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def clear_log(self):
        """Clear log"""
        self.log_text.clear()
    
    def load_audio_file(self):
        """Load audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "data/audio", "WAV Files (*.wav)")
        
        if file_path:
            try:
                reader = WAVReader(file_path)
                self.audio_data, self.sample_rate = reader.read()
                
                # 更新界面
                self.signal_widget.plot_audio_signal(
                    self.audio_data, self.sample_rate, f"Loaded Audio: {os.path.basename(file_path)}")
                
                self.log_message(f"Successfully loaded audio file: {file_path}")
                self.log_message(f"Sample rate: {self.sample_rate} Hz, Duration: {len(self.audio_data)/self.sample_rate:.2f} seconds")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load audio file: {e}")
                self.log_message(f"Failed to load audio file: {e}")
    
    def save_recording(self):
        """Save Recording"""
        if self.audio_data is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Recording", "data/audio/recording.wav", "WAV Files (*.wav)")
            
            if file_path:
                try:
                    # 使用录音器保存
                    self.recorder.save_recording(self.audio_data, file_path)
                    self.log_message(f"Recording saved to: {file_path}")
                    QMessageBox.information(self, "成功", "Recording saved successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"Save Recording失败: {e}")
                    self.log_message(f"Save Recording失败: {e}")
        else:
            QMessageBox.warning(self, "警告", "No recording data to save")
    
    def toggle_recording(self):
        """Toggle recording status"""
        if not self.recorder.is_recording:
            # Start Recording
            self.start_recording()
        else:
            # Stop recording
            self.stop_recording()
    
    def start_recording(self):
        """Start Recording"""
        try:
            # Update sample rate
            self.recorder.sample_rate = self.sample_rate_spin.value()
            
            # Setup real-time analysis callback
            def audio_callback(audio_chunk):
                if self.realtime_checkbox.isChecked():
                    # Convert to float
                    audio_float = audio_chunk.astype(np.float32) / 32768.0
                    result = self.realtime_analyzer.add_audio_chunk(audio_float)
                    if result:
                        # Update real-time info
                        self.realtime_info.setText(
                            f"Energy: {result['energy']:.4f}, "
                            f"Amplitude: {result['amplitude']:.4f}, "
                            f"ZCR: {result['zcr']:.4f}, "
                            f"Speech: {'Yes' if result['is_speech'] else 'No'}"
                        )
            
            if self.recorder.start_recording(audio_callback):
                self.record_button.setText("Stop Recording")
                self.record_status.setText("Status: Recording...")
                self.log_message("Starting recording...")
                
                # Start real-time analysis timer
                if self.realtime_checkbox.isChecked():
                    self.timer.start(100)  # 100ms更新一次
            else:
                QMessageBox.critical(self, "错误", "无法Start Recording，请检查音频设备")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Start Recording失败: {e}")
            self.log_message(f"Start Recording失败: {e}")
    
    def stop_recording(self):
        """Stop recording"""
        try:
            self.audio_data = self.recorder.stop_recording()
            
            if len(self.audio_data) > 0:
                # Update display
                self.signal_widget.plot_audio_signal(
                    self.audio_data, self.recorder.sample_rate, "Recording Result")
                
                self.record_button.setText("Start Recording")
                self.record_status.setText("Status: Recording completed")
                self.log_message(f"Recording completed, duration: {len(self.audio_data)/self.recorder.sample_rate:.2f} seconds")
                
                # Stop timer
                self.timer.stop()
            else:
                self.record_button.setText("Start Recording")
                self.record_status.setText("Status: Recording failed")
                self.log_message("Recording failed, no audio data received")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Stop recording failed: {e}")
            self.log_message(f"Stop recording failed: {e}")
    
    def analyze_audio(self):
        """Analyze audio"""
        if self.audio_data is None:
            QMessageBox.warning(self, "Warning", "Please load an audio file or record first")
            return
        
        try:
            self.analyze_button.setEnabled(False)
            self.progress_bar.setValue(0)
            
            # Create worker thread
            self.analysis_worker = AudioAnalysisWorker(self.audio_data, self.sample_rate)
            self.analysis_worker.analysis_complete.connect(self.on_analysis_complete)
            self.analysis_worker.progress_update.connect(self.progress_bar.setValue)
            self.analysis_worker.start()
            
            self.log_message("Starting audio analysis...")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Analysis failed: {e}")
            self.log_message(f"Analysis failed: {e}")
            self.analyze_button.setEnabled(True)
    
    def on_analysis_complete(self, result):
        """Analysis complete callback"""
        try:
            # Display analysis results
            self.analysis_widget.plot_analysis_results(result['analysis'], result['sample_rate'])
            
            # Display endpoint detection results
            if result['endpoints']['endpoints']:
                endpoints = result['endpoints']['endpoints']
                self.log_message(f"Detected {len(endpoints)} speech segments")
                for i, endpoint in enumerate(endpoints):
                    start_time = endpoint['start_time']
                    end_time = endpoint['end_time']
                    self.log_message(f"Speech segment {i+1}: {start_time:.2f}s - {end_time:.2f}s")
            else:
                self.log_message("No speech segments detected")
            
            self.analyze_button.setEnabled(True)
            self.log_message("Audio analysis completed")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Display analysis results失败: {e}")
            self.log_message(f"Display analysis results失败: {e}")
            self.analyze_button.setEnabled(True)
    
    def recognize_speech(self):
        """Recognize speech"""
        if self.audio_data is None:
            QMessageBox.warning(self, "Warning", "Please load an audio file or record first")
            return
        
        try:
            self.recognize_button.setEnabled(False)
            self.progress_bar.setValue(0)
            
            # Create worker thread
            self.recognition_worker = SpeechRecognitionWorker(
                self.audio_data, self.sample_rate, self.recognizer)
            self.recognition_worker.recognition_complete.connect(self.on_recognition_complete)
            self.recognition_worker.progress_update.connect(self.progress_bar.setValue)
            self.recognition_worker.start()
            
            self.log_message("Starting speech recognition...")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Recognition failed: {e}")
            self.log_message(f"Recognition failed: {e}")
            self.recognize_button.setEnabled(True)
    
    def on_recognition_complete(self, result):
        """Recognition complete callback"""
        try:
            if result['success']:
                digit = result['digit']
                confidence = result['confidence']
                self.recognition_result.setText(f"Recognition result: {digit} (Confidence: {confidence:.2f})")
                self.log_message(f"Recognition result: {digit}, Confidence: {confidence:.2f}")
            else:
                self.recognition_result.setText("Recognition result: 识别失败")
                self.log_message("Speech recognition failed")
            
            self.recognize_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Failed to process recognition result: {e}")
            self.log_message(f"Failed to process recognition result: {e}")
            self.recognize_button.setEnabled(True)
    
    def update_realtime_info(self):
        """Update real-time info"""
        # 这个方法会被定时器调用
        pass
    
    def closeEvent(self, event):
        """Close event handler"""
        # Stop recording
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        
        # Cleanup resources
        self.recorder.cleanup()
        
        event.accept()


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("Speech Signal Processing System")
    app.setApplicationVersion("1.0.0")
    
    # Create main window
    window = SpeechProcessingGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
