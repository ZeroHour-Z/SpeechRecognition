# 快速开始指南

## 🚀 5分钟快速体验

### 1. 环境准备
```bash
# 创建虚拟环境（推荐）
conda create -n dsp1 python=3.10
conda activate dsp1

# 安装依赖
pip install -r requirements.txt
```

### 2. 生成测试音频
```bash
python examples/generate_test_audio.py
```
这将自动生成14个测试WAV文件到 `data/audio/` 目录。

### 3. 运行主程序
```bash
python main.py
```

### 4. 选择功能
在交互式菜单中：
- 选择 `1` - 基础分析演示
- 选择 `2` - 窗函数比较  
- 选择 `3` - 端点检测演示
- 选择 `4` - 完整分析流程

## 📁 项目结构概览

```
dsp/
├── main.py                    # 🎯 主程序入口
├── speech_processing/         # 📦 核心功能包
├── examples/                  # 📚 示例程序
├── tests/                     # 🧪 测试文件
└── data/                      # 📊 数据目录
    ├── audio/                 # 🎵 音频文件
    └── results/               # 📈 分析结果
```

## 🎯 核心功能演示

### 基础分析
```python
from speech_processing import WAVReader, FrameProcessor, TimeDomainAnalyzer

# 读取音频
reader = WAVReader("data/audio/test_speech_like.wav")
audio_data, sample_rate = reader.read()

# 分帧处理
processor = FrameProcessor(sample_rate, 25.0, 10.0)
frames, windowed_frames = processor.process_signal(audio_data, 'hamming')

# 时域分析
analyzer = TimeDomainAnalyzer(sample_rate, 25.0, 10.0)
result = analyzer.analyze_signal(audio_data, 'hamming')
```

### 端点检测
```python
from speech_processing import DualThresholdEndpointDetector

detector = DualThresholdEndpointDetector(sample_rate, 25.0, 10.0)
result = detector.detect_endpoints(audio_data)
print(f"检测到 {len(result['endpoints'])} 个语音段")
```

## 📊 测试文件说明

生成的测试文件包括：

| 文件名 | 描述 | 用途 |
|--------|------|------|
| `test_440hz.wav` | 440Hz纯音 | 测试基本功能 |
| `test_880hz.wav` | 880Hz纯音 | 频率对比 |
| `test_speech_like.wav` | 语音模拟 | 时域分析 |
| `test_speech_with_silence.wav` | 带静音语音 | 端点检测 |
| `digit_0.wav` ~ `digit_9.wav` | 数字语音模拟 | 语音识别 |

## 🔧 常见问题

### Q: 提示"没有找到WAV文件"
**A:** 运行 `python examples/generate_test_audio.py` 生成测试文件

### Q: 导入错误
**A:** 确保在项目根目录运行，并且已安装依赖

### Q: 图形不显示
**A:** 确保matplotlib后端正确配置，或使用 `plt.show()` 显示图形

### Q: 中文显示为方块
**A:** 使用英文版本示例：
```bash
python examples/simple_demo.py
python examples/window_comparison_english.py
```

## 📈 实验建议

1. **从基础开始**：先运行基础分析演示
2. **对比窗函数**：观察不同窗函数的效果
3. **调整参数**：尝试不同的帧长和阈值
4. **分析结果**：查看生成的分析报告

## 🎓 学习路径

1. **基础概念**：理解分帧、加窗、时域特征
2. **参数调优**：学习如何调整分析参数
3. **结果解读**：学会分析可视化结果
4. **扩展应用**：尝试自己的音频文件

---

**开始您的语音信号处理之旅吧！** 🎉
