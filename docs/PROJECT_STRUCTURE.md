# 项目结构说明

## 目录结构

```
SpeechRecognition/
├── src/                          # 源代码
│   ├── __init__.py               # 包初始化文件
│   ├── core/                     # 核心功能模块
│   │   ├── __init__.py
│   │   ├── audio_recorder.py     # 音频录制
│   │   ├── wav_reader.py         # WAV文件读取
│   │   ├── frame_window.py       # 分帧和窗函数
│   │   ├── time_domain_analysis.py # 时域分析
│   │   └── endpoint_detection.py # 端点检测
│   ├── recognition/              # 语音识别模块
│   │   ├── __init__.py
│   │   ├── classifiers.py        # 分类器
│   │   ├── simple_recognizer.py  # 简单识别器
│   │   └── advanced_recognizer.py # 高级识别器
│   └── utils/                    # 工具模块
│       ├── __init__.py
│       └── plot_config.py        # 绘图配置
├── apps/                         # 应用程序
│   ├── main.py                   # 主程序
│   ├── qt_interface.py          # GUI界面
│   ├── run_gui_conda.py         # GUI启动器
│   └── run_gui_universal.py     # 通用启动器
├── examples/                     # 示例和演示
│   ├── analysis/                 # 分析示例
│   │   ├── basic_analysis_demo.py
│   │   ├── simple_analysis_demo.py
│   │   ├── speech_analysis_demo.py
│   │   └── window_comparison_demo.py
│   ├── recognition/              # 识别示例
│   │   ├── speech_recognition_demo.py
│   │   ├── classifier_comparison_demo.py
│   │   ├── cnn_demo.py
│   │   └── neural_network_demo.py
│   ├── detection/                # 检测示例
│   │   └── endpoint_detection_demo.py
│   └── utils/                    # 工具示例
│       └── generate_test_audio.py
├── tests/                        # 测试文件
│   ├── test_frame_processor.py
│   └── test_wav_reader.py
├── data/                         # 数据文件
│   ├── audio/                    # 音频数据
│   │   ├── input/                # 待处理音频
│   │   ├── output/               # 处理后音频
│   │   ├── training/             # 训练数据
│   │   │   ├── 0/                # 数字0训练数据
│   │   │   ├── 1/                # 数字1训练数据
│   │   │   ├── ...
│   │   │   ├── 9/                # 数字9训练数据
│   │   │   └── unknown/          # 未知类别训练数据
│   │   ├── testing/              # 测试数据
│   │   │   ├── 0/                # 数字0测试数据
│   │   │   ├── 1/                # 数字1测试数据
│   │   │   ├── ...
│   │   │   ├── 9/                # 数字9测试数据
│   │   │   └── unknown/          # 未知类别测试数据
│   │   ├── models/               # 保存的模型
│   │   └── results/              # 分析结果
│   └── README.md                 # 数据说明
├── docs/                         # 文档
│   ├── README.md                 # 项目说明
│   ├── GUI_README.md             # GUI使用说明
│   ├── guidance.md               # 使用指导
│   └── PROJECT_STRUCTURE.md      # 项目结构说明
├── config.py                     # 配置文件
├── requirements.txt              # 依赖文件
└── LICENSE                       # 许可证
```

## 模块说明

### src/ - 源代码
- **core/**: 核心功能模块，包含音频处理的基础功能
- **recognition/**: 语音识别模块，包含各种分类器和识别器
- **utils/**: 工具模块，包含绘图配置等辅助功能

### apps/ - 应用程序
- **main.py**: 主程序，提供交互式菜单
- **qt_interface.py**: PyQt5图形界面
- **run_gui_*.py**: 不同环境的GUI启动器

### examples/ - 示例和演示
- **analysis/**: 音频分析相关示例
- **recognition/**: 语音识别相关示例
- **detection/**: 端点检测相关示例
- **utils/**: 工具使用示例

### data/ - 数据文件
- **audio/**: 音频数据存储，按功能分类组织
- **training/**: 训练数据，按数字分类存储
- **testing/**: 测试数据，按数字分类存储
- **models/**: 保存训练好的模型
- **results/**: 分析结果和报告

### docs/ - 文档
- 项目相关文档和说明

## 使用说明

1. **运行主程序**: `python apps/main.py`
2. **运行GUI**: `python apps/run_gui_conda.py`
3. **查看示例**: 进入 `examples/` 对应子目录运行示例
4. **添加数据**: 将音频文件放入 `data/audio/` 对应目录
5. **查看结果**: 分析结果保存在 `data/audio/results/`

## 环境要求

- Python 3.8+
- conda环境: dsp1
- 依赖包: 见 requirements.txt
