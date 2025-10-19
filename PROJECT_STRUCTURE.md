# 项目结构说明

## 目录结构

```
SpeechRecognition/
├── 📁 apps/                    # 应用程序入口
│   ├── gui_interface.py        # GUI界面实现
│   ├── main.py                 # 命令行主程序
│   └── run_gui.py              # GUI启动器
├── 📁 build/                   # 构建和打包脚本
│   ├── app.spec                # PyInstaller配置文件
│   ├── build.py                # 构建脚本
│   ├── build.bat               # Windows构建脚本
│   ├── install.bat             # Windows安装脚本
│   └── uninstall.bat           # Windows卸载脚本
├── 📁 config/                  # 配置文件
│   └── config.py               # 系统配置参数
├── 📁 data/                    # 数据目录
│   ├── 📁 audio/               # 音频文件
│   │   ├── 📁 samples/         # 测试音频样本
│   │   └── 📁 output/          # 输出音频文件
│   ├── 📁 train/               # 训练数据（数字0-9）
│   ├── 📁 test/                # 测试数据
│   └── README.md               # 数据说明
├── 📁 docs/                    # 文档目录
├── 📁 examples/                # 示例程序
│   ├── 📁 analysis/            # 信号分析示例
│   ├── 📁 detection/           # 端点检测示例
│   ├── 📁 experiments/         # 实验脚本
│   ├── 📁 recognition/         # 识别算法示例
│   └── 📁 utils/               # 工具示例
├── 📁 src/                     # 源代码
│   ├── 📁 core/                # 核心模块
│   │   ├── audio_recorder.py   # 音频录制
│   │   ├── endpoint_detection.py # 端点检测
│   │   ├── frame_window.py     # 分帧加窗
│   │   ├── time_domain_analysis.py # 时域分析
│   │   └── wav_reader.py       # WAV文件读取
│   ├── 📁 recognition/         # 识别模块
│   │   ├── advanced_recognizer.py # 高级识别器
│   │   ├── classifiers.py      # 分类器实现
│   │   └── simple_recognizer.py # 简单识别器
│   └── 📁 utils/               # 工具模块
│       └── plot_config.py      # 绘图配置
├── 📁 tests/                   # 测试代码
│   ├── test_frame.py           # 分帧测试
│   └── test_wav.py             # WAV读取测试
├── 📁 paper/                   # 论文目录
├── 📄 requirements.txt         # Python依赖
├── 📄 LICENSE                  # 许可证
├── 📄 README.md                # 项目说明
└── 📄 PROJECT_STRUCTURE.md     # 本文档
```

## 模块说明

### 核心模块 (src/core/)

- **wav_reader.py**: WAV音频文件读取和解析
- **frame_window.py**: 音频分帧和窗函数应用
- **time_domain_analysis.py**: 时域特征提取（短时能量、过零率、平均幅度）
- **endpoint_detection.py**: 双门限端点检测算法
- **audio_recorder.py**: 实时音频录制功能

### 识别模块 (src/recognition/)

- **simple_recognizer.py**: 基于模板匹配的简单识别器
- **advanced_recognizer.py**: 支持多种分类器的高级识别器
- **classifiers.py**: 6种分类器实现（SVM、神经网络、朴素贝叶斯、KNN、Fisher判别、决策树）

### 应用程序 (apps/)

- **main.py**: 命令行界面主程序
- **gui_interface.py**: PyQt5图形用户界面
- **run_gui.py**: GUI启动器

### 示例程序 (examples/)

- **analysis/**: 信号分析示例和演示
- **detection/**: 端点检测算法演示
- **recognition/**: 识别算法使用示例
- **experiments/**: 实验脚本和性能测试
- **utils/**: 工具函数示例

## 数据组织

### 训练数据 (data/train/)

按数字0-9分类存储的音频文件，每个数字包含多个发音样本。

### 测试数据 (data/test/)

用于测试和验证的音频样本。

### 音频样本 (data/audio/samples/)

各种测试音频文件，包括不同频率的纯音和语音样本。

## 配置文件

### config/config.py

包含系统的默认参数配置：

- 分帧参数（帧长、帧移）
- 窗函数类型
- 端点检测阈值
- 文件路径设置
- 可视化参数

## 构建系统

### build/

包含用于打包和分发应用程序的脚本：

- **build.py**: 主要的构建脚本
- **app.spec**: PyInstaller配置文件
- **build.bat**: Windows批处理构建脚本
- **install.bat/uninstall.bat**: 安装和卸载脚本

## 文档

### docs/

- **GUI_README.md**: 图形界面使用说明
- **第一周实验周报.md**: 实验进展报告

### 论文

- **paper.tex**: LaTeX源文件
- **paper.pdf**: 编译后的PDF论文
- **system_architecture.png**: 系统架构图

## 开发规范

1. **代码组织**: 按功能模块划分，每个模块职责明确
2. **命名规范**: 使用下划线命名法，文件名清晰表达功能
3. **文档完整**: 每个模块都有相应的文档和示例
4. **测试覆盖**: 核心功能都有对应的测试用例
5. **配置分离**: 系统参数统一在config.py中管理
