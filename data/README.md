# 数据目录说明

## 📁 目录结构

```
data/
├── audio/                  # 音频数据目录
│   ├── samples/           # 示例音频文件
│   │   ├── digit_0.wav    # 数字0示例
│   │   ├── digit_1.wav    # 数字1示例
│   │   ├── ...
│   │   ├── digit_9.wav    # 数字9示例
│   │   ├── test_440hz.wav # 测试音频440Hz
│   │   ├── test_880hz.wav # 测试音频880Hz
│   │   ├── test_speech_like.wav # 类语音测试
│   │   ├── test_speech_with_silence.wav # 带静音测试
│   │   └── TS_VFX2_Vocal_Shot_50.wav # 音效文件
│   ├── input/             # 输入音频文件 (待处理)
│   ├── output/            # 输出音频文件 (处理结果)
│   ├── processed/         # 中间处理文件 (调试用)
│   ├── temp/              # 临时音频文件
│   │   └── recording.wav  # 录音文件
│   └── models/            # 模型文件 (训练好的模型)
├── train/                 # 训练数据 (已整理)
│   ├── 0/                # 数字0训练样本
│   │   ├── digit_0_sample_01.wav
│   │   ├── digit_0_sample_02.wav
│   │   ├── digit_0_sample_03.wav
│   │   └── digit_0_sample_04.wav
│   ├── 1/                # 数字1训练样本
│   ├── ...
│   ├── 9/                # 数字9训练样本
│   └── unknown/          # 未知类别样本
├── test/                  # 测试数据 (已整理)
│   ├── 0/                # 数字0测试样本
│   │   └── test_0_01.wav
│   ├── 1/                # 数字1测试样本
│   ├── ...
│   ├── 9/                # 数字9测试样本
│   └── unknown/          # 未知类别测试样本
├── results/               # 实验结果
│   ├── ablation/          # 消融实验结果
│   │   ├── feature_comparison.png
│   │   └── feature_importance.png
│   ├── comparison/        # 对比实验结果
│   │   ├── confusion_matrix_svm.png
│   │   ├── metrics_comparison.png
│   │   └── time_comparison.png
│   ├── performance/       # 性能测试结果
│   │   ├── comprehensive_performance.png
│   │   ├── memory_usage.png
│   │   ├── prediction_time.png
│   │   └── training_time.png
│   └── 实验结果说明.md    # 实验结果说明
└── README.md              # 本说明文件
```

## 🎯 文件命名规范

### 训练数据
- 格式：`digit_{数字}_sample_{序号}.wav`
- 示例：`digit_0_sample_01.wav`, `digit_1_sample_02.wav`

### 测试数据
- 格式：`test_{数字}_{序号}.wav`
- 示例：`test_0_01.wav`, `test_1_01.wav`

### 示例数据
- 格式：`digit_{数字}.wav` 或 `test_{描述}.wav`
- 示例：`digit_0.wav`, `test_440hz.wav`

## 📊 数据统计

### 训练数据
- 每个数字：4个样本
- 总计：40个训练样本
- 格式：WAV, 16kHz, 单声道

### 测试数据
- 每个数字：1个样本
- 总计：10个测试样本
- 格式：WAV, 16kHz, 单声道

### 示例数据
- 数字样本：10个（0-9）
- 测试音频：4个（不同频率和类型）
- 音效文件：1个

## 🔧 使用说明

### 添加新的训练数据
1. 将音频文件放入对应的数字目录
2. 按照命名规范重命名文件
3. 确保音频格式为WAV, 16kHz, 单声道

### 添加新的测试数据
1. 将音频文件放入 `testing/{数字}/` 目录
2. 按照 `test_{数字}_{序号}.wav` 格式命名

### 处理音频文件
1. 将待处理的音频放入 `input/` 目录
2. 处理后的文件会保存在 `output/` 目录
3. 临时文件保存在 `temp/` 目录

## 📈 实验结果

实验结果保存在 `results/` 目录下，包含：
- 消融实验：特征重要性分析
- 对比实验：不同分类器性能对比
- 性能测试：训练时间、预测时间、内存使用等

## 🚀 快速开始

```bash
# 查看数据统计
python examples/utils/generate_audio.py

# 运行语音识别
python examples/recognition/speech_demo.py

# 运行实验
python examples/experiments/run_experiments.py
```

## 📝 注意事项

1. **音频格式**：建议使用WAV格式，16kHz采样率，单声道
2. **文件命名**：严格按照命名规范，便于程序识别
3. **目录结构**：不要随意修改目录结构，以免影响程序运行
4. **数据备份**：重要数据请及时备份
5. **文件大小**：单个音频文件建议不超过10MB