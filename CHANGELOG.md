# 项目更新日志

记录项目的重要更新和改进。

## [2025-10-13] - 项目大整理

### 📁 目录结构优化
- **新建目录**：
  - `scripts/` - 存放打包和实验脚本
  - `config/` - 存放配置文件
  
- **文件迁移**：
  - 打包脚本移至 `scripts/`
  - 配置文件移至 `config/`
  - 保持根目录整洁

### 📝 文档精简（15个→8个）
- **删除冗余文档** (8个)：
  - PROJECT_LAYOUT.md
  - QUICK_REFERENCE.md
  - data/results/README.md
  - data/results/EXPERIMENT_REPORT.md
  - docs/PROJECT_STRUCTURE.md
  - docs/guidance.md
  - docs/实验总结.md
  - config/README.md
  - scripts/README.md

- **精简现有文档**：
  - `data/results/实验结果说明.md` - 从7.3KB精简到2.9KB
  - `data/README.md` - 简化内容

- **新增文档**：
  - `docs/README.md` - 文档导航索引
  - `docs/Git使用说明.md` - Git配置和使用指南

### 🔧 Git配置优化
- **更新 .gitignore**：
  - 添加构建产物忽略规则
  - 添加HTML导出文件忽略
  - 添加临时分析报告忽略
  - 优化音频文件配置说明
  - 添加speech_processing旧目录忽略

- **新增 .gitattributes**：
  - 配置文本文件行尾处理
  - 标记二进制文件类型
  - 配置Git LFS规则

### 🗑️ 清理工作
- 删除老旧的 `speech_processing/` 目录
- 清理重复和过时的文档

### 📊 实验结果展示
- 主README添加7张实验结果图表
- 优化图表说明和结论
- 添加清晰的文档导航

### ✨ 其他改进
- 优化README语言，减少AI味道
- 添加开发人员信息
- 统一文档格式和风格
- 提升项目专业度

## [2025-10-12] - 实验系统完成

### 🔬 实验评估
- 实现对比实验（6种分类器）
- 实现消融实验（7种特征组合）
- 实现性能测试（速度、内存、吞吐量）
- 生成11张可视化图表

### 📈 实验结果
- SVM性能最佳（F1: 0.465）
- Template Matching速度最快（0.40ms）
- 全特征组合效果最好

## [2025-09-28] - 核心功能完成

### 🎯 主要功能
- WAV文件读取与解析
- 分帧加窗处理
- 时域特征提取
- 双门限端点检测
- 语音识别功能

### 💻 界面开发
- Qt图形界面
- 命令行界面
- 打包为独立exe

### 📦 项目结构
- 建立src/核心代码结构
- 创建apps/应用程序
- 组织examples/示例程序

---

**维护者**: DSP项目组  
**最后更新**: 2025-10-13

