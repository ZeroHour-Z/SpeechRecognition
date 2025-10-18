# 学术论文

本目录包含语音识别系统的学术论文相关文件。

## 文件说明

- **paper.tex** - LaTeX源文件
- **paper.pdf** - 编译后的PDF论文
- **acmart.cls** - ACM论文模板类文件
- **acmart-taps.sty** - ACM论文样式文件
- **system_architecture.png** - 系统架构图

## 编译说明

### 编译PDF
```bash
# 使用XeLaTeX编译（支持中文）
xelatex paper.tex

# 或使用PDFLaTeX
pdflatex paper.tex
```

### 清理临时文件
```bash
# 删除编译产生的临时文件
rm -f *.aux *.log *.out *.synctex.gz *.toc *.lof *.lot
```

## 论文信息

- **标题**: 基于时域特征的孤立字语音识别系统设计与实现
- **作者**: 周湛昊、张振鑫、孙鑫磊、王毅
- **单位**: 西安交通大学
- **会议**: Chinese Conference on Computer Vision and Pattern Recognition (CCFA '25)
- **年份**: 2025

## 系统架构

论文中使用的系统架构图位于 `system_architecture.png`，展示了语音识别系统的整体结构。

## 注意事项

1. 论文使用ACM模板格式
2. 支持中文内容（使用xeCJK包）
3. 已配置为无版权模式（`\setcopyright{none}`）
4. 使用XeLaTeX编译器以获得最佳中文支持
