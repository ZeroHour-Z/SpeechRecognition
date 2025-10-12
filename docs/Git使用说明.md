# Git使用说明

项目的Git配置和常用命令说明。

## .gitignore 说明

已配置忽略以下文件/目录：

### Python相关
- `__pycache__/` - Python缓存
- `*.pyc, *.pyo` - 编译的Python文件
- `*.egg-info/` - 包信息

### 构建产物
- `build/` - PyInstaller构建目录
- `dist/` - 打包输出目录
- `*.exe` - 可执行文件
- `*.spec` - PyInstaller配置（已移到scripts/）

### IDE和编辑器
- `.idea/` - PyCharm
- `.vscode/` - VSCode
- `*.code-workspace` - VSCode工作区

### 临时文件
- `*.log` - 日志文件
- `*.tmp, *.bak` - 临时/备份文件
- `data/results/*.txt` - 分析报告临时文件
- `docs/*.html` - HTML导出文件

### 音频数据（可选）
默认**不**忽略音频文件，如果数据太大可以取消注释：
```gitignore
data/audio/*.wav
data/train/*/*.wav
data/test/*/*.wav
```

## .gitattributes 说明

配置了文件类型和行尾处理：

- **Python文件**: LF行尾（Unix风格）
- **批处理文件**: CRLF行尾（Windows风格）
- **音频/图片**: 标记为二进制
- **大模型文件**: 使用Git LFS

## 常用Git命令

### 初始提交

```bash
# 初始化仓库
git init

# 添加所有文件
git add .

# 查看将要提交的文件
git status

# 提交
git commit -m "Initial commit: 语音识别系统"

# 添加远程仓库
git remote add origin https://github.com/yourusername/dsp.git

# 推送
git push -u origin main
```

### 日常使用

```bash
# 查看状态
git status

# 添加修改的文件
git add <file>
# 或添加所有修改
git add .

# 提交
git commit -m "描述修改内容"

# 推送
git push

# 拉取更新
git pull
```

### 忽略已追踪的文件

如果某些文件已经被git追踪，想要忽略它们：

```bash
# 停止追踪但保留文件
git rm --cached <file>

# 停止追踪整个目录
git rm -r --cached <directory>

# 然后提交
git commit -m "Remove tracked files that should be ignored"
```

### 查看忽略的文件

```bash
# 查看会被忽略的文件
git status --ignored

# 检查某个文件是否被忽略
git check-ignore -v <file>
```

## 分支管理

### 创建功能分支

```bash
# 创建并切换到新分支
git checkout -b feature/new-feature

# 或者
git branch feature/new-feature
git checkout feature/new-feature
```

### 合并分支

```bash
# 切换到主分支
git checkout main

# 合并功能分支
git merge feature/new-feature

# 删除已合并的分支
git branch -d feature/new-feature
```

## 常见问题

### Q: 不小心提交了大文件怎么办？

```bash
# 从历史中删除文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch <path-to-file>" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送
git push --force
```

### Q: 如何查看文件大小？

```bash
# 查看大文件
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | sort -n -k 2
```

### Q: .gitignore不生效？

可能是文件已经被追踪了，需要先移除追踪：

```bash
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
```

## 建议的提交规范

使用清晰的commit message：

```
feat: 添加新功能
fix: 修复bug
docs: 更新文档
style: 代码格式调整
refactor: 重构代码
test: 添加测试
chore: 构建/工具相关

例如：
feat: 添加SVM分类器
fix: 修复端点检测bug
docs: 更新README实验结果
refactor: 重组项目结构
```

## 文件大小建议

- 单个文件 < 10MB
- 如果有大量音频数据，考虑使用Git LFS
- 训练好的模型文件使用Git LFS

## 推荐的.gitignore模板

根据项目需要，可以参考：
- [Python .gitignore](https://github.com/github/gitignore/blob/main/Python.gitignore)
- [Windows .gitignore](https://github.com/github/gitignore/blob/main/Global/Windows.gitignore)

---

**提示**：本项目已配置好.gitignore和.gitattributes，可以直接使用！

