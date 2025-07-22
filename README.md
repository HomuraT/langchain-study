# LangChain 学习记录

这是一个 LangChain 框架的学习项目，记录学习过程中的示例代码、笔记和实验。

## 环境要求

- Python >= 3.11
- uv (Python 包管理工具)

## 快速开始

### 1. 安装 uv

```bash
# 使用 pip 安装 uv
pip install uv
```

### 2. 克隆项目并安装依赖

```bash
git clone <your-repo-url>
cd langchain-study

# 使用 uv 创建虚拟环境并安装依赖
uv sync
```

### 3. 激活虚拟环境

```bash
# 激活虚拟环境
source .venv/bin/activate

# 或者直接使用 uv run 运行脚本
uv run python main.py
```

## 项目结构

```
langchain-study/
├── src/                # 示例代码
├── docs/               # 学习笔记
├── unitests/           # 测试用例
├── main.py             # 主程序入口
├── pyproject.toml      # 项目配置
└── README.md           # 项目说明
```

## 使用方法

```bash
# 运行主程序
uv run python main.py

# 运行特定示例（开发中）
uv run python src/example_name.py

# 运行测试
uv run python -m pytest unitests/
```

## 学习内容

- [ ] LangChain 基础概念
- [ ] LLM 集成和使用
- [ ] Chain 和 Agent 开发
- [ ] 文档处理和向量检索
- [ ] 实际应用案例

## 参考资料

- 📖 [LangChain 官方教程](https://python.langchain.com/docs/tutorials/) - 官方文档和教程
- 🦜 [LangChain GitHub](https://github.com/langchain-ai/langchain) - 源代码仓库

## 依赖管理

本项目使用 uv 进行依赖管理，主要依赖：

- `langchain>=0.3.26` - LangChain 核心框架

添加新依赖：
```bash
uv add package_name
```

## 开发

```bash
# 安装开发依赖
uv add --dev pytest black flake8

# 代码格式化
uv run black .

# 代码检查
uv run flake8 .
```

---

📚 持续学习中...
