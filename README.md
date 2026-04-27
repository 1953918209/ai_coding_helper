# 🤖 AI 代码助手 (AI Coding Helper)

基于 **LangChain + LangGraph + Tkinter** 构建的桌面端 AI 编程辅助工具，支持 Agent 自动编辑代码、RAG 代码检索、文件版本管理等功能。

---

## ✨ 功能特性

- 🧠 **Agent 智能编程** — 基于 LangChain ReAct 模式的代码 Agent，自动执行文件读写、编辑、创建、删除等操作
- 🔍 **RAG 代码检索** — 对工作区代码进行 AST 解析 + 向量化存储，支持语义检索，让 LLM 理解你的项目
- 💬 **Code / Chat 双模式** — Code 模式下 Agent 自动操作文件，Chat 模式下直接对话
- 🔄 **文件版本管理** — 每次修改自动保存快照，支持一键回退到历史版本
- 🖥️ **图形化界面** — 基于 Tkinter 的桌面 GUI，操作直观，支持日志实时输出
- 🔌 **多模型支持** — 内置 DeepSeek 和阿里云通义千问，支持自定义模型接入
- 🎯 **智能上下文修剪** — 自动压缩历史会话中的文件内容，防止 Token 超限

---

## 📋 系统要求

- Python 3.10+
- Windows / macOS / Linux
- 具备访问大模型 API 的网络环境

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/1953918209/ai_coding_helper.git
cd ai_coding_helper
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

**本项目通过 `.env` 文件加载 API Key，不会将密钥写死在代码中。**

在项目根目录创建 `.env` 文件，填入你的 API Key：

```bash
# .env

# DeepSeek API Key（从 https://platform.deepseek.com 获取）
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here

# 阿里云通义千问 API Key（从 https://dashscope.aliyun.com 获取）
DASHSCOPE_API_KEY=sk-your-dashscope-api-key-here
```

> ⚠️ **安全提醒**：`.env` 文件请勿提交到版本控制中（已默认加入 `.gitignore`），API Key 仅存储在本地。

### 5. 启动应用

```bash
python app.py
```

---

## 🔧 使用指南

### 选择工作区

点击顶部工具栏的 📁 按钮，选择你的项目文件夹作为工作区。Agent 将在此目录下执行文件操作。

### 选择模型

从下拉框选择模型（DeepSeek / 通义千问），点击 🟢 测试按钮验证连通性。点击 📋 管理按钮可添加自定义模型。

### Code / Chat 模式

- **Code 模式** — Agent 可以读取、编辑、创建、删除工作区中的文件
- **Chat 模式** — 仅进行对话，不会操作文件

### 文件版本回退

每次修改文件都会自动创建快照，在功能区输入文件名，点击 ↩ 可回退到上一版本。

---

## 🌐 环境变量说明

| 变量名 | 说明 | 获取地址 |
|--------|------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek 大模型 API 密钥 | [platform.deepseek.com](https://platform.deepseek.com) |
| `DASHSCOPE_API_KEY` | 阿里云通义千问 API 密钥 | [dashscope.aliyun.com](https://dashscope.aliyun.com) |

> **API Key 优先级**：用户在界面手动填写 > 环境变量 > 空字符串

> **自定义模型**：可通过模型管理界面添加任意 OpenAI 兼容的 API，自定义模型需手动填写 API Key（不支持环境变量回退）

---

## 🏗️ 项目架构

```
ai-coding-helper/
├── app.py                      # 应用入口，MVC View 组装层
├── requirements.txt            # Python 依赖
├── model_configs.json          # 模型配置持久化（运行时自动生成）
├── .env                        # 环境变量（API Key，需自行创建）
│
├── ui/                         # 用户界面层
│   ├── config.py               # 全局配置（模型列表、颜色、尺寸等）
│   ├── widgets.py              # UI 组件工厂函数
│   ├── handlers.py             # 事件处理（Controller）
│   └── model_manager.py        # 模型配置管理（增删改查 + 持久化）
│
├── core/                       # 核心业务逻辑
│   ├── agent.py                # Agent 编排（LLM + 工具 + 上下文管理）
│   ├── file_tool.py            # 14+ 个文件/文件夹操作工具
│   └── rag/                    # RAG 检索管道
│       ├── code_parser.py      # 多语言 AST 解析
│       ├── vector_store.py     # ChromaDB 向量存储
│       ├── rag_manager.py      # 索引生命周期管理
│       └── rag_tool.py         # LangChain @tool 检索接口
│
└── utils/                      # 工具模块
    ├── hash_index.py           # 多语言代码块 hash 定位
    ├── path_validator.py       # 文件路径安全校验
    └── version_manager.py      # 文件版本快照/回退
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---
个人学习项目，大家可以随意使用

