"""全局配置模块 —— 所有可调参数和常量的集中管理。

【设计原则】
- 避免魔法数字：所有阈值、尺寸、颜色都有命名常量
- 单一修改点：调参只需改这个文件
- 环境变量注入：API Key 等敏感信息通过 .env 文件加载，不写死在代码中

【模块组织】
  用户配置区:   默认模型、临时工作区路径、版本数量上限
  上下文管理:   最大历史轮数、是否启用智能修剪
  RAG 配置:     向量模型、集合名、存储目录
  路径校验:     允许/禁止的文件类型和目录
  模型配置:     内置模型列表（会被 model_configs.json 覆盖）
  UI 样式:      窗口尺寸、字体、颜色常量
  持久化加载:   启动时从 model_configs.json 覆盖 SUPPORTED_MODELS
"""
import os
from dotenv import load_dotenv

# ---- 启动时加载 .env 文件 ----
# python-dotenv 库：读取项目根目录的 .env 文件，注入到 os.environ
# 这样开发者只需填写 .env 文件，无需修改代码
load_dotenv()

# ==================== 用户配置区 ====================
DEFAULT_MODEL = "deepseek"                       # 默认使用的模型标识

# 临时工作区：当用户未设置工作区但提交了外部文件时，自动创建临时目录托管
TEMP_WORKSPACE = "./temp_workspace"

# 版本管理：每个文件最多保留多少个历史快照（超过则自动清理旧快照）
MAX_HISTORY_VERSIONS = 2

# ==================== 上下文管理核心配置 ====================
MAX_HISTORY_ROUNDS = 50                          # 最多保留多少轮对话（当前未强制限制，由 trimmer 动态处理）
AUTO_TRIM_CONTEXT = True                         # True=每次LLM调用前自动修剪历史消息，False=不修剪（全量发送）

# ==================== RAG 核心配置 ====================
RAG_DEFAULT_TOP_K = 5                            # 默认检索返回数量（已被动态阈值取代，保留作为 fallback）
RAG_EMBEDDING_MODEL = "text-embedding-v3"        # 阿里云 DashScope 嵌入模型
RAG_COLLECTION_NAME = "code_assistant_symbols"    # ChromaDB 集合名（相当于数据库的"表名"）
RAG_PERSIST_DIR = ".code_rag_index"              # 向量库持久化目录（相对于工作区）

# ==================== 路径校验配置 ====================

# 允许操作/索引的文件后缀（白名单）
# 只有这些类型的文件才能被 Agent 读写和 RAG 索引
ALLOWED_EXT = {
    ".py", ".js", ".ts", ".java", ".cpp", ".cxx", ".cc", ".c", ".h", ".hpp", ".hh", ".hxx", ".go", ".rs",
    ".html", ".css", ".vue", ".tsx", ".jsx",
    ".xml", ".ini", ".toml",
    ".pro", ".pri", ".ui", ".qrc", ".qml", ".cmake",     # Qt/C++ 项目文件
}

# 禁止操作的目录（这些目录在任何操作中都被跳过）
FORBIDDEN_DIR = {
    ".workspace_versions", ".git", ".svn", ".idea", ".vscode",
    "__pycache__", "node_modules", ".code_rag_index"
}

# 强制禁止操作的文件后缀（即使后缀在 ALLOWED_EXT 中也不允许 —— 但当前 ALLOWED_EXT 中不含这些）
FORBIDDEN_EXT = {".md", ".txt", ".json", ".yml", ".yaml", ".pdf", ".docx", ".log"}

# 无扩展名/特殊文件名的白名单（通过 basename 匹配，跳过后缀检查）
ALLOWED_FILENAMES = {"CMakeLists.txt", "Makefile", "Dockerfile"}

# ==================== 模型配置 ====================
# 这是内置默认配置。如果 model_configs.json 存在，会被 _load_persisted_models 覆盖。
# api_key 从环境变量读取的原因：不应该把密钥写死在代码或 JSON 配置中
SUPPORTED_MODELS = {
    "deepseek": {
        "name": "DeepSeek",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),    # 环境变量 $DEEPSEEK_API_KEY
        "base_url": "https://api.deepseek.com",          # OpenAI 兼容的 API 地址
        "model_name": "deepseek-reasoner",               # 实际调用的模型名
    },
    "qwen": {
        "name": "阿里云通义千问",
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),   # 环境变量 $DASHSCOPE_API_KEY
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3.6-plus",
    }
}

# ==================== UI 配置 ====================
WINDOW_SIZE = "1400x850"                        # 默认窗口尺寸
WINDOW_MIN_SIZE = (1100, 700)                   # 最小窗口尺寸（防止缩太小导致界面错乱）
FONT_NORMAL = ("Microsoft YaHei", 10)            # 普通文本字体（微软雅黑 10pt）
FONT_BOLD = ("Microsoft YaHei", 10, "bold")     # 粗体（标签标题）
FONT_LOG = ("Consolas", 9)                       # 等宽字体（日志/代码）
FONT_TITLE = ("Microsoft YaHei", 11, "bold")    # 标题字体
FONT_CODE = ("Consolas", 10)                     # 代码块字体

# 日志面板的样式标签（tag）定义
# Tkinter Text 控件通过 tag_config 应用字体和颜色
LOG_TAGS = {
    "model":       {"foreground": "#6633CC", "font": ("Consolas", 9, "bold")},  # 模型思考日志（紫色粗体）
    "tool_call":   {"foreground": "#FF8800", "font": FONT_LOG},                  # 工具调用（橙色）
    "tool_running": {"foreground": "#1E90FF", "font": FONT_LOG},                 # 工具运行中（蓝色）
    "tool_success": {"foreground": "#009900", "font": FONT_LOG},                 # 工具成功（绿色）
    "tool_fail":   {"foreground": "#FF3333", "font": ("Consolas", 9, "bold")},  # 工具失败（红色粗体）
    "system":      {"foreground": "#666666", "font": FONT_LOG},                  # 系统消息（灰色）
}

# 模型连通性测试的状态指示灯
STATUS_LIGHTS = {
    "untested": "⚪",   # 灰圆 = 未测试
    "testing":  "⏳",   # 沙漏 = 测试中
    "success":  "🟢",   # 绿圆 = 通过
    "fail":     "🔴",   # 红圆 = 失败
}

# ==================== 块式 UI 颜色配置 ====================
# 聊天消息块的颜色方案（Material Design 风格）
CHAT_BG = "#F5F5F5"                    # 聊天区背景
BLOCK_USER_BG = "#E3F2FD"              # 用户消息块背景（浅蓝）
BLOCK_USER_FG = "#0D47A1"              # 用户消息块文字（深蓝）
BLOCK_AI_BG = "#F3E5F5"               # AI 消息块背景（浅紫）
BLOCK_AI_FG = "#4A148C"               # AI 消息块文字（深紫）
BLOCK_SYSTEM_BG = "#FFF3E0"           # 系统消息块背景（浅橙）
BLOCK_SYSTEM_FG = "#E65100"           # 系统消息块文字（深橙）
BLOCK_DIFF_BORDER = "#E0E0E0"         # Diff 对比块边框
BLOCK_DIFF_HEADER_BG = "#FAFAFA"      # Diff 文件头背景
DIFF_ADD_BG = "#E8F5E9"               # Diff 新增行背景（浅绿）
DIFF_ADD_FG = "#1B5E20"               # Diff 新增行文字（深绿）
DIFF_DEL_BG = "#FFEBEE"               # Diff 删除行背景（浅红）
DIFF_DEL_FG = "#B71C1C"               # Diff 删除行文字（深红）
DIFF_NEUTRAL_BG = "#F5F5F5"           # Diff 无变化行背景
BUTTON_ACCEPT_BG = "#4CAF50"          # 接受按钮背景（绿色）
BUTTON_ACCEPT_FG = "white"
BUTTON_REJECT_BG = "#EF5350"          # 拒绝按钮背景（红色）
BUTTON_REJECT_FG = "white"
BUTTON_DISABLED_BG = "#BDBDBD"        # 禁用按钮背景（灰色）
STATUS_PENDING_FG = "#FF9800"         # 待处理状态（橙色）
STATUS_ACCEPTED_FG = "#4CAF50"        # 已接受状态（绿色）
STATUS_REJECTED_FG = "#EF5350"        # 已拒绝状态（红色）

# ==================== 持久化模型配置加载 ====================
# 启动时从 model_configs.json 读取用户自定义的模型配置，
# 覆盖上面的 SUPPORTED_MODELS 和 DEFAULT_MODEL。
# 【优先级】：用户编辑过并保存的值（JSON中非空）> 环境变量 > 空字符串
def _load_persisted_models():
    global SUPPORTED_MODELS, DEFAULT_MODEL
    import json as _json, os as _os
    _config_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "model_configs.json")
    if not _os.path.exists(_config_path):
        return   # 没有持久化文件 → 使用上面的默认配置

    try:
        with open(_config_path, "r", encoding="utf-8") as _f:
            _data = _json.load(_f)
        _loaded = _data.get("models", {})
        _default = _data.get("default_model", DEFAULT_MODEL)

        # API Key 回退映射：如果用户没在界面上填 api_key → 尝试从环境变量读取
        _env_map = {"deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY"}

        SUPPORTED_MODELS.clear()
        for _k, _m in _loaded.items():
            if _m.get("enabled", True):   # 只加载已启用的模型
                _api_key = _m.get("api_key", "")
                if not _api_key.strip() and _k in _env_map:
                    _api_key = _os.getenv(_env_map[_k], "") or _api_key   # 环境变量回退
                SUPPORTED_MODELS[_k] = {
                    "name": _m["name"], "api_key": _api_key,
                    "base_url": _m["base_url"], "model_name": _m["model_name"],
                }

        if _default in SUPPORTED_MODELS:
            DEFAULT_MODEL = _default
        elif SUPPORTED_MODELS:
            DEFAULT_MODEL = next(iter(SUPPORTED_MODELS.keys()), "deepseek")
    except Exception:
        pass   # JSON 损坏或格式错误 → 静默回退到默认配置

_load_persisted_models()
