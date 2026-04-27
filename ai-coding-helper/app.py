"""
================================================================================
应用入口 —— Tkinter GUI 窗口组装与启动
================================================================================

【架构定位】
本文件是 MVC 中的 View 组装层，负责：
  1. 创建所有 UI 组件（工具栏、聊天区、日志区、输入区）
  2. 将所有组件引用打包成 widgets 字典传给 Controller（AppHandlers）
  3. 初始化 ModelConfigManager（模型配置持久化）
  4. 注册窗口关闭回调
  5. 启动 Tkinter 主事件循环

【设计模式：函数式的 UI 组装】
每个 UI 区域由 ui/widgets.py 中的独立工厂函数创建：
  create_workspace_widgets() → 工作区输入框 + 选择/设置/清理按钮
  create_model_widgets()     → 模型下拉框 + 测试/管理按钮
  create_mode_toggle()       → Code/Chat 模式切换按钮
  create_rollback_widgets()  → 版本回退输入框 + 选择/回退/历史按钮
  create_main_paned()        → 左右分栏容器
  create_chat_header()       → 聊天区标题 + 清空按钮
  create_log_frame()         → 日志文本区域
  create_input_frame()       → 底部输入框 + 发送/提交文件按钮

所有返回值通过 widgets 字典统一传给 AppHandlers，
实现 UI 层与业务逻辑层的解耦。

【项目整体架构（MVC + Agent）】
  ┌─────────────────────────────────────────────────────────────────┐
  │  app.py (View 组装)                                              │
  │    ├── ui/widgets.py    →  UI 组件工厂 (Tkinter 控件)            │
  │    ├── ui/config.py     →  全局配置 (颜色/尺寸/模型列表)          │
  │    ├── ui/handlers.py   →  Controller (事件处理 + Agent 管理)    │
  │    └── ui/model_manager.py → Model CRUD (JSON持久化)             │
  │                                                                   │
  │  core/                                                            │
  │    ├── agent.py         →  Agent 编排 (LLM + 工具注册 + 上下文)  │
  │    ├── file_tool.py     →  14+5 个文件/文件夹操作工具             │
  │    └── rag/              →  RAG 检索管道                          │
  │         ├── code_parser.py  → 多语言 AST 解析                     │
  │         ├── vector_store.py → ChromaDB 向量存储                   │
  │         ├── rag_manager.py  → 索引生命周期管理                    │
  │         └── rag_tool.py     → LangChain @tool 检索接口            │
  │                                                                   │
  │  utils/                                                           │
  │    ├── hash_index.py    →  多语言代码块hash定位                   │
  │    ├── path_validator.py →  文件路径安全校验                       │
  │    └── version_manager.py → 文件版本快照/回退                     │
  └─────────────────────────────────────────────────────────────────┘

【面试高频题结合本项目】
Q1: 什么是 LangChain？你们项目中怎么用的？
A:  LangChain 是一个 LLM 应用开发框架，提供 Chains/Agents/Tools/Memory 等抽象。
    本项目使用：@tool 装饰器注册工具、create_agent 创建 ReAct Agent、
    ChatOpenAI 调用 LLM（兼容 DeepSeek/通义千问等 OpenAI 格式 API）、
    LangGraph MemorySaver 管理会话状态（断点续传）。

Q2: 什么是 Agent？ReAct 模式是什么？
A:  Agent = LLM + 工具 + 循环决策。ReAct（Reasoning + Acting）模式：
    LLM 思考 → 决定调用哪个工具 → 执行工具获取结果 → 观察结果 → 继续思考
    → 直到认为任务完成，输出最终文本回复。本项目在 core/agent.py 中实现。

Q3: 什么是 RAG？为什么需要它？
A:  RAG（检索增强生成）= 检索 + 生成。LLM 不知道你的项目代码，RAG 先把代码
    向量化存入 ChromaDB，用户提问时检索最相关代码片段注入 Prompt。
    本项目完整实现了：代码解析 → 向量化 → 存储 → 检索 → Reranking 全链路。

Q4: 怎么做上下文管理避免 Token 超限？
A:  smart_context_trimmer（agent.py）：历史轮次的 ToolMessage（文件内容）
    被替换为操作摘要，只保留最新一轮完整内容。DeepSeek 有 128K 窗口，
    但文件内容可能几万 tokens，修剪后有效利用窗口。
"""
import tkinter as tk
import logging
from ui.widgets import (
    create_main_window, create_top_frame, create_workspace_widgets,
    create_model_widgets, create_mode_toggle, create_func_frame,
    create_rollback_widgets, create_main_paned, create_log_frame,
    create_input_frame, create_workspace_info_panel, create_chat_header,
    ScrollableChatFrame
)
from ui.handlers import AppHandlers
from ui.model_manager import ModelConfigManager

# logging.basicConfig 配置日志格式：时间戳 - 级别 - 消息
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    应用启动入口函数。
    职责：创建窗口 → 组装UI → 创建Controller → 启动事件循环。
    无参数，无返回值（进入 Tkinter 主循环后阻塞直到窗口关闭）。
    """

    # ---- 创建主窗口 ----
    # tk.Tk() 是 Tkinter 的根窗口对象，整个 GUI 的容器
    root = tk.Tk()
    root = create_main_window(root)   # 设置标题("AI 代码助手")、尺寸("1400x850")、背景色(#FFFFFF)

    # ---- 初始化模型配置管理器（加载 model_configs.json） ----
    # ModelConfigManager 负责 JSON 持久化，启动时加载已保存的模型配置并回填环境变量
    config_manager = ModelConfigManager()

    # ---- Windows 高 DPI 适配 ----
    # 在高分辨率屏幕上防止界面模糊（设置进程级 DPI 感知）
    # SetProcessDpiAwareness(1) = PROCESS_SYSTEM_DPI_AWARE，让 Windows 不缩放此进程
    try:
        import ctypes
        if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'shcore'):
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e:
        logger.warning(f"高DPI适配失败: {e}")

    # ========== 顶部工具栏 ==========
    # top_frame: 白色背景、40px高、底部有分割线
    top_frame = create_top_frame(root)

    # 工作区控件组：输入框 + 选择(📁)/设置(⚡)/清理(🗑)按钮
    # workspace_entry: ttk.Entry, 用于输入/显示工作区路径
    workspace_entry, select_workspace_btn, set_workspace_btn, clear_memory_btn = create_workspace_widgets(top_frame)

    # 模型控件组：下拉框 + 名称标签 + 状态灯(⚪/⏳/🟢/🔴) + 测试/管理按钮
    model_combobox, model_name_label, status_light, test_model_btn, model_manage_btn = create_model_widgets(top_frame)

    # Code/Chat 模式切换按钮：初始为 "💻 Code"（蓝色）
    mode_toggle_btn = create_mode_toggle(top_frame)

    # ========== 功能区（第二排） ==========
    func_frame = create_func_frame(root)
    # 版本回退控件组：输入框 + 选择文件(📁)/回退(↩)/历史(📋)按钮
    rollback_file_entry, select_rollback_file_btn, rollback_btn, history_btn = create_rollback_widgets(func_frame)

    # ========== 主面板（左右分栏：PanedWindow 可拖动调整分屏比例） ==========
    # ttk.PanedWindow：Tkinter 的可拖拽分栏容器，weight 决定初始占比
    main_paned = create_main_paned(root)

    # ------- 左侧：聊天区（比例 3） -------
    chat_container = tk.Frame(main_paned, bg="white")
    clear_chat_btn = create_chat_header(chat_container, title="对话", clear_command=None)
    # ScrollableChatFrame: Canvas+Frame 可滚动容器（支持鼠标滚轮）
    chat_area = ScrollableChatFrame(chat_container)
    chat_area.pack(fill="both", expand=True)

    # ------- 右侧：工作区信息 + 日志（比例 1） -------
    right_panel = tk.Frame(main_paned, bg="white")
    # notebook: ttk.Notebook 标签页容器, info_text: 工作区信息文本区(scrolledtext,只读)
    notebook, info_text = create_workspace_info_panel(right_panel)
    # log_frame: ttk.Frame, log_text: scrolledtext.ScrolledText（等宽字体,只读）
    log_frame, log_text = create_log_frame(right_panel)
    notebook.add(log_frame, text="📜 日志")

    # 清空日志按钮（右面板底部，右对齐）
    clear_log_btn = tk.Button(
        right_panel, text="🗑 清空日志", font=("Microsoft YaHei", 9),
        bg="#F5F5F5", relief="flat", padx=6, cursor="hand2"
    )
    clear_log_btn.pack(side="bottom", anchor="e", padx=6, pady=2)

    # weight=3:weight=1 → 聊天区占3/4, 日志区占1/4
    main_paned.add(chat_container, weight=3)
    main_paned.add(right_panel, weight=1)

    # ========== 底部输入区 ==========
    # input_entry: 输入框(fill=x,expand=True 占据剩余空间)
    # submit_file_btn: "提交文件", send_btn: "发送 ➤"(蓝色)
    input_container, input_entry, submit_file_btn, send_btn = create_input_frame(root)

    # ========== 组装 widgets 字典 ==========
    # 将所有 UI 组件的引用通过字典传给 AppHandlers（Controller）。
    # 键名是 handler._unpack_widgets() 解包时的约定标识。
    # 字典传参的好处：新增 UI 组件只需改 app.py + handlers.py 两处，不改函数签名。
    widgets = {
        "workspace_entry": workspace_entry,
        "select_workspace_btn": select_workspace_btn,
        "set_workspace_btn": set_workspace_btn,
        "clear_memory_btn": clear_memory_btn,
        "model_combobox": model_combobox,
        "model_name_label": model_name_label,
        "status_light": status_light,
        "test_model_btn": test_model_btn,
        "model_manage_btn": model_manage_btn,
        "mode_toggle_btn": mode_toggle_btn,
        "rollback_file_entry": rollback_file_entry,
        "select_rollback_file_btn": select_rollback_file_btn,
        "rollback_btn": rollback_btn,
        "history_btn": history_btn,
        "chat_area": chat_area,
        "clear_chat_btn": clear_chat_btn,
        "log_text": log_text,
        "input_entry": input_entry,
        "send_btn": send_btn,
        "clear_log_btn": clear_log_btn,
        "submit_file_btn": submit_file_btn,
        "info_text": info_text,
    }

    # ---- 创建主控制器（MVC 中的 Controller） ----
    # AppHandlers 构造函数内：
    #   1. _unpack_widgets() 从 dict 提取 UI 组件引用
    #   2. _bind_events() 绑定按钮事件到处理方法
    #   3. StdoutRedirector 重定向 print() 到日志面板
    app = AppHandlers(root, widgets, config_manager)

    # ---- 绑定清空聊天按钮（handler 创建后才能设置 command） ----
    if clear_chat_btn:
        clear_chat_btn.config(command=app._clear_chat)

    # ---- 窗口关闭回调（优雅退出：释放资源 + 清理临时目录） ----
    # protocol("WM_DELETE_WINDOW") 拦截窗口关闭事件
    # on_closing → 停止文件监听 → 关闭 ChromaDB → 清理临时工作区 → 销毁窗口
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    logger.info("桌面应用启动成功")
    # root.mainloop() 进入 Tkinter 主事件循环（阻塞式，持续监听用户交互事件）
    root.mainloop()


if __name__ == "__main__":
    main()
