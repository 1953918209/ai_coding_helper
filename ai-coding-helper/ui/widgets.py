"""
================================================================================
UI 组件库 —— Tkinter 界面元素工厂函数 & 自定义消息块
================================================================================

【模块职责】
为 app.py 提供创建 UI 组件的工厂函数，以及聊天块类。
所有 UI 创建集中在此，便于统一管理样式和布局。

【核心组件】
  ScrollableChatFrame:  Canvas + Frame 实现的可滚动聊天容器（鼠标滚轮支持）
  UserBlock:           用户消息块（浅蓝背景）
  AITextBlock:         AI 文本消息块（浅紫背景）
  SystemBlock:         系统通知块（浅橙背景）
  DiffBlock:           文件 diff 对比块（含接受/拒绝按钮，核心交互组件）

【DiffBlock 交互流程】
  Agent 修改文件 → handler 检测到变更 → 生成 DiffBlock
  → 用户点击 [接受] → 回调 _on_diff_accept → _check_diff_all_done()
  → 用户点击 [拒绝] → 回调 _on_diff_reject → 回滚文件 → _check_diff_all_done()
"""
import difflib
import tkinter as tk
from tkinter import ttk, scrolledtext
from ui.config import (
    WINDOW_SIZE, WINDOW_MIN_SIZE, FONT_NORMAL, FONT_BOLD, FONT_TITLE, FONT_CODE,
    CHAT_BG, BLOCK_USER_BG, BLOCK_USER_FG, BLOCK_AI_BG, BLOCK_AI_FG,
    BLOCK_SYSTEM_BG, BLOCK_SYSTEM_FG, BLOCK_DIFF_BORDER, BLOCK_DIFF_HEADER_BG,
    DIFF_ADD_BG, DIFF_ADD_FG, DIFF_DEL_BG, DIFF_DEL_FG, DIFF_NEUTRAL_BG,
    BUTTON_ACCEPT_BG, BUTTON_ACCEPT_FG, BUTTON_REJECT_BG, BUTTON_REJECT_FG,
    BUTTON_DISABLED_BG, STATUS_PENDING_FG, STATUS_ACCEPTED_FG, STATUS_REJECTED_FG,
    LOG_TAGS, FONT_LOG
)


# ==============================================================================
# 窗口级 UI 组件
# ==============================================================================

def create_main_window(root):
    """配置主窗口的基本属性（标题、尺寸、背景色）"""
    root.title("AI 代码助手")
    root.geometry(WINDOW_SIZE)          # "1400x850"
    root.minsize(*WINDOW_MIN_SIZE)      # (1100, 700)
    root.configure(bg="#FFFFFF")
    return root


def create_top_toolbar(root):
    """创建顶部工具栏容器（白色背景，底部有分隔线）"""
    toolbar = tk.Frame(root, bg="#FFFFFF", height=40)
    toolbar.pack(fill="x", padx=0, pady=0)
    ttk.Separator(root, orient="horizontal").pack(fill="x")
    return toolbar


# ==============================================================================
# 工作区控件
# ==============================================================================

def create_workspace_widgets(parent):
    """工作区：标签 + 输入框 + 选择/设置/清理按钮"""
    container = tk.Frame(parent, bg="#FFFFFF")
    container.pack(side="left", padx=8, pady=6)

    ttk.Label(container, text="工作区：", font=FONT_BOLD).pack(side="left")
    workspace_entry = ttk.Entry(container, width=30, font=FONT_NORMAL)
    workspace_entry.pack(side="left", padx=4)

    select_btn = tk.Button(container, text="📁 选择", font=FONT_NORMAL,
                          bg="#F5F5F5", relief="flat", padx=8, cursor="hand2")
    select_btn.pack(side="left", padx=2)

    set_btn = tk.Button(container, text="⚡ 设置", font=FONT_NORMAL,
                       bg="#1976D2", fg="white", relief="flat", padx=8, cursor="hand2")
    set_btn.pack(side="left", padx=2)

    clear_memory_btn = tk.Button(container, text="🗑 清理", font=FONT_NORMAL,
                                 bg="#F5F5F5", relief="flat", padx=8, cursor="hand2")
    clear_memory_btn.pack(side="left", padx=2)

    return workspace_entry, select_btn, set_btn, clear_memory_btn


# ==============================================================================
# 模型选择控件
# ==============================================================================

def create_model_widgets(parent):
    """模型选择：下拉框 + 型号显示 + 状态灯 + 测试/管理按钮"""
    container = tk.Frame(parent, bg="#FFFFFF")
    container.pack(side="left", padx=12, pady=6)

    ttk.Label(container, text="模型：", font=FONT_BOLD).pack(side="left")
    # 从配置动态读取模型列表（包括用户自定义的模型）
    from ui.config import SUPPORTED_MODELS, DEFAULT_MODEL
    model_combobox = ttk.Combobox(
        container, values=list(SUPPORTED_MODELS.keys()),
        state="readonly", width=10, font=FONT_NORMAL
    )
    model_combobox.set(DEFAULT_MODEL)
    model_combobox.pack(side="left", padx=4)

    # 显示当前模型的友好名称（如 "DeepSeek"）
    model_name_label = tk.Label(container, text=SUPPORTED_MODELS.get(DEFAULT_MODEL, {}).get("name", ""),
                                font=FONT_NORMAL, bg="#FFFFFF", fg="#666666")
    model_name_label.pack(side="left", padx=2)

    # 连通性测试状态灯
    status_light = tk.Label(container, text="⚪", font=("Microsoft YaHei", 12), bg="#FFFFFF")
    status_light.pack(side="left", padx=2)

    test_btn = tk.Button(container, text="测试连通性", font=FONT_NORMAL,
                        bg="#F5F5F5", relief="flat", padx=8, cursor="hand2")
    test_btn.pack(side="left", padx=4)

    manage_btn = tk.Button(container, text="⚙ 管理", font=FONT_NORMAL,
                          bg="#F5F5F5", relief="flat", padx=8, cursor="hand2")
    manage_btn.pack(side="left", padx=4)

    return model_combobox, model_name_label, status_light, test_btn, manage_btn


# ==============================================================================
# Code/Chat 模式切换按钮
# ==============================================================================

def create_mode_toggle(parent):
    """模式切换按钮：💻 Code（蓝色）←→ 💬 Chat（绿色）"""
    container = tk.Frame(parent, bg="#FFFFFF")
    container.pack(side="left", padx=8, pady=6)

    ttk.Label(container, text="模式：", font=FONT_BOLD).pack(side="left")
    mode_btn = tk.Button(container, text="💻 Code", font=FONT_NORMAL,
                         bg="#1976D2", fg="white", relief="flat", padx=12, cursor="hand2",
                         width=12)
    mode_btn.pack(side="left", padx=4)
    return mode_btn


# ==============================================================================
# 功能性容器 + 隔离条
# ==============================================================================

def create_func_frame(root):
    """第二排功能区容器"""
    container = tk.Frame(root, bg="#FFFFFF")
    container.pack(fill="x", padx=0, pady=0)
    return container


def create_main_paned(root):
    """
    左右分栏容器（PanedWindow）。
    用户可以拖动分栏分隔线调整聊天区和日志区的比例。
    """
    main_paned = ttk.PanedWindow(root, orient="horizontal")
    main_paned.pack(fill="both", expand=True, padx=0, pady=0)
    return main_paned


# ==============================================================================
# 版本回退控件
# ==============================================================================

def create_rollback_widgets(parent):
    """版本回退：标签 + 输入框 + 选择/回退/历史按钮"""
    container = tk.Frame(parent, bg="#FFFFFF")
    container.pack(side="left", padx=8, pady=2)

    ttk.Label(container, text="回退：", font=FONT_BOLD).pack(side="left")
    entry = ttk.Entry(container, width=28, font=FONT_NORMAL)
    entry.pack(side="left", padx=4)

    select_btn = tk.Button(container, text="📁 选择", font=FONT_NORMAL,
                          bg="#F5F5F5", relief="flat", padx=6, cursor="hand2")
    select_btn.pack(side="left", padx=2)

    rollback_btn = tk.Button(container, text="↩ 回退", font=FONT_NORMAL,
                            bg="#F5F5F5", relief="flat", padx=6, cursor="hand2")
    rollback_btn.pack(side="left", padx=2)

    history_btn = tk.Button(container, text="📋 历史", font=FONT_NORMAL,
                           bg="#F5F5F5", relief="flat", padx=6, cursor="hand2")
    history_btn.pack(side="left", padx=2)

    return entry, select_btn, rollback_btn, history_btn


def create_top_frame(root):
    """旧接口兼容：create_top_frame → create_top_toolbar"""
    return create_top_toolbar(root)


# ==============================================================================
# 聊天区头部
# ==============================================================================

def create_chat_header(parent, title="对话", clear_command=None):
    """聊天区标题栏 + 清空按钮"""
    header = tk.Frame(parent, bg="white", height=36)
    header.pack(fill="x")
    tk.Label(header, text=f"💬 {title}", font=FONT_BOLD,
            bg="white", fg="#212121").pack(side="left", padx=12, pady=6)
    btn = tk.Button(header, text="🗑 清空", font=("Microsoft YaHei", 9),
                   bg="#F5F5F5", relief="flat", padx=8, cursor="hand2",
                   command=clear_command if clear_command else lambda: None)
    btn.pack(side="right", padx=8, pady=4)
    return btn


# ==============================================================================
# ScrollableChatFrame —— 可滚动聊天容器（Canvas + Frame + 鼠标滚轮）
# ==============================================================================
# Tkinter 的 Frame 本身不可滚动。通过在 Canvas 中嵌入 Frame，
# 利用 Canvas 的滚动能力实现虚拟滚动聊天区域。
class ScrollableChatFrame(tk.Frame):
    """
    可滚动聊天框：
    1. Canvas 创建滚动区域
    2. scrollable_frame 是实际的聊天块父容器
    3. 鼠标事件处理滚轮滚动（Enter 绑定，Leave 解绑）
    4. add_block 后自动滚动到底部
    """
    def __init__(self, parent, bg=CHAT_BG):
        super().__init__(parent, bg=bg)
        self.inner_bg = bg

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, relief="flat")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        self.scrollable_frame = tk.Frame(self.canvas, bg=bg)
        # 当 scrollable_frame 大小变化时，更新 Canvas 的滚动区域
        self.scrollable_frame.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Configure>", self._on_canvas_configure)     # 宽度自适应
        self.canvas.bind("<Enter>", self._bind_mousewheel)             # 鼠标进入 → 绑定滚轮
        self.canvas.bind("<Leave>", self._unbind_mousewheel)           # 鼠标离开 → 解绑滚轮

    def _on_canvas_configure(self, event):
        """Canvas 宽度变化时，让内部 frame 宽度同步"""
        self.canvas.itemconfig(self.canvas_window, width=event.width - 4)

    def _bind_mousewheel(self, _event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        """Windows 鼠标滚轮：event.delta 为正=向上滚，为负=向下滚"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def clear(self):
        """清空所有聊天块"""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def add_block(self, block):
        """添加消息块并自动滚动到底部"""
        block.pack(fill="x", padx=0, pady=0)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """滚动到最底部"""
        self.canvas.yview_moveto(1.0)   # 1.0 = 100% = 最底部
        self.update_idletasks()


# ==============================================================================
# 消息块基类 —— BaseBlock
# ==============================================================================
class BaseBlock(tk.Frame):
    """
    所有聊天块的基类。
    内嵌一个 inner Frame（带背景色、内边距），实际内容在 inner 中。
    """
    def __init__(self, parent, bg, **kwargs):
        super().__init__(parent, bg=CHAT_BG)
        self.inner = tk.Frame(self, bg=bg, padx=14, pady=10)
        self.inner.pack(fill="x", padx=16, pady=6)


# ==============================================================================
# UserBlock —— 用户消息块（浅蓝色）
# ==============================================================================
class UserBlock(BaseBlock):
    def __init__(self, parent, message):
        super().__init__(parent, bg=BLOCK_USER_BG)
        header = tk.Frame(self.inner, bg=BLOCK_USER_BG)
        tk.Label(header, text="🧑 你", font=FONT_BOLD,
                bg=BLOCK_USER_BG, fg=BLOCK_USER_FG).pack(side="left")
        header.pack(fill="x", pady=(0, 6))

        # 自适应高度：至少2行，最多8行
        txt = tk.Text(self.inner, wrap="word", font=FONT_NORMAL,
                      bg=BLOCK_USER_BG, fg="#212121", relief="flat",
                      height=max(2, min(8, message.count("\n") + 1)),
                      padx=4, pady=2, highlightthickness=0)
        txt.insert("1.0", message)
        txt.config(state="disabled")
        txt.pack(fill="x")


# ==============================================================================
# AITextBlock —— AI 文本消息块（浅紫色）
# ==============================================================================
class AITextBlock(BaseBlock):
    def __init__(self, parent, message):
        super().__init__(parent, bg=BLOCK_AI_BG)
        header = tk.Frame(self.inner, bg=BLOCK_AI_BG)
        tk.Label(header, text="🤖 AI 助手", font=FONT_BOLD,
                bg=BLOCK_AI_BG, fg=BLOCK_AI_FG).pack(side="left")
        header.pack(fill="x", pady=(0, 6))

        txt = tk.Text(self.inner, wrap="word", font=FONT_NORMAL,
                      bg=BLOCK_AI_BG, fg="#212121", relief="flat",
                      height=max(2, min(15, message.count("\n") + 1)),
                      padx=4, pady=2, highlightthickness=0)
        txt.insert("1.0", message)
        txt.config(state="disabled")
        txt.pack(fill="x")


# ==============================================================================
# SystemBlock —— 系统消息块（浅橙色）
# ==============================================================================
class SystemBlock(BaseBlock):
    def __init__(self, parent, message):
        super().__init__(parent, bg=BLOCK_SYSTEM_BG)
        header = tk.Frame(self.inner, bg=BLOCK_SYSTEM_BG)
        tk.Label(header, text="🔔 系统", font=FONT_BOLD,
                bg=BLOCK_SYSTEM_BG, fg=BLOCK_SYSTEM_FG).pack(side="left")
        header.pack(fill="x", pady=(0, 6))

        txt = tk.Text(self.inner, wrap="word", font=("Microsoft YaHei", 10, "italic"),
                      bg=BLOCK_SYSTEM_BG, fg=BLOCK_SYSTEM_FG, relief="flat",
                      height=max(1, min(5, message.count("\n") + 1)),
                      padx=4, pady=2, highlightthickness=0)
        txt.insert("1.0", message)
        txt.config(state="disabled")
        txt.pack(fill="x")


# ==============================================================================
# DiffBlock —— 文件修改对比 & 接受/拒绝（核心交互组件）
# ==============================================================================
# 当 Agent 修改了文件，handler._detect_file_changes 检测到变更后，
# 生成 DiffBlock 展示差异，用户可逐个接受或拒绝。
class DiffBlock(tk.Frame):
    def __init__(self, parent, file_path, old_content, new_content,
                 on_accept=None, on_reject=None):
        super().__init__(parent, bg=CHAT_BG)
        self.status = "pending"        # pending → accepted / rejected
        self.file_path = file_path
        self.old_content = old_content
        self.new_content = new_content

        # 白色卡片容器
        card = tk.Frame(self, bg="white", highlightbackground=BLOCK_DIFF_BORDER,
                       highlightthickness=1)
        card.pack(fill="x", padx=16, pady=6)

        self._build_header(card)       # 文件名 + 类型标签
        self._build_diff_view(card)    # unified diff 视图
        self._build_actions(card)      # 接受/拒绝按钮

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=BLOCK_DIFF_HEADER_BG)
        tk.Label(header, text=f"📄 {self.file_path}", font=FONT_BOLD,
                bg=BLOCK_DIFF_HEADER_BG, fg="#212121").pack(side="left", padx=12, pady=8)
        tk.Label(header, text="文件修改建议", font=("Microsoft YaHei", 9),
                bg=BLOCK_DIFF_HEADER_BG, fg="#9E9E9E").pack(side="left", padx=4, pady=8)
        header.pack(fill="x")

    def _build_diff_view(self, parent):
        """使用 difflib.unified_diff 生成 diff 文本，按行着色"""
        old_lines = self.old_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)

        # 生成 unified diff（上下文行数 3）
        diff = list(difflib.unified_diff(old_lines, new_lines, n=3))
        # 去掉文件头（--- 和 +++ 行）
        diff = [l for l in diff if not l.startswith("---") and not l.startswith("+++")]
        if not diff:
            diff = [f"  {l.rstrip()}" for l in old_lines]

        diff_frame = tk.Frame(parent, bg="white")
        txt = tk.Text(diff_frame, wrap="none", font=FONT_CODE, relief="flat",
                     padx=12, pady=8, height=min(20, len(diff) + 2),
                     highlightthickness=0, bg="white")

        for line in diff:
            if line.startswith("+"):
                txt.insert("end", line, ("add",))
            elif line.startswith("-"):
                txt.insert("end", line, ("del",))
            elif line.startswith("@@"):
                txt.insert("end", line, ("info",))
            else:
                txt.insert("end", line, ("normal",))
        txt.config(state="disabled")
        txt.tag_config("add", background=DIFF_ADD_BG, foreground=DIFF_ADD_FG)         # 绿色=新增
        txt.tag_config("del", background=DIFF_DEL_BG, foreground=DIFF_DEL_FG)         # 红色=删除
        txt.tag_config("info", foreground="#1976D2")                                   # 蓝色=行号信息
        txt.tag_config("normal", foreground="#424242")                                  # 灰色=上下文
        txt.pack(fill="both", expand=True)
        diff_frame.pack(fill="x")
        self.diff_text = txt

    def _build_actions(self, parent):
        """接受/拒绝按钮 + 状态标签"""
        action_frame = tk.Frame(parent, bg="white")

        self.status_label = tk.Label(action_frame, text="⏳ 待处理", font=("Microsoft YaHei", 9),
                                     bg="white", fg=STATUS_PENDING_FG)
        self.status_label.pack(side="right", padx=12, pady=6)

        self.accept_btn = tk.Button(action_frame, text="✓  接受修改", font=("Microsoft YaHei", 9),
                                    bg=BUTTON_ACCEPT_BG, fg=BUTTON_ACCEPT_FG,
                                    relief="flat", padx=14, pady=4, cursor="hand2")
        self.accept_btn.pack(side="left", padx=(12, 4), pady=6)

        self.reject_btn = tk.Button(action_frame, text="✗  拒绝修改", font=("Microsoft YaHei", 9),
                                    bg=BUTTON_REJECT_BG, fg=BUTTON_REJECT_FG,
                                    relief="flat", padx=14, pady=4, cursor="hand2")
        self.reject_btn.pack(side="left", padx=4, pady=6)

        action_frame.pack(fill="x")

    def _disable_buttons(self):
        """操作完成后禁用按钮（防止重复操作）"""
        self.accept_btn.config(state="disabled", bg=BUTTON_DISABLED_BG)
        self.reject_btn.config(state="disabled", bg=BUTTON_DISABLED_BG)

    def _on_accept(self):
        if self.status != "pending": return
        self.status = "accepted"
        self.status_label.config(text="✓ 已接受", fg=STATUS_ACCEPTED_FG)
        self._disable_buttons()

    def _on_reject(self):
        if self.status != "pending": return
        self.status = "rejected"
        self.status_label.config(text="✗ 已拒绝", fg=STATUS_REJECTED_FG)
        self._disable_buttons()

    def set_accept_callback(self, callback):
        """设置接受按钮的回调（_on_accept 先执行 → 然后 callback）"""
        self.accept_btn.config(command=lambda: [self._on_accept(), callback()])

    def set_reject_callback(self, callback):
        """设置拒绝按钮的回调"""
        self.reject_btn.config(command=lambda: [self._on_reject(), callback()])


# ==============================================================================
# 日志面板
# ==============================================================================

def create_log_frame(parent):
    """右侧日志面板：带滚动条，支持彩色 tag 渲染"""
    frame = ttk.LabelFrame(parent, text="执行日志", padding="4")
    log_text = scrolledtext.ScrolledText(
        frame, wrap="word", state="disabled",
        font=FONT_LOG, bg="#F5F5F5", height=8
    )
    log_text.pack(fill="both", expand=True)
    # 注册日志 tag 样式（颜色 + 字体）
    for tag, cfg in LOG_TAGS.items():
        log_text.tag_config(tag, **cfg)
    return frame, log_text


# ==============================================================================
# 底部输入面板
# ==============================================================================

def create_input_frame(root):
    """底部输入栏：提交文件按钮 + 输入框 + 发送按钮"""
    container = tk.Frame(root, bg="#FFFFFF")
    container.pack(fill="x")
    ttk.Separator(root, orient="horizontal").pack(fill="x")   # 顶部分隔线

    inner = tk.Frame(container, bg="#FFFFFF", padx=16, pady=10)
    inner.pack(fill="x")

    submit_file_btn = tk.Button(inner, text="📎 提交文件", font=FONT_NORMAL,
                                bg="#F5F5F5", relief="flat", padx=10, cursor="hand2")
    submit_file_btn.pack(side="left", padx=(0, 8))

    # 输入框：fill="x" expand=True 占据中间所有剩余空间
    input_entry = ttk.Entry(inner, font=("Microsoft YaHei", 11))
    input_entry.pack(side="left", padx=4, fill="x", expand=True, ipady=4)

    send_btn = tk.Button(inner, text="发送 ➤", font=FONT_BOLD,
                         bg="#1976D2", fg="white", relief="flat",
                         padx=16, pady=4, cursor="hand2")
    send_btn.pack(side="left", padx=(8, 0))

    return container, input_entry, submit_file_btn, send_btn


# ==============================================================================
# 工作区信息面板（右侧标签页）
# ==============================================================================

def create_workspace_info_panel(parent):
    """右侧标签页容器：工作区信息 + 日志（日志由外部 add）"""
    notebook = ttk.Notebook(parent)
    notebook.pack(fill="both", expand=True)

    info_frame = tk.Frame(notebook, bg="white")
    notebook.add(info_frame, text="📋 工作区")

    info_text = scrolledtext.ScrolledText(
        info_frame, wrap="word", state="disabled",
        font=("Consolas", 9), bg="white", fg="#424242",
        relief="flat", highlightthickness=0
    )
    info_text.pack(fill="both", expand=True, padx=4, pady=4)

    return notebook, info_text
