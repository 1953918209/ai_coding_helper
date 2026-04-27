"""
================================================================================
UI 业务逻辑层 —— 桌面应用的核心控制器
================================================================================

【MVC 架构中的角色】
本模块是 MVC 中的 Controller，负责：
  - 接收用户输入 → 调用 Agent → 展示结果
  - 管理 Agent 生命周期（创建/切换/销毁）
  - 文件变更监听（watchdog）→ RAG 索引实时同步
  - 工作区管理（设置/切换/临时工作区）
  - Code/Chat 模式切换
  - DiffBlock 审阅 → 自动继续机制
  - RAG 增强：每次发送消息前自动检索相关文件

【关键设计模式】
1. StdoutRedirector：print() 重定向到 UI 日志面板
2. FileChangeHandler（watchdog）：监听文件系统变更，实时更新 RAG 索引
3. 快照+DiffBlock：Agent 执行前后对比文件，生成可视化 diff 卡片
4. 闭包 lambda：Tkinter UI 线程与工作线程之间的桥接（root.after）
5. threading.Timer 滑动窗口：每次 stream yield 重置 600s 超时计时器

【线程模型】
- UI 主线程：Tkinter 事件循环，处理用户交互
- Agent 工作线程：_run_agent_stream 在独立线程中执行，通过 root.after 更新 UI
- 索引构建线程：rag_manager.build_full_index_async 后台构建，不阻塞 UI
- Watchdog 线程：文件系统监听

【面试高频题 —— MVC + GUI + 多线程】
Q17: Tkinter 中如何处理多线程？为什么不能直接在工作线程更新 UI？
A:  Tkinter 不是线程安全的，所有 UI 操作必须在主线程（事件循环线程）执行。
    本项目使用 root.after(0, callback) 将 UI 更新任务调度回主线程。
    after(0) 表示"尽快但不阻塞当前线程"，实际是在主线程空闲时立即执行。

Q18: StdoutRedirector 是怎么工作的？
A:  替换 sys.stdout 为一个自定义类，其 write() 方法捕获所有 print() 输出，
    通过关键词过滤只显示重要日志，然后用 root.after() 安全地更新 UI。

Q19: watchdog 防抖是什么？为什么需要它？
A:  文件保存时可能触发多次 modify 事件（IDE 自动保存、临时文件写入等）。
    2秒防抖：每次变更重置计时器，只有2秒无新变更时才批量处理，
    避免频繁重建 RAG 索引造成性能浪费。

Q20: DiffBlock 的"自动继续"机制是什么？
A:  Agent 修改文件后生成 DiffBlock 让用户审阅。全部审阅完毕后：
    - 如果 Agent 已给出文本总结 → 任务完成
    - 如果 Agent 还有未完成工作 → 自动发送"继续执行剩余任务"指令
    这个机制让 Agent 和用户形成协作循环。
"""
import gc, os, uuid, threading, logging, sys, shutil
from tkinter import filedialog, messagebox
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from watchdog.observers import Observer                         # 文件系统监听
from watchdog.events import FileSystemEventHandler               # 文件变更事件处理
from ui.config import (
    SUPPORTED_MODELS, DEFAULT_MODEL, STATUS_LIGHTS, ALLOWED_EXT, FORBIDDEN_DIR, TEMP_WORKSPACE
)
from ui.widgets import ScrollableChatFrame, UserBlock, AITextBlock, SystemBlock, DiffBlock
from utils.version_manager import init_version_dir, get_file_history_versions, rollback_last_modify
from core.agent import create_agent_executor
from core.rag.rag_manager import init_rag_manager, clear_rag_manager, get_rag_manager

logger = logging.getLogger(__name__)


# ==============================================================================
# StdoutRedirector —— 命令行输出重定向到 UI 日志面板
# ==============================================================================
# Python 的 print() 和 logger 输出默认到终端，GUI 应用中看不到。
# 通过替换 sys.stdout，将所有 print 输出捕获到右侧日志面板。
# 关键词过滤：只显示 Agent 相关日志（模型思考/工具调用等），过滤掉不重要的噪音。
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout  # 保留原始 stdout，退出时恢复

    def write(self, message):
        if message.strip():
            # 关键词过滤：只显示特定类型的日志
            if any(key in message for key in [
                "【模型思考】", "【模型深度思考】", "上下文修剪",
                "模型请求", "模型响应", "【工具调用】"
            ]):
                # Tkinter 不是线程安全的，必须用 after() 回到主线程更新 UI
                self.text_widget.after(0, self._append_log, message.strip())

    def flush(self):
        pass   # 兼容文件对象接口

    def _append_log(self, message):
        self.text_widget.config(state="normal")
        self.text_widget.insert("end", f"{message}\n", "model")
        self.text_widget.see("end")
        self.text_widget.config(state="disabled")


# ==============================================================================
# FileChangeHandler —— 文件系统变更监听（基于 watchdog）
# ==============================================================================
# 监听工作区目录的文件创建/修改/删除事件。
# 2秒防抖：短时间内的多次变更合并为一次处理。
# 变更后自动同步 RAG 索引，确保向量检索数据始终与磁盘一致。
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, app_handler):
        self.app_handler = app_handler
        self.workspace = app_handler.workspace
        self.pending_changes = {}     # 待处理的变更 {rel_path: "created"/"modified"/"deleted"}
        self.debounce_delay = 2       # 防抖延迟（秒）
        self.timer = None

    def _is_allowed_file(self, path):
        """检查文件是否在允许索引的范围内"""
        for forbidden in FORBIDDEN_DIR:
            if forbidden in path:
                return False
        ext = os.path.splitext(path)[-1].lower()
        return ext in ALLOWED_EXT

    def _get_rel_path(self, abs_path):
        try:
            rel = os.path.relpath(abs_path, self.workspace)
            if rel.startswith(".."): return None
            return rel
        except ValueError:
            return None

    def _process_pending_changes(self):
        """处理累积的变更（防抖触发）"""
        changes = self.pending_changes.copy()
        self.pending_changes.clear()
        for rel_path, change_type in changes.items():
            if change_type == "created":
                self.app_handler.handle_file_create(rel_path)
            elif change_type == "modified":
                self.app_handler.handle_file_modify(rel_path)
            elif change_type == "deleted":
                self.app_handler.handle_file_delete(rel_path)

    def _debounce(self, rel_path, change_type):
        """防抖：新变更重置计时器，2秒无新变更才处理"""
        self.pending_changes[rel_path] = change_type
        if self.timer: self.timer.cancel()
        self.timer = threading.Timer(self.debounce_delay, self._process_pending_changes)
        self.timer.start()

    def on_created(self, event):
        if event.is_directory: return
        rel_path = self._get_rel_path(event.src_path)
        if rel_path is None or not self._is_allowed_file(event.src_path): return
        self.app_handler.root.after(0, lambda: self._debounce(rel_path, "created"))

    def on_modified(self, event):
        if event.is_directory: return
        rel_path = self._get_rel_path(event.src_path)
        if rel_path is None or not self._is_allowed_file(event.src_path): return
        self.app_handler.root.after(0, lambda: self._debounce(rel_path, "modified"))

    def on_deleted(self, event):
        if event.is_directory: return
        rel_path = self._get_rel_path(event.src_path)
        if rel_path is None or not self._is_allowed_file(event.src_path): return
        self.app_handler.root.after(0, lambda: self._debounce(rel_path, "deleted"))


# ==============================================================================
# TempFileHandler —— 临时文件监听
# ==============================================================================
class TempFileHandler(FileSystemEventHandler):
    def __init__(self, app_handler, temp_file_path):
        self.app_handler = app_handler
        self.temp_file_path = os.path.abspath(temp_file_path)

    def on_modified(self, event):
        if not event.is_directory and os.path.abspath(event.src_path) == self.temp_file_path:
            self.app_handler.root.after(0, lambda: self.app_handler.handle_temp_file_modify(self.temp_file_path))

    def on_deleted(self, event):
        if not event.is_directory and os.path.abspath(event.src_path) == self.temp_file_path:
            self.app_handler.root.after(0, lambda: self.app_handler.handle_temp_file_delete(self.temp_file_path))


# ==============================================================================
# AppHandlers —— 主控制器类
# ==============================================================================
class AppHandlers:
    """应用的主控制器：管理 Agent 生命周期 + UI 事件处理 + 文件监听"""

    def __init__(self, root, widgets, config_manager=None):
        self.root = root
        self.widgets = widgets
        self.config_manager = config_manager      # 模型配置管理器
        self.workspace = ""
        self.sid = ""                              # 会话ID（UUID），用作 LangGraph thread_id
        self.agent = None                          # LangChain Agent 实例
        self.checkpointer = None                   # MemorySaver 检查点
        self.agent_config = None                   # RunnableConfig（含 thread_id）
        self.selected_model = DEFAULT_MODEL
        self.model_test_passed = False
        self.chat_mode = False                     # Code模式 vs Chat 模式
        self.temp_files = set()                    # 临时外部文件集合
        self.temp_file_contents = {}               # 临时文件内容缓存
        self.temp_observers = []                   # 临时文件监听器列表
        self._unpack_widgets()                     # 从 dict 拆解 UI 组件引用
        self._bind_events()                        # 绑定按钮事件
        self.stdout_redirector = StdoutRedirector(self.log_text)
        sys.stdout = self.stdout_redirector         # 全局 print 重定向
        self.observer = Observer()
        self.file_handler = None
        self.processed_files = set()

    def _unpack_widgets(self):
        """从 widgets 字典中提取 UI 组件引用"""
        self.workspace_entry = self.widgets["workspace_entry"]
        self.model_combobox = self.widgets["model_combobox"]
        self.model_name_label = self.widgets["model_name_label"]
        self.status_light = self.widgets["status_light"]
        self.model_manage_btn = self.widgets.get("model_manage_btn")
        self.mode_toggle_btn = self.widgets.get("mode_toggle_btn")
        self.rollback_file_entry = self.widgets["rollback_file_entry"]
        self.chat_area = self.widgets["chat_area"]
        self.log_text = self.widgets["log_text"]
        self.input_entry = self.widgets["input_entry"]
        self.send_btn = self.widgets["send_btn"]
        self.clear_log_btn = self.widgets["clear_log_btn"]
        self.clear_chat_btn = self.widgets["clear_chat_btn"]
        self.info_text = self.widgets.get("info_text")

    def _bind_events(self):
        """绑定 UI 事件到处理方法"""
        self.widgets["select_workspace_btn"]["command"] = self._select_workspace
        self.widgets["set_workspace_btn"]["command"] = self._set_workspace
        self.widgets["clear_memory_btn"]["command"] = self._clear_memory
        self.model_combobox.bind("<<ComboboxSelected>>", self._on_model_selected)
        self.widgets["test_model_btn"]["command"] = self._test_model_connection
        self.widgets["select_rollback_file_btn"]["command"] = self._select_rollback_file
        self.widgets["rollback_btn"]["command"] = self._rollback_last_version
        self.widgets["history_btn"]["command"] = self._show_history_versions
        self.input_entry.bind("<Return>", lambda event: self._send_message() or "break")
        self.widgets["send_btn"]["command"] = self._send_message
        self.clear_log_btn["command"] = self._clear_log
        self.clear_chat_btn["command"] = self._clear_chat
        if self.model_manage_btn:
            self.model_manage_btn["command"] = self._open_model_management
        if self.mode_toggle_btn:
            self.mode_toggle_btn["command"] = self._toggle_mode
        self.widgets["submit_file_btn"]["command"] = self.handle_submit_file

    # ---- UI 辅助方法 ----
    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.config(state="disabled")

    def _clear_chat(self):
        self.chat_area.clear()
        self.chat_area.add_block(SystemBlock(self.chat_area.scrollable_frame, "聊天界面已清空"))
        logger.info("聊天界面已清空")

    def _unlock_input_controls(self):
        """解锁输入框和发送按钮（Agent 执行完成后调用）"""
        self.input_entry.delete(0, "end")
        self.input_entry.config(state="normal")
        self.send_btn.config(state="normal")
        self.widgets["submit_file_btn"].config(state="normal")
        self.input_entry.focus_set()

    def _generate_workspace_file_tree(self):
        """生成工作区文件树（缩进格式，注入到 system_prompt 中）"""
        if not self.workspace: return "无可用工作区"
        file_tree = []
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in FORBIDDEN_DIR]
            rel_root = os.path.relpath(root, self.workspace)
            level = rel_root.count(os.sep) if rel_root != "." else 0
            indent = "  " * level
            if rel_root != ".":
                file_tree.append(f"{indent}{os.path.basename(root)}/")
            sub_indent = "  " * (level + 1)
            for file in files:
                if os.path.splitext(file)[-1].lower() in ALLOWED_EXT:
                    file_tree.append(f"{sub_indent}{file}")
        return "\n".join(file_tree) if file_tree else "工作区内无代码文件"

    # ---- 工作区管理 ----
    def _select_workspace(self):
        """选择工作区文件夹"""
        path = filedialog.askdirectory(title="选择工作区文件夹", initialdir=os.path.expanduser("~"))
        if path:
            self.workspace_entry.delete(0, "end")
            self.workspace_entry.insert(0, path)

    def _set_workspace(self):
        """
        设置工作区：验证路径 → 初始化RAG管理器 → 异步构建索引 → 创建Agent实例 → 启动文件监听。

        这是应用的核心初始化流程，每切换一次工作区都会重新执行。
        """
        path = self.workspace_entry.get().strip()
        if not path:
            messagebox.showerror("错误", "请先选择工作区文件夹！"); return
        if not os.path.exists(path):
            messagebox.showerror("错误", f"工作区路径不存在"); return
        if not os.path.isdir(path):
            messagebox.showerror("错误", "请选择有效的目录"); return
        try:
            old_ws = self.workspace
            new_ws_abs = os.path.abspath(path)
            self.workspace = path
            init_version_dir(self.workspace)
            model_cfg = SUPPORTED_MODELS.get(self.selected_model, {})
            if not model_cfg:
                messagebox.showerror("错误", "当前选中的模型不存在"); return

            clear_rag_manager()
            init_rag_manager(self.workspace)
            rag_mgr = get_rag_manager()

            # 处理临时文件：如果在新的工作区内 → 删除重复索引；否则保留
            if self.temp_files:
                for temp_file_abs in self.temp_files:
                    temp_file_abs = os.path.abspath(temp_file_abs)
                    if temp_file_abs.startswith(new_ws_abs):
                        rag_mgr.remove_file_from_index(temp_file_abs)
                        self.temp_files.discard(temp_file_abs)

            rag_mgr.set_index_callbacks(
                on_complete=lambda total: self.root.after(0, lambda:
                    self._append_chat_message("系统", f"代码索引构建完成：{total}个符号")),
                on_failed=lambda error_msg: self.root.after(0, lambda:
                    self._append_chat_message("系统", f"索引构建失败：{error_msg}"))
            )
            self._append_chat_message("系统", "正在后台重建代码索引...")
            rag_mgr.build_full_index_async()

            # 创建 Agent 实例
            file_tree = self._generate_workspace_file_tree()
            self.sid = str(uuid.uuid4())
            self.agent, self.checkpointer, self.agent_config = create_agent_executor(
                self.selected_model, self.workspace, self.sid, model_cfg, self.root, file_tree
            )
            self._append_chat_message("系统", f"工作区已设置 | 会话ID：{self.sid} | 模型：{model_cfg['name']}")

            # 启动文件系统监听（watchdog）
            if hasattr(self, 'observer') and self.observer and self.observer.is_alive():
                self.observer.stop(); self.observer.join(timeout=2)
            self.observer = Observer()
            self.file_handler = FileChangeHandler(self)
            self.observer.schedule(self.file_handler, path=self.workspace, recursive=True)
            self.observer.start()

            # 临时工作区迁移逻辑
            if old_ws == TEMP_WORKSPACE and self.temp_files:
                for fp in self.temp_files:
                    try: rag_mgr.add_file_to_index(fp)
                    except Exception: pass
                if os.path.exists(TEMP_WORKSPACE): shutil.rmtree(TEMP_WORKSPACE, ignore_errors=True)
        except Exception as e:
            messagebox.showerror("错误", f"设置工作区失败：{str(e)}")

    def _clear_memory(self):
        """
        清理会话记忆：销毁 Agent 实例 → 清理临时文件索引 → 清空聊天区。
        RAG 索引和版本历史保留。
        """
        if not self.sid:
            messagebox.showinfo("提示", "当前无会话记忆！"); return
        try:
            if self.observer.is_alive(): self.observer.stop(); self.observer.join(timeout=2)
            for obs in self.temp_observers:
                if obs.is_alive(): obs.stop(); obs.join(timeout=2)
            self.temp_observers.clear()
            self.agent = None; self.checkpointer = None; self.agent_config = None
            self.sid = ""; self.processed_files = set()
            if self.temp_files:
                rag_manager = get_rag_manager()
                if rag_manager:
                    for temp_file in self.temp_files: rag_manager.remove_file_from_index(temp_file)
                self.temp_files.clear()
            self.temp_file_contents.clear(); self.chat_area.clear()
            self._append_chat_message("系统", "会话记忆已清理（RAG索引+版本历史保留）")
        except Exception as e:
            messagebox.showerror("错误", f"清理记忆失败：{str(e)}")

    # ---- 模型管理 ----
    def _on_model_selected(self, _):
        self.selected_model = self.model_combobox.get()
        model_info = SUPPORTED_MODELS.get(self.selected_model, {})
        self.model_name_label.config(text=model_info.get("name", self.selected_model))
        self.model_test_passed = False
        self.status_light.config(text=STATUS_LIGHTS["untested"])
        self._append_chat_message("系统", f"已选择模型：{model_info.get('name', self.selected_model)}\n请点击测试连通性")

    def _test_model_connection(self):
        """测试模型 API 连通性：发送"你好" → 确认回复含"ok" → 标记通过"""
        model_info = SUPPORTED_MODELS.get(self.selected_model, {})
        if not model_info: messagebox.showerror("错误", "当前模型不存在"); return
        if not model_info.get("api_key") or "你的" in model_info.get("api_key", ""):
            messagebox.showerror("错误", "请先在模型管理中配置API_KEY"); return
        self.widgets["test_model_btn"].config(state="disabled")
        self.model_combobox.config(state="disabled")
        self.status_light.config(text=STATUS_LIGHTS["testing"])
        self._append_chat_message("系统", f"正在测试 [{model_info.get('name')}] 连通性...")

        def test_task():
            try:
                model_config = SUPPORTED_MODELS[self.selected_model]
                llm = ChatOpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"],
                                 model=model_config["model_name"], temperature=0.1, timeout=30, max_retries=2)
                res = llm.invoke("你好，仅回复ok即可")
                if res and res.content and "ok" in res.content.lower():
                    self.root.after(0, self._on_test_success)
                else:
                    raise Exception(f"模型返回异常：{res.content if res else '空响应'}")
            except Exception as e:
                self.root.after(0, lambda: self._on_test_fail(str(e)))

        threading.Thread(target=test_task, daemon=True).start()

    def _on_test_success(self):
        self.model_test_passed = True
        self.status_light.config(text=STATUS_LIGHTS["success"])
        self.widgets["test_model_btn"].config(state="normal")
        self.model_combobox.config(state="readonly")
        model_info = SUPPORTED_MODELS.get(self.selected_model, {})
        self._append_chat_message("系统", f"[{model_info.get('name', self.selected_model)}] 连通性测试通过")
        # 自动重建 Agent（保留原会话ID，记忆不丢）
        if self.workspace and self.sid:
            model_cfg = SUPPORTED_MODELS[self.selected_model]
            file_tree = self._generate_workspace_file_tree()
            self.agent, self.checkpointer, self.agent_config = create_agent_executor(
                self.selected_model, self.workspace, self.sid, model_cfg, self.root, file_tree
            )
            self._append_chat_message("系统", "智能体重建完成，后续请求将使用新模型")

    def _on_test_fail(self, error_msg):
        self.model_test_passed = False
        self.status_light.config(text=STATUS_LIGHTS["fail"])
        self.widgets["test_model_btn"].config(state="normal")
        self.model_combobox.config(state="readonly")
        self._append_chat_message("系统", f"模型测试失败：{error_msg}")

    # ---- Code/Chat 模式切换 ----
    def _toggle_mode(self):
        self.chat_mode = not self.chat_mode
        if self.chat_mode:
            self.mode_toggle_btn.config(text="Chat", bg="#4CAF50", fg="white")
            self._append_chat_message("系统", "已切换到聊天模式")
        else:
            self.mode_toggle_btn.config(text="Code", bg="#1976D2", fg="white")
            self._append_chat_message("系统", "已切换到代码模式")

    def _open_model_management(self):
        """打开模型管理对话框"""
        if not self.config_manager:
            messagebox.showwarning("提示", "模型管理器未初始化"); return
        from ui.model_manager import ModelManageDialog
        ModelManageDialog(self.root, self.config_manager, on_changed=self._on_models_changed)

    def _on_models_changed(self):
        self._refresh_model_combobox()

    def _refresh_model_combobox(self):
        """模型配置变更后刷新下拉列表"""
        from ui.config import SUPPORTED_MODELS, DEFAULT_MODEL
        current = self.model_combobox.get()
        keys = list(SUPPORTED_MODELS.keys())
        self.model_combobox["values"] = keys
        if current in keys: self.model_combobox.set(current)
        elif DEFAULT_MODEL in keys: self.model_combobox.set(DEFAULT_MODEL)
        elif keys: self.model_combobox.set(keys[0])
        self._on_model_selected(None)

    # ---- 版本回退 ----
    def _select_rollback_file(self):
        if not self.workspace: messagebox.showwarning("提示", "请先设置工作区"); return
        file_path = filedialog.askopenfilename(title="选择要回退的代码文件", initialdir=self.workspace)
        if file_path:
            rel_path = os.path.relpath(file_path, self.workspace)
            if rel_path.startswith(".."): messagebox.showerror("错误", "文件不在工作区内"); return
            self.rollback_file_entry.delete(0, "end")
            self.rollback_file_entry.insert(0, rel_path)

    def _rollback_last_version(self):
        rel_path = self.rollback_file_entry.get().strip()
        if not rel_path or not self.workspace: return
        full_path = os.path.join(self.workspace, rel_path)
        success, msg = rollback_last_modify(full_path, self.workspace)
        if success:
            rag_manager = get_rag_manager()
            if rag_manager: rag_manager.update_file_index(rel_path)
            self._append_chat_message("系统", f"文件[{rel_path}]回退成功")
        else:
            self._append_chat_message("系统", f"文件回退失败：{msg}")

    def _show_history_versions(self):
        rel_path = self.rollback_file_entry.get().strip()
        if not rel_path or not self.workspace: return
        full_path = os.path.join(self.workspace, rel_path)
        history = get_file_history_versions(full_path, self.workspace)
        if not history: messagebox.showinfo("提示", "该文件无历史版本"); return
        history_msg = f"文件[{rel_path}]的历史版本：\n"
        for i, item in enumerate(history):
            history_msg += f"\n{i + 1}. {item['create_time']}"
        self._append_chat_message("系统", history_msg)

    # ==========================================================================
    # _send_message —— 发送消息的核心入口
    # ==========================================================================
    # 流程：
    #   1. 验证前提条件（工作区 + Agent + 索引就绪）
    #   2. 组装增强 prompt（原始问题 + RAG检索结果 + 临时文件内容）
    #   3. 锁定输入控件 → 拍摄文件快照 → 启动 Agent 工作线程
    def _send_message(self):
        user_prompt = self.input_entry.get().strip()
        if not user_prompt and not self.temp_file_contents: return
        if not self.workspace or not self.sid or not self.agent:
            messagebox.showwarning("提示", "请先设置工作区"); return
        rag_mgr = get_rag_manager()
        if rag_mgr and not rag_mgr.is_index_ready():
            self._append_chat_message("系统", "代码索引正在构建中，请稍候"); return

        self._agent_already_summarized = False

        # 组装临时文件内容
        model_prompt = user_prompt
        if self.temp_file_contents:
            model_prompt += "\n\n【用户提交的文件内容】"
            for fp, content in self.temp_file_contents.items():
                model_prompt += f"\n\n文件：{fp}\n{content}"
            self.temp_file_contents.clear()

        display_message = user_prompt
        if self.temp_files:
            display_message += f"\n\n已附加 {len(self.temp_files)} 个外部文件"

        self.input_entry.delete(0, "end")
        self.input_entry.config(state="disabled")
        self.send_btn.config(state="disabled")
        self.widgets["submit_file_btn"].config(state="disabled")

        self._append_chat_message("用户", display_message)
        self._append_chat_message("系统", "正在处理您的指令，请稍候...")

        self._snapshot_workspace_files()   # 拍摄前快照
        self.processed_files = set()
        threading.Thread(target=self._run_agent_stream, args=(model_prompt,), daemon=True).start()

    # ==========================================================================
    # _run_agent_stream —— Agent 流式执行核心
    # ==========================================================================
    # 【LangGraph stream 模式】
    # agent.stream(input_state, stream_mode="values") 逐状态 yield。
    # 每个 state 包含完整的 messages 列表。
    # 取 state["messages"][-1] 就是最新的消息。
    #
    # 【超时机制（滑动窗口）】
    # 每次 stream yield 都重置 600s 计时器。
    # 如果 600s 内没有新 yield → TimeoutError。
    # 模型返回内容后自动停止计时（循环自然结束）。
    #
    # 【agent_gave_text_reply 标志】
    # 当 Agent 输出文本（而非 tool_calls）时设为 True。
    # 传给 DiffBlock 处理逻辑：如果 Agent 已文本总结 + 有文件修改
    # → DiffBlock 审阅后不自动继续（Agent 已经完成任务）。
    def _run_agent_stream(self, prompt):
        timeout_timer = [None]
        timeout_event = threading.Event()
        agent_gave_text_reply = False

        def _reset_timeout():
            if timeout_timer[0]: timeout_timer[0].cancel()
            timeout_event.clear()
            t = threading.Timer(600, timeout_event.set)
            t.daemon = True; timeout_timer[0] = t; t.start()

        def _cancel_timeout():
            if timeout_timer[0]: timeout_timer[0].cancel(); timeout_timer[0] = None
            timeout_event.clear()

        try:
            augmented_prompt = self._augment_prompt_with_rag(prompt)
            input_state = {"messages": [{"role": "user", "content": augmented_prompt}]}
            stream = self.agent.stream(input_state, config=self.agent_config, stream_mode="values")
            final_answer = ""; last_message_id = None; tool_call_detected = False
            for state in stream:
                _reset_timeout()
                if timeout_event.is_set(): raise TimeoutError("单次请求超时（600秒）")
                if "messages" not in state or len(state["messages"]) == 0: continue
                current_msg = state["messages"][-1]
                if id(current_msg) == last_message_id: continue
                last_message_id = id(current_msg)
                if isinstance(current_msg, AIMessage):
                    if hasattr(current_msg, "tool_calls") and current_msg.tool_calls:
                        tool_call_detected = True
                        for tc in current_msg.tool_calls:
                            print(f"【模型思考】调用工具：{tc.get('name', '未知')}")
                        continue
                    if current_msg.content and current_msg.content.strip():
                        final_answer = current_msg.content.strip()
                        agent_gave_text_reply = True   # Agent 输出了文本总结
                        print(f"【模型思考】{final_answer[:150]}")
                        continue
            if final_answer:
                self.root.after(0, lambda t=final_answer: self._append_chat_message("AI", t))
            else:
                if tool_call_detected:
                    self.root.after(0, lambda: self._append_chat_message("AI", "指令执行完成"))
                else:
                    self.root.after(0, lambda: self._append_chat_message("AI", "处理完成"))
        except Exception as e:
            self.root.after(0, lambda e=e: self._append_chat_message("系统", f"处理遇到问题：{e}，请重试"))
        finally:
            _cancel_timeout(); self.root.after(0, self._unlock_input_controls)
            try:
                changes = self._detect_file_changes()
                if changes:
                    # 传入 agent_gave_text_reply 判断是否已总结
                    self.root.after(0, lambda c=changes, d=agent_gave_text_reply:
                                    self._on_agent_changes_detected(c, d))
            except Exception as e:
                logger.exception(f"检测文件变化失败：{e}")

    def _append_chat_message(self, sender, message):
        """以块式 UI 方式添加聊天消息"""
        chat_parent = self.chat_area.scrollable_frame
        if sender == "用户":     block = UserBlock(chat_parent, message)
        elif sender == "AI":     block = AITextBlock(chat_parent, message)
        else:                    block = SystemBlock(chat_parent, message)
        self.chat_area.add_block(block)

    # ---- 文件快照 + DiffBlock 变更追踪 ----
    def _iter_workspace_files(self):
        if not self.workspace: return
        from utils.path_validator import iter_workspace_files
        yield from iter_workspace_files(self.workspace)

    def _snapshot_workspace_files(self):
        """Agent 执行前拍摄所有文件的快照（存内存）"""
        self._pre_run_snapshots = {}
        for rel_path, abs_path in self._iter_workspace_files():
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    self._pre_run_snapshots[rel_path] = f.read()
            except Exception: pass

    def _detect_file_changes(self):
        """Agent 执行后对比快照，找出所有变更的文件"""
        changes = []
        snapshots = getattr(self, '_pre_run_snapshots', None)
        if not snapshots: return changes
        for rel_path, abs_path in self._iter_workspace_files():
            old_content = snapshots.get(rel_path, "")
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    new_content = f.read()
            except Exception: continue
            if old_content != new_content:
                changes.append((rel_path, old_content, new_content))
        for rel_path in (set(snapshots.keys()) - {r for r, _ in self._iter_workspace_files()}):
            changes.append((rel_path, snapshots[rel_path], "(文件已被删除)"))
        self._pre_run_snapshots = None
        return changes

    def _on_agent_changes_detected(self, changes, agent_already_done=False):
        """生成 DiffBlock 卡片，设置接受/拒绝回调"""
        from functools import partial
        self._pending_diff_count = len(changes)
        self._agent_already_summarized = agent_already_done
        self._append_chat_message("系统", f"检测到 {len(changes)} 个文件修改建议：")
        for rel_path, old_content, new_content in changes:
            block = DiffBlock(self.chat_area.scrollable_frame, rel_path, old_content, new_content)
            block.set_accept_callback(partial(self._on_diff_accept, rel_path))
            block.set_reject_callback(partial(self._on_diff_reject, rel_path, old_content))
            self.chat_area.add_block(block)

    def _on_diff_accept(self, rel_path):
        self._pending_diff_count -= 1
        logger.info(f"用户已接受：{rel_path}")
        self._check_diff_all_done()

    def _on_diff_reject(self, rel_path, old_content):
        """拒绝修改：回滚文件到 Agent 执行前的内容"""
        self._pending_diff_count -= 1
        if not self.workspace: return
        abs_path = os.path.join(self.workspace, rel_path)
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f: f.write(old_content)
            rag_manager = get_rag_manager()
            if rag_manager: rag_manager.update_file_index(rel_path)
        except Exception as e:
            logger.exception(f"回滚文件失败：{rel_path}")
        self._check_diff_all_done()

    def _check_diff_all_done(self):
        """
        所有 DiffBlock 审阅完毕后的判断逻辑：
        - 如果 Agent 已经给出了文本总结 → 任务完成，不自动继续
        - 如果 Agent 还有未完成的工作 → 自动发送继续指令
        """
        if self._pending_diff_count > 0: return
        if getattr(self, '_agent_already_summarized', False):
            self._append_chat_message("系统", "任务完成，Agent 已给出总结")
            self._agent_already_summarized = False; return
        self.input_entry.config(state="disabled")
        self.send_btn.config(state="disabled")
        self.widgets["submit_file_btn"].config(state="disabled")
        self._snapshot_workspace_files()
        self.processed_files = set()
        continuation = (
            "你已经审阅了上一轮的修改结果。请严格按照以下规则继续：\n"
            "1. 首先判断任务是否已全部完成 —— 对照你在第一轮中列出的任务清单逐项检查\n"
            "2. 如果已全部完成：直接回复最终总结，不要再调用任何工具\n"
            "3. 如果还有未完成项：仅处理尚未完成的项，完成后立即总结\n"
            "4. 禁止重复执行已经完成的操作"
        )
        self._append_chat_message("系统", "所有修改已审阅完毕，正在继续执行剩余任务...")
        threading.Thread(target=self._run_agent_stream, args=(continuation,), daemon=True).start()

    # ---- 文件监听回调 ----
    @staticmethod
    def handle_file_create(rel_path):
        rag_manager = get_rag_manager()
        if rag_manager: rag_manager.add_file_to_index(rel_path)

    @staticmethod
    def handle_file_modify(rel_path):
        rag_manager = get_rag_manager()
        if rag_manager: rag_manager.update_file_index(rel_path)

    @staticmethod
    def handle_file_delete(rel_path):
        rag_manager = get_rag_manager()
        if rag_manager: rag_manager.remove_file_from_index(rel_path)

    # ---- 临时文件管理 ----
    def handle_temp_file_modify(self, file_path):
        rag_manager = get_rag_manager()
        if rag_manager and file_path in self.temp_files: rag_manager.update_file_index(file_path)

    def handle_temp_file_delete(self, file_path):
        rag_manager = get_rag_manager()
        if rag_manager and file_path in self.temp_files:
            rag_manager.remove_file_from_index(file_path); self.temp_files.discard(file_path)

    def start_temp_file_listener(self, file_path):
        observer = Observer()
        observer.schedule(TempFileHandler(self, os.path.abspath(file_path)),
                         os.path.dirname(os.path.abspath(file_path)), recursive=False)
        observer.start(); self.temp_observers.append(observer)

    def handle_submit_file(self):
        """提交外部文件到对话"""
        if not self.workspace:
            try:
                os.makedirs(TEMP_WORKSPACE, exist_ok=True); self.workspace = TEMP_WORKSPACE
                init_version_dir(self.workspace)
                model_cfg = SUPPORTED_MODELS.get(self.selected_model, {})
                if not model_cfg:
                    messagebox.showerror("错误", "当前选中的模型不存在"); return
                init_rag_manager(self.workspace); self.sid = str(uuid.uuid4())
                file_tree = self._generate_workspace_file_tree()
                self.agent, self.checkpointer, self.agent_config = create_agent_executor(
                    self.selected_model, self.workspace, self.sid, model_cfg, self.root, file_tree
                )
            except Exception as e: messagebox.showerror("错误", str(e)); return
        file_paths = filedialog.askopenfilenames(
            title="选择要提交的文件", initialdir=self.workspace,
            filetypes=[("代码文件", " ".join(f"*{ext}" for ext in ALLOWED_EXT))]
        )
        if not file_paths: return
        ws_abs = os.path.abspath(self.workspace)
        for fp in file_paths:
            try:
                if os.path.getsize(fp) > MAX_FILE_SIZE_BYTES:
                    messagebox.showwarning("警告", f"文件{os.path.basename(fp)}超50MB，已跳过"); continue
                with open(fp, "r", encoding="utf-8") as f: content = f.read()
                fp_abs = os.path.abspath(fp)
                if fp_abs.startswith(ws_abs):
                    self.temp_file_contents[os.path.relpath(fp_abs, ws_abs)] = content
                else:
                    self.temp_files.add(fp_abs); self.temp_file_contents[fp_abs] = content
                    rag_mgr = get_rag_manager()
                    if rag_mgr: rag_mgr.add_file_to_index(fp_abs)
                    self.start_temp_file_listener(fp_abs)
            except Exception as e: messagebox.showerror("错误", str(e))
        self._append_chat_message("系统", f"成功提交文件，内容将附加到下条指令")

    def on_closing(self):
        """应用退出时的清理工作"""
        sys.stdout = self.stdout_redirector.original_stdout
        if self.observer.is_alive(): self.observer.stop(); self.observer.join(timeout=2)
        for obs in self.temp_observers:
            if obs.is_alive(): obs.stop(); obs.join(timeout=2)
        clear_rag_manager()
        if os.path.exists(TEMP_WORKSPACE): shutil.rmtree(TEMP_WORKSPACE, ignore_errors=True)
        self.root.destroy()

    # ==========================================================================
    # _augment_prompt_with_rag —— RAG 增强（每次发送消息前调用）
    # ==========================================================================
    # Chat 模式 → 跳过 RAG，直接附加聊天提示
    # 全局查询（≥70%文件命中）→ 附加完整文件树 + 文件路径清单
    # 精准查询 → 附加 RAG 检索结果列表
    def _augment_prompt_with_rag(self, original_prompt: str) -> str:
        from core.rag.rag_tool import retrieve_related_files_structured
        from utils.path_validator import iter_workspace_files

        if self.chat_mode:
            return original_prompt + (
                "\n\n【系统提示】当前是聊天模式，请直接以对话方式回复，"
                "不要调用任何工具，不要提代码文件。"
            )

        file_count = sum(1 for _ in iter_workspace_files(self.workspace))
        augmented_content = "\n\n【系统自动补充信息】\n"
        try:
            results = retrieve_related_files_structured(original_prompt)
            if not results:
                augmented_content += "⚠️ 未检索到高度相关的文件，请手动指定。\n"
                if file_count < 10:
                    augmented_content += self._generate_workspace_file_tree()
            elif len(results) >= file_count * 0.7 and file_count >= 5:
                augmented_content += (
                    f"你的查询涉及整个项目（{len(results)}/{file_count} 个文件匹配），"
                    "附上完整文件树和文件路径清单。\n"
                )
                augmented_content += self._generate_workspace_file_tree()
                all_paths = [rel for rel, _ in iter_workspace_files(self.workspace)]
                augmented_content += f"\n完整文件列表（{len(all_paths)}个），建议一次性全部读取：\n  " + "\n  ".join(sorted(all_paths))
            else:
                augmented_content += f"检索到 {len(results)} 个相关文件：\n"
                for r in results:
                    augmented_content += f"  - {r['file_path']} (综合分: {r['final_score']:.2f})\n"
                if file_count < 10:
                    augmented_content += f"\n附上完整文件树（共{file_count}个文件）：\n" + self._generate_workspace_file_tree()
        except Exception as e:
            augmented_content += f"检索过程出错：{str(e)}\n" + self._generate_workspace_file_tree()
        return original_prompt + augmented_content
