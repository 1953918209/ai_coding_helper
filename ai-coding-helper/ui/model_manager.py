"""
================================================================================
模型配置管理器 —— 持久化 + 管理界面
================================================================================

【模块职责】
1. ModelConfigManager: 模型配置的增删改查 + JSON 持久化
2. ModelEditDialog:    添加/编辑单个模型的对话框
3. ModelManageDialog:  模型列表管理主界面（含导入导出）

【持久化机制】
配置存储在项目根目录的 model_configs.json：
  {
    "models": {
      "deepseek": {"name":"DeepSeek","api_key":"","base_url":"...","model_name":"...","enabled":true,"builtin":true},
      "qwen":    {...},
      "my_model": {...}   ← 用户自定义
    },
    "default_model": "deepseek"
  }

【API Key 优先级（内置模型）】
  用户手动填写的值（JSON中非空） > 环境变量 > 空字符串
  自定义模型必须填写 API Key（不支持环境变量回退）

【环境变量映射】
  deepseek → DEEPSEEK_API_KEY
  qwen     → DASHSCOPE_API_KEY
"""
import json, os, copy, logging, tkinter as tk
from tkinter import ttk, messagebox, filedialog

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_configs.json")

# 内置模型的默认配置（api_key 为空 → 启动时从环境变量回填）
BUILTIN_DEFAULTS = {
    "deepseek": {"name": "DeepSeek", "api_key": "", "base_url": "https://api.deepseek.com",
                 "model_name": "deepseek-reasoner", "enabled": True, "builtin": True},
    "qwen": {"name": "阿里云通义千问", "api_key": "",
             "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "model_name": "qwen3.6-plus", "enabled": True, "builtin": True},
}

DEFAULT_MODEL_KEY = "deepseek"

_BUILTIN_ENV_MAP = {"deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY"}


class ModelConfigManager:
    """模型配置管理器：JSON 持久化 + CRUD"""
    def __init__(self):
        self.models: dict = {}
        self.default_key: str = DEFAULT_MODEL_KEY
        self._load()

    def _resolve_api_key(self, model_key: str, stored_api_key: str) -> str:
        """
        API Key 回退解析。
        优先级：stored_api_key（用户填写）> 环境变量 > 空字符串
        """
        if stored_api_key.strip():
            return stored_api_key
        env_var = _BUILTIN_ENV_MAP.get(model_key, "")
        if env_var:
            env_val = os.getenv(env_var, "")
            if env_val:
                return env_val
        return stored_api_key

    def _load(self):
        """从 model_configs.json 加载配置，失败则用内置默认"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.models = data.get("models", {})
                self.default_key = data.get("default_model", DEFAULT_MODEL_KEY)
                self._ensure_builtins_exist()  # 确保内置模型不丢失
                logger.info(f"已加载模型配置，共 {len(self.models)} 个模型")
                return
            except Exception as e:
                logger.warning(f"加载模型配置失败：{e}")
        self.models = copy.deepcopy(BUILTIN_DEFAULTS)
        self.default_key = DEFAULT_MODEL_KEY
        self._save()

    def _ensure_builtins_exist(self):
        """如果 JSON 中缺少内置模型，自动补回"""
        for key, cfg in BUILTIN_DEFAULTS.items():
            if key not in self.models:
                self.models[key] = copy.deepcopy(cfg)

    def _save(self):
        try:
            data = {"models": self.models, "default_model": self.default_key}
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存模型配置失败：{e}")

    def get_models(self) -> dict:                    return self.models
    def get_enabled_models(self) -> dict:            return {k: v for k, v in self.models.items() if v.get("enabled", True)}
    def get_model(self, key: str) -> dict:           return self.models.get(key, {})
    def get_default_key(self) -> str:
        if self.default_key not in self.models:
            self.default_key = next(iter(self.models.keys()), DEFAULT_MODEL_KEY)
            self._save()
        return self.default_key

    def set_default(self, key: str):
        if key not in self.models: raise ValueError(f"模型 {key} 不存在")
        self.default_key = key; self._save()

    def add_model(self, key: str, config: dict):
        if key in self.models: raise ValueError(f"模型标识 {key} 已存在")
        if not self._validate_config(config, key): raise ValueError("模型配置不完整")
        config["builtin"] = False; config["enabled"] = True
        self.models[key] = config; self._save()

    def update_model(self, key: str, config: dict):
        if key not in self.models: raise ValueError(f"模型 {key} 不存在")
        if not self._validate_config(config, key): raise ValueError("模型配置不完整")
        builtin = self.models[key].get("builtin", False)
        config["builtin"] = builtin
        self.models[key] = config; self._save()

    def delete_model(self, key: str):
        if key not in self.models: raise ValueError(f"模型 {key} 不存在")
        if self.models[key].get("builtin"): raise ValueError("内置模型不可删除")
        del self.models[key]
        if self.default_key == key:
            self.default_key = next(iter(self.models.keys()), DEFAULT_MODEL_KEY)
        self._save()

    def toggle_model(self, key: str):
        if key not in self.models: raise ValueError(f"模型 {key} 不存在")
        self.models[key]["enabled"] = not self.models[key].get("enabled", True)
        self._save()

    def reset_builtin(self, key: str):
        if key not in self.models or not self.models[key].get("builtin"):
            raise ValueError("只能重置内置模型")
        self.models[key] = copy.deepcopy(BUILTIN_DEFAULTS[key]); self._save()

    def _validate_config(self, config: dict, model_key: str = "") -> bool:
        """验证模型配置完整性，内置模型允许空 api_key（回退环境变量）"""
        for k in ["name", "base_url", "model_name"]:
            if k not in config or not config[k]: return False
        stored_api_key = config.get("api_key", "")
        if stored_api_key.strip(): return True
        env_var = _BUILTIN_ENV_MAP.get(model_key, "")
        if env_var and os.getenv(env_var, ""): return True
        return bool(stored_api_key.strip())

    def export_to_file(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"models": self.models, "default_model": self.default_key}, f, ensure_ascii=False, indent=2)
        logger.info(f"模型配置已导出到 {filepath}")

    def import_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        imported = data.get("models", {})
        for key, cfg in imported.items():
            if cfg.get("builtin"): cfg["builtin"] = False  # 导入的内置模型转为自定义
            if key in self.models:
                key = self._make_unique_key(key)
            self.models[key] = cfg
        if "default_model" in data:
            self.default_key = data["default_model"]
            if self.default_key not in self.models:
                self.default_key = next(iter(self.models.keys()), DEFAULT_MODEL_KEY)
        self._save()

    def _make_unique_key(self, base_key: str) -> str:
        if base_key not in self.models: return base_key
        i = 1
        while f"{base_key}_{i}" in self.models: i += 1
        return f"{base_key}_{i}"

    def sync_to_config(self):
        """同步到 ui.config 的全局变量（让其他模块感知变更）"""
        import ui.config as cfg
        cfg.SUPPORTED_MODELS.clear()
        for key, m in self.get_enabled_models().items():
            cfg.SUPPORTED_MODELS[key] = {
                "name": m["name"], "api_key": self._resolve_api_key(key, m.get("api_key", "")),
                "base_url": m["base_url"], "model_name": m["model_name"],
            }
        cfg.DEFAULT_MODEL = self.get_default_key()
        if cfg.DEFAULT_MODEL not in cfg.SUPPORTED_MODELS:
            cfg.DEFAULT_MODEL = next(iter(cfg.SUPPORTED_MODELS.keys()), DEFAULT_MODEL_KEY)
            self.default_key = cfg.DEFAULT_MODEL; self._save()


# ==============================================================================
# ModelEditDialog —— 模型编辑对话框
# ==============================================================================
class ModelEditDialog(tk.Toplevel):
    """添加/编辑单个模型配置的弹窗（模态对话框）"""
    def __init__(self, parent, title, model_key="", model_config=None, is_builtin=False):
        super().__init__(parent)
        self.title(title)
        self.model_key = model_key
        self.is_builtin = is_builtin
        self.result = None
        self._build_ui(model_config)
        self.transient(parent); self.grab_set()
        self.resizable(False, False)
        self._center(parent)

    def _center(self, parent):
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h = 440, 320
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

    def _build_ui(self, config):
        frame = tk.Frame(self, bg="white", padx=16, pady=12)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="模型标识 (唯一Key):", font=("Microsoft YaHei", 10), bg="white", anchor="w").pack(fill="x")
        self.key_entry = tk.Entry(frame, font=("Consolas", 10))
        self.key_entry.pack(fill="x", pady=(0, 8))
        if self.model_key:
            self.key_entry.insert(0, self.model_key)
            if self.is_builtin: self.key_entry.config(state="readonly")  # 内置模型不可改名

        tk.Label(frame, text="显示名称:", font=("Microsoft YaHei", 10), bg="white", anchor="w").pack(fill="x")
        self.name_entry = tk.Entry(frame, font=("Consolas", 10))
        self.name_entry.pack(fill="x", pady=(0, 8))

        tk.Label(frame, text="API Key:", font=("Microsoft YaHei", 10), bg="white", anchor="w").pack(fill="x")
        api_frame = tk.Frame(frame, bg="white"); api_frame.pack(fill="x", pady=(0, 2))
        self.api_entry = tk.Entry(api_frame, font=("Consolas", 10), show="*")
        self.api_entry.pack(side="left", fill="x", expand=True)
        self.show_btn = tk.Button(api_frame, text="👁", relief="flat", bg="#F5F5F5", command=self._toggle_api)
        self.show_btn.pack(side="left", padx=(2, 0))

        # 提示标签（显示环境变量状态）
        self.api_hint_label = tk.Label(frame, text="", font=("Microsoft YaHei", 8), bg="white", fg="#999999", anchor="w")
        self.api_hint_label.pack(fill="x", pady=(0, 8))

        tk.Label(frame, text="Base URL:", font=("Microsoft YaHei", 10), bg="white", anchor="w").pack(fill="x")
        self.url_entry = tk.Entry(frame, font=("Consolas", 10)); self.url_entry.pack(fill="x", pady=(0, 8))

        tk.Label(frame, text="模型名称 (model):", font=("Microsoft YaHei", 10), bg="white", anchor="w").pack(fill="x")
        self.model_entry = tk.Entry(frame, font=("Consolas", 10)); self.model_entry.pack(fill="x", pady=(0, 12))

        if config:
            self.name_entry.insert(0, config.get("name", ""))
            stored_api_key = config.get("api_key", "")
            self.api_entry.insert(0, stored_api_key)
            self.url_entry.insert(0, config.get("base_url", ""))
            self.model_entry.insert(0, config.get("model_name", ""))
            if self.is_builtin and not stored_api_key.strip():
                env_var = _BUILTIN_ENV_MAP.get(self.model_key, "")
                if env_var and os.getenv(env_var):
                    self.api_hint_label.config(text=f"⚡ 当前使用环境变量 {env_var}（留空则自动读取）")
                else:
                    self.api_hint_label.config(text="⚠ 环境变量未设置，请输入 API Key")
            elif self.is_builtin and stored_api_key.strip():
                self.api_hint_label.config(text="✏️ 已手动设置（清空则恢复使用环境变量）")

        btn_frame = tk.Frame(frame, bg="white"); btn_frame.pack(fill="x", pady=(4, 0))
        tk.Button(btn_frame, text="取消", font=("Microsoft YaHei", 10), bg="#F5F5F5", relief="flat", width=10,
                 cursor="hand2", command=self.destroy).pack(side="right", padx=4)
        tk.Button(btn_frame, text="保存", font=("Microsoft YaHei", 10, "bold"), bg="#1976D2", fg="white",
                 relief="flat", width=10, cursor="hand2", command=self._on_save).pack(side="right", padx=4)

    def _toggle_api(self):
        if self.api_entry.cget("show") == "*": self.api_entry.config(show=""); self.show_btn.config(text="🙈")
        else: self.api_entry.config(show="*"); self.show_btn.config(text="👁")

    def _on_save(self):
        key = self.key_entry.get().strip(); name = self.name_entry.get().strip()
        api_key = self.api_entry.get().strip(); base_url = self.url_entry.get().strip()
        model_name = self.model_entry.get().strip()
        if not key or not name or not base_url or not model_name:
            messagebox.showwarning("验证失败", "除 API Key 外的所有字段均为必填", parent=self); return
        if not base_url.startswith("http"):
            messagebox.showwarning("验证失败", "Base URL 必须以 http 开头", parent=self); return
        if self.is_builtin and not api_key:
            env_var = _BUILTIN_ENV_MAP.get(self.model_key, "")
            if not env_var or not os.getenv(env_var):
                messagebox.showwarning("验证失败", f"环境变量 {env_var} 未设置，请输入 API Key", parent=self); return
        if not self.is_builtin and not api_key:
            messagebox.showwarning("验证失败", "自定义模型必须填写 API Key", parent=self); return
        self.result = {"key": key, "name": name, "api_key": api_key, "base_url": base_url.rstrip("/"), "model_name": model_name}
        self.destroy()


# ==============================================================================
# ModelManageDialog —— 模型管理主界面
# ==============================================================================
class ModelManageDialog(tk.Toplevel):
    """
    模型管理对话框：Treeview 列表 + 操作按钮 + 导入导出。

    布局：
    ┌──────────────────────────────────────┐
    │  📋 模型列表          默认：deepseek  │
    │  ┌──────────────────────────────────┐│
    │  │ 状态 │ 标识 │ 名称 │ 模型名 │类型││
    │  │ ✅  │ deep.│ DeepS│ deeps..│内置 ││
    │  │ ✅  │ qwen │ 通义.│ qwen3..│内置 ││
    │  └──────────────────────────────────┘│
    │  [添加] [编辑] [删除] [禁用] [默认]  │
    │                         [导入] [导出]│
    └──────────────────────────────────────┘
    """
    def __init__(self, parent, config_manager: ModelConfigManager, on_changed=None):
        super().__init__(parent)
        self.title("模型管理")
        self.config_manager = config_manager
        self.on_changed = on_changed
        self._build_ui(); self._refresh_list()
        self.transient(parent)
        self.resizable(True, True); self.minsize(650, 460)
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        self.geometry(f"700x500+{px + (pw - 700) // 2}+{py + (ph - 500) // 2}")
        self.grab_set()

    def _build_ui(self):
        main = tk.Frame(self, bg="white", padx=12, pady=10); main.pack(fill="both", expand=True)
        header = tk.Frame(main, bg="white"); header.pack(fill="x", pady=(0, 8))
        tk.Label(header, text="📋 模型列表", font=("Microsoft YaHei", 12, "bold"), bg="white").pack(side="left")
        default_key = self.config_manager.get_default_key()
        tk.Label(header, text=f"   默认：{default_key}", font=("Microsoft YaHei", 9), bg="white", fg="#666666").pack(side="left")

        list_frame = tk.Frame(main, bg="white", relief="solid", bd=1); list_frame.pack(fill="both", expand=True, pady=(0, 8))
        columns = ("enabled", "key", "name", "model", "type")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10, selectmode="browse")
        for col, txt, w, anchor in [("enabled","状态",50,"center"),("key","标识",100,""),("name","显示名称",120,""),("model","模型名",140,""),("type","类型",60,"center")]:
            self.tree.heading(col, text=txt); self.tree.column(col, width=w, anchor=anchor)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", lambda e: self._edit_model())

        btn_frame = tk.Frame(main, bg="white"); btn_frame.pack(fill="x")
        self._btn_row1 = tk.Frame(btn_frame, bg="white"); self._btn_row1.pack(fill="x", pady=(0, 4))
        self._btn_row2 = tk.Frame(btn_frame, bg="white"); self._btn_row2.pack(fill="x", pady=(0, 4))
        self._build_button_row(self._btn_row1, [("➕ 添加模型","#4CAF50","white",self._add_model),("✏️ 编辑","#1976D2","white",self._edit_model),("🗑 删除","#EF5350","white",self._delete_model)])
        self._build_button_row(self._btn_row2, [("🔄 启用/禁用","#FF9800","white",self._toggle_model),("⭐ 设默认","#9C27B0","white",self._set_default),("📥 导入","#F5F5F5","black",self._import_config),("📤 导出","#F5F5F5","black",self._export_config)])
        self.bind("<Configure>", self._on_dialog_resize, add="+")

    def _build_button_row(self, parent, buttons):
        for text, bg, fg, cmd in buttons:
            btn = tk.Button(parent, text=text, font=("Microsoft YaHei", 10), bg=bg, fg=fg, relief="flat", cursor="hand2", command=cmd)
            btn.pack(side="left", fill="x", expand=True, padx=1, pady=2)

    def _on_dialog_resize(self, event):
        pass

    def _refresh_list(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for key, m in self.config_manager.get_models().items():
            self.tree.insert("", "end", iid=key, values=("✅" if m.get("enabled", True) else "❌", key, m["name"], m["model_name"], "内置" if m.get("builtin") else "自定义"))

    def _get_selected(self):
        sel = self.tree.selection(); return sel[0] if sel else None

    def _add_model(self):
        dlg = ModelEditDialog(self, "添加新模型"); self.wait_window(dlg)
        if dlg.result:
            try: self.config_manager.add_model(dlg.result["key"], {"name":dlg.result["name"],"api_key":dlg.result["api_key"],"base_url":dlg.result["base_url"],"model_name":dlg.result["model_name"]}); self._refresh_list(); self._notify_change()
            except ValueError as e: messagebox.showerror("错误", str(e), parent=self)

    def _edit_model(self):
        key = self._get_selected()
        if not key: return
        model = self.config_manager.get_model(key)
        dlg = ModelEditDialog(self, f"编辑模型 - {key}", key, model, model.get("builtin", False)); self.wait_window(dlg)
        if dlg.result:
            try: self.config_manager.update_model(key, {"name":dlg.result["name"],"api_key":dlg.result["api_key"],"base_url":dlg.result["base_url"],"model_name":dlg.result["model_name"]}); self._refresh_list(); self._notify_change()
            except ValueError as e: messagebox.showerror("错误", str(e), parent=self)

    def _delete_model(self):
        """删除选中的自定义模型（内置模型不可删除，会弹确认框）"""
        key = self._get_selected()
        if not key: return
        model = self.config_manager.get_model(key)
        if model.get("builtin"): messagebox.showwarning("无法删除", "内置模型不可删除", parent=self); return
        if not messagebox.askyesno("确认删除", f"确定要删除模型 [{model['name']}] 吗？", parent=self): return
        try: self.config_manager.delete_model(key); self._refresh_list(); self._notify_change()
        except ValueError as e: messagebox.showerror("错误", str(e), parent=self)

    def _toggle_model(self):
        """启用/禁用选中的模型（禁用后不会出现在下拉列表中）"""
        key = self._get_selected()
        if not key: return
        self.config_manager.toggle_model(key); self._refresh_list(); self._notify_change()

    def _set_default(self):
        """将选中的模型设为默认（禁用的模型不能设为默认）"""
        key = self._get_selected()
        if not key: return
        model = self.config_manager.get_model(key)
        if not model.get("enabled"): messagebox.showwarning("提示", "禁用的模型不能设为默认", parent=self); return
        self.config_manager.set_default(key); self._refresh_list(); self._notify_change()

    def _export_config(self):
        """导出当前模型配置到 JSON 文件（另存为对话框）"""
        path = filedialog.asksaveasfilename(title="导出模型配置", defaultextension=".json", filetypes=[("JSON 文件","*.json")], initialfile="model_configs_export.json", parent=self)
        if path:
            try: self.config_manager.export_to_file(path); messagebox.showinfo("导出成功", f"配置已导出到：\n{path}", parent=self)
            except Exception as e: messagebox.showerror("导出失败", str(e), parent=self)

    def _import_config(self):
        """从 JSON 文件导入模型配置（合并而非覆盖已有配置）"""
        path = filedialog.askopenfilename(title="导入模型配置", filetypes=[("JSON 文件","*.json")], parent=self)
        if not path: return
        try: self.config_manager.import_from_file(path); self._refresh_list(); self._notify_change(); messagebox.showinfo("导入成功", "模型配置已导入", parent=self)
        except Exception as e: messagebox.showerror("导入失败", f"文件格式错误：{e}", parent=self)

    def _notify_change(self):
        """通知外部（AppHandlers）模型配置已变更，同步到 ui.config 全局变量"""
        self.config_manager.sync_to_config()
        if self.on_changed: self.on_changed()
