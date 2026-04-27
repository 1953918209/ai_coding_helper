"""
================================================================================
RAG 索引管理器 —— 代码符号索引的生命周期管理
================================================================================

【模块职责】
- 索引的构建/更新/删除（全量重建 + 增量更新）
- 文件哈希变更检测（避免对未修改的文件重复索引）
- 对外提供统一检索接口

【增量索引原理】
每次文件写入后触发 update_file_index()：
  1. 扫描文件 → 解析代码符号
  2. 删除该文件的旧向量 → 写入新向量
  3. 更新文件哈希记录

设置工作区时触发 build_full_index_async()：
  1. 扫描所有文件 → 计算 MD5 哈希
  2. 与历史哈希对比 → 找出新增/修改/删除的文件
  3. 只对有变更的文件重建索引（增量）
  4. 如果哈希文件不存在或向量库为空 → 全量重建
"""
import os, hashlib, json, logging
from typing import List, Callable, Optional, Dict, Any
from core.rag.vector_store import CodeVectorStore
from core.rag.code_parser import scan_workspace_code_to_symbol, should_index_file

logger = logging.getLogger(__name__)

_rag_manager: Optional['RAGManager'] = None  # 全局单例


def init_rag_manager(workspace: str):
    global _rag_manager
    _rag_manager = RAGManager(workspace)
    return _rag_manager

def get_rag_manager() -> Optional['RAGManager']:
    return _rag_manager

def clear_rag_manager():
    global _rag_manager
    if _rag_manager:
        _rag_manager.close()
        _rag_manager = None


# ==============================================================================
# FileHashStore —— 文件哈希存储
# ==============================================================================
# 原理：为每个已索引文件保存 MD5 哈希，下次比对时只处理内容变化的文件。
# 存储位置：workspace/.code_rag_index/file_hashes.json
class FileHashStore:
    def __init__(self, workspace: str):
        self.path = os.path.join(workspace, ".code_rag_index", "file_hashes.json")
        self.hashes: Dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self.hashes = json.load(f)
            except Exception:
                self.hashes = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.hashes, f, indent=2)

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def get(self, rel_path: str) -> Optional[str]:
        return self.hashes.get(rel_path)

    def set(self, rel_path: str, file_hash: str):
        self.hashes[rel_path] = file_hash

    def remove(self, rel_path: str):
        self.hashes.pop(rel_path, None)

    def get_changed(self, current: Dict[str, str]) -> List[str]:
        """返回哈希有变化的文件（新增+修改）"""
        return [p for p, h in current.items() if self.hashes.get(p) != h]

    def get_deleted(self, current_paths: set) -> List[str]:
        """返回已被外部删除的文件"""
        return [p for p in self.hashes if p not in current_paths]


class RAGManager:
    """RAG 管理器核心类：协调向量库、哈希存储、代码解析器"""
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.vector_store = CodeVectorStore(workspace)     # 向量库
        self.hash_store = FileHashStore(workspace)         # 哈希存储
        self.on_index_complete: Optional[Callable[[int], None]] = None   # 索引完成回调
        self.on_index_failed: Optional[Callable[[str], None]] = None     # 索引失败回调
        self._index_ready = False                          # 索引就绪标志

    def close(self):
        if hasattr(self, 'hash_store'): self.hash_store.save()
        if hasattr(self, 'vector_store') and self.vector_store:
            self.vector_store.close()
        logger.info("RAG管理器已关闭")

    def is_index_ready(self) -> bool:
        return self._index_ready

    def set_index_callbacks(self, on_complete=None, on_failed=None):
        self.on_index_complete = on_complete
        self.on_index_failed = on_failed

    def build_full_index_async(self):
        """异步构建索引（独立线程，不阻塞UI）"""
        import threading
        threading.Thread(target=self._build_full_index_sync, daemon=True).start()

    def _scan_workspace_files_with_hashes(self) -> Dict[str, str]:
        """扫描工作区，返回 {文件路径: MD5哈希}"""
        from core.rag.code_parser import EXCLUDED_DIRS
        result = {}
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            for file in files:
                fp = os.path.join(root, file)
                if not should_index_file(fp, self.workspace):
                    continue
                try:
                    with open(fp, 'rb') as f:
                        result[os.path.relpath(fp, self.workspace)] = hashlib.md5(f.read()).hexdigest()
                except Exception:
                    continue
        return result

    def _do_full_rebuild(self, current_hashes: dict) -> int:
        """全量重建索引：扫描→解析→写入向量库→保存哈希"""
        symbols = scan_workspace_code_to_symbol(self.workspace)
        if not symbols:
            if self.on_index_failed: self.on_index_failed("未找到可解析的代码符号")
            return 0
        self.vector_store.clear_index()
        batch_size = 100  # 分批写入，避免内存峰值
        for i in range(0, len(symbols), batch_size):
            self.vector_store.add_symbols(symbols[i:i + batch_size])
        self.hash_store.hashes = current_hashes
        self.hash_store.save()
        return len(symbols)

    def _build_full_index_sync(self):
        """同步构建索引（在异步线程中执行）"""
        total = 0
        try:
            current_hashes = self._scan_workspace_files_with_hashes()
            if not self.hash_store.exists():
                total = self._do_full_rebuild(current_hashes)  # 首次→全量重建
                self._index_ready = total > 0
            else:
                stats = self.vector_store.get_index_stats()
                if stats.get("total_symbols", 0) == 0 and current_hashes:
                    # 哈希文件存在但向量库为空→数据损坏，回退全量重建
                    total = self._do_full_rebuild(current_hashes)
                    self._index_ready = total > 0
                    if total > 0 and self.on_index_complete: self.on_index_complete(total)
                    return
                deleted = self.hash_store.get_deleted(set(current_hashes.keys()))
                changed = self.hash_store.get_changed(current_hashes)
                if not deleted and not changed:
                    total = stats.get("total_symbols", 0)
                    self._index_ready = True  # 无变化→直接就绪
                    if self.on_index_complete: self.on_index_complete(total)
                    return
                # 增量更新：只处理变更的文件
                for rel_path in deleted:
                    self.vector_store.delete_file_symbols(rel_path)
                    self.hash_store.remove(rel_path)
                for rel_path in changed:
                    full_path = os.path.join(self.workspace, rel_path)
                    self.vector_store.delete_file_symbols(rel_path)
                    if os.path.exists(full_path):
                        from core.rag.code_parser import parse_code_file_to_symbol
                        symbols = parse_code_file_to_symbol(full_path, self.workspace)
                        if symbols:
                            self.vector_store.add_symbols(symbols)
                    self.hash_store.set(rel_path, current_hashes[rel_path])
                self.hash_store.save()
                self._index_ready = True
                total = self.vector_store.get_index_stats().get("total_symbols", 0)
            if total > 0 and self.on_index_complete: self.on_index_complete(total)
        except Exception as e:
            logger.exception("构建索引失败")
            if self.on_index_failed: self.on_index_failed(str(e))

    # ---- 单文件索引维护（文件监听 + Agent操作触发）----
    def add_file_to_index(self, rel_path: str):
        full_path = os.path.join(self.workspace, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"文件不存在：{full_path}")
        from core.rag.code_parser import parse_code_file_to_symbol
        symbols = parse_code_file_to_symbol(full_path, self.workspace)
        if symbols: self.vector_store.add_symbols(symbols)
        try:
            with open(full_path, 'rb') as f:
                self.hash_store.set(rel_path, hashlib.md5(f.read()).hexdigest())
            self.hash_store.save()
        except Exception: pass

    def update_file_index(self, rel_path: str):
        """更新单文件索引：先删旧数据→重新解析→写入新数据"""
        self.vector_store.delete_file_symbols(rel_path)
        full_path = os.path.join(self.workspace, rel_path)
        if os.path.exists(full_path):
            from core.rag.code_parser import parse_code_file_to_symbol
            symbols = parse_code_file_to_symbol(full_path, self.workspace)
            if symbols: self.vector_store.add_symbols(symbols)
            try:
                with open(full_path, 'rb') as f:
                    self.hash_store.set(rel_path, hashlib.md5(f.read()).hexdigest())
            except Exception: pass
        else:
            self.hash_store.remove(rel_path)
        self.hash_store.save()

    def remove_file_from_index(self, rel_path: str):
        self.vector_store.delete_file_symbols(rel_path)
        self.hash_store.remove(rel_path)
        self.hash_store.save()

    # ---- 检索接口（代理到 vector_store）----
    def search_related_files(self, query, top_k=5): return self.vector_store.search_related_files(query, top_k)
    def search_related_files_with_scores(self, query, top_k=20): return self.vector_store.search_related_files_with_scores(query, top_k)
    def search_related_files_with_details(self, query, top_k=10): return self.vector_store.search_related_files_with_details(query, top_k)
