"""
================================================================================
代码解析器 —— 将源码文件提取为结构化符号（供 RAG 索引使用）
================================================================================

【双引擎架构】
本模块使用两个互补的解析器处理不同语言：

1. tree-sitter（Python专用）：C语言编写的增量解析器，精确提取AST节点
   - 函数定义、类定义、import语句、装饰器、类型注解、文档字符串
   - 优势：精确到AST级别，能提取文档字符串和类型注解

2. hash_index（通用语言）：基于正则的轻量解析器
   - 支持 Java/C++/C#/JS/TS/Go/Rust/SQL/HTML/CSS/PHP/Swift/Kotlin/QML/CMake 等
   - 优势：零依赖，任何语言都能提取基本结构

【为什么需要代码解析】
RAG 系统如果直接把整个文件当作文本进行向量化：
  - 每个文件只有一个嵌入向量，查询"login函数"可能匹配到包含"login"注释的文件
  - 文件内容太长，嵌入质量下降

解析后：
  - 每个函数/类/方法单独生成嵌入向量
  - 查询"login"能精确定位到 login() 函数，而不是包含"login"字样的整个文件
  - 符号名+代码片段拼接的嵌入文本，比全文更有语义区分度

【重要性评分】
每个符号有一个 importance_score（0.5~2.0），用于向量检索的加权排序：
  - class = 1.5（类比函数更重要）
  - function = 1.3
  - import = 0.8（不太重要）
  - 有文档字符串 ×1.2（说明被认真维护）
  - 私有变量 ×0.8（_开头）或 ×0.9（__开头）
"""
import os, logging, re
from typing import List, Dict, Any, cast, Optional, Tuple
from tree_sitter import Parser, Language, Node   # tree-sitter：C语言增量解析器
import tree_sitter_python as tsp                  # Python 语言包（语法定义）
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from utils.hash_index import generate_hash_index

# hash_index block 类型 → RAG symbol_type 映射表
BLOCK_TYPE_TO_SYMBOL = {
    "import": "import", "function": "function", "class": "class",
    "method": "method", "decorator": "decorator", "global": "variable",
    "macro": "macro", "decl": "declaration", "annotation": "annotation",
}

# ---- tree-sitter 初始化（Python 语言） ----
# tree-sitter 需要编译后的语言 .so/.dll 文件。
# tree-sitter-python 包内置了 Python 语法定义。
_raw_language = tsp.language()
try:
    PY_LANGUAGE: Language = Language(_raw_language)
except TypeError:
    PY_LANGUAGE = cast(Language, _raw_language)
try:
    parser = Parser(language=PY_LANGUAGE)
except TypeError:
    parser = Parser()
    if hasattr(parser, "set_language"):
        parser.set_language(PY_LANGUAGE)
    else:
        parser.language = PY_LANGUAGE

# ---- tree-sitter 节点类型 → 符号类型映射 ----
# tree-sitter 解析后每个节点有一个 type 属性（如 "function_definition"），
# 我们映射为更语义化的符号类型（如 "function"）
SYMBOL_NODE_TYPES = {
    "function_definition": "function",
    "class_definition": "class",
    "assignment": "variable",
    "import_statement": "import",
    "import_from_statement": "import_from",
    "decorated_definition": "decorated",
    "async_function_definition": "async_function",
    "with_statement": "context_manager",
    "try_statement": "try_block",
}

# ---- 过滤规则 ----
EXCLUDED_DIRS = {
    ".workspace_versions", ".git", ".svn", ".idea", ".vscode",
    "__pycache__", "node_modules", ".code_rag_index", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox"
}
EXCLUDED_EXTENSIONS = {".md", ".txt", ".json", ".yml", ".yaml", ".pdf", ".docx", ".log", ".lock", ".toml"}


# ==============================================================================
# should_index_file —— 文件过滤判断
# ==============================================================================
# 四层过滤：
#   1. 路径是否包含禁止目录（如 .git、node_modules）
#   2. 文件名是否在白名单中（CMakeLists.txt、Makefile、Dockerfile）
#   3. 扩展名是否在允许列表中
#   4. 文件是否为二进制（读前1024字节，含 \x00 就是二进制）
def should_index_file(file_path: str, workspace: str) -> bool:
    rel_path = os.path.relpath(file_path, workspace)
    parts = rel_path.split(os.sep)
    for part in parts:
        if part in EXCLUDED_DIRS:
            return False
    from ui.config import ALLOWED_EXT, ALLOWED_FILENAMES
    basename = os.path.basename(file_path)
    if basename in ALLOWED_FILENAMES:
        if not _is_binary(file_path):
            return True
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in EXCLUDED_EXTENSIONS:
        return False
    if ext not in ALLOWED_EXT:
        return False
    if _is_binary(file_path):
        return False
    return True


def _is_binary(file_path: str) -> bool:
    """二进制检测：读取前1024字节，检查是否包含空字节 \x00"""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
        return b"\x00" in chunk
    except Exception:
        return True


# ---- Python AST 符号提取（tree-sitter） ----

def extract_docstring(node: Node, code_bytes: bytes) -> Optional[str]:
    """提取函数/类定义后的文档字符串（三引号字符串）"""
    try:
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        expr = stmt.child(0)
                        if expr and expr.type == "string":
                            docstring = expr.text.decode('utf-8')
                            docstring = re.sub(r'^[\"\']{3}|[\"\']{3}$', '', docstring).strip()
                            if len(docstring) > 0:
                                return docstring[:500]
    except Exception:
        pass
    return None


def extract_type_annotations(node: Node, code_bytes: bytes) -> Dict[str, str]:
    """提取函数参数的类型注解（如 def foo(x: int) → {"param:x": "int"}）"""
    type_info = {}
    try:
        if node.type in ["function_definition", "async_function_definition"]:
            parameters_node = None
            for child in node.children:
                if child.type == "parameters":
                    parameters_node = child
                    break
            if parameters_node:
                for param in parameters_node.children:
                    if param.type == "identifier":
                        param_name = param.text.decode('utf-8')
                        for sibling in param.parent.children:
                            if sibling.type == "type":
                                type_info[f"param:{param_name}"] = sibling.text.decode('utf-8')
                                break
    except Exception:
        pass
    return type_info


def extract_decorators(node: Node, code_bytes: bytes) -> List[str]:
    """提取装饰器列表（如 @staticmethod, @property）"""
    decorators = []
    try:
        for child in node.children:
            if child.type == "decorator":
                decorators.append(child.text.decode('utf-8').strip())
    except Exception:
        pass
    return decorators


# ==============================================================================
# parse_code_file —— 解析单个文件（Python → tree-sitter / 其他 → hash_index）
# ==============================================================================
def parse_code_file(file_path: str, workspace: str) -> List[Dict[str, Any]]:
    """解析单个文件，返回符号列表"""
    if not should_index_file(file_path, workspace):
        return []
    rel_path = os.path.relpath(file_path, workspace)
    if not os.path.exists(file_path):
        return []

    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in ('.py',):
        return _parse_non_python_file(file_path, workspace)  # 非Python → hash_index

    # ---- Python 文件：tree-sitter 解析 ----
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        if not code.strip():
            return [{"symbol_type": "file", "symbol_name": os.path.basename(file_path),
                     "file_path": rel_path, "line_start": 1, "line_end": 1,
                     "code_snippet": f"[空Python文件] {rel_path}",
                     "docstring": "", "type_annotations": {}, "decorators": [], "importance_score": 1.0}]

        code_bytes = code.encode('utf-8')
        tree = parser.parse(code_bytes)          # tree-sitter 生成 CST（具体语法树）
        symbols = _extract_enhanced_symbols(tree.root_node, rel_path, code_bytes)

        # 每个文件追加一个 file_meta 占位符号，确保全项目查询时所有文件都有底分
        symbols.append({
            "symbol_type": "file_meta", "symbol_name": f"file:{os.path.basename(file_path)}",
            "file_path": rel_path, "line_start": 1, "line_end": 1,
            "code_snippet": f"文件: {rel_path} | 项目中的位置",
            "docstring": "", "type_annotations": {}, "decorators": [], "importance_score": 0.8
        })
        return symbols
    except Exception as e:
        return [{"symbol_type": "file", "symbol_name": os.path.basename(file_path),
                 "file_path": rel_path, "line_start": 1, "line_end": 1,
                 "code_snippet": f"[解析失败] {rel_path}",
                 "docstring": "", "type_annotations": {}, "decorators": [], "importance_score": 1.0}]


def scan_workspace_code(workspace_path: str) -> List[Dict[str, Any]]:
    """扫描整个工作区，递归解析所有允许的文件"""
    code_symbols = []
    try:
        for root, dirs, files in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]  # 原地过滤，跳过禁止目录
            for file in files:
                symbols = parse_code_file(os.path.join(root, file), workspace_path)
                code_symbols.extend(symbols)
    except Exception as e:
        logger.error(f"扫描工作区失败: {str(e)}")
    return code_symbols


# ==============================================================================
# _extract_enhanced_symbols —— tree-sitter AST 递归提取
# ==============================================================================
# 递归遍历 tree-sitter 的语法树，对每个符合条件的节点（函数/类/赋值/import等）
# 提取：符号名、代码片段、文档字符串、类型注解、装饰器、重要性分数
def _extract_enhanced_symbols(node: Node, file_path: str, code_bytes: bytes, symbols=None) -> List[Dict[str, Any]]:
    if symbols is None:
        symbols = []
    if node.type in SYMBOL_NODE_TYPES:
        try:
            code_snippet = node.text.decode('utf-8') if node.text else ""
        except Exception:
            code_snippet = ""
        code_snippet = code_snippet[:2000]

        name_node = node.child_by_field_name("name")
        name = name_node.text.decode('utf-8') if name_node else ""
        docstring = extract_docstring(node, code_bytes)
        type_annotations = extract_type_annotations(node, code_bytes)
        decorators = extract_decorators(node, code_bytes)

        importance_score = calculate_importance_score(
            node_type=node.type, name=name,
            has_docstring=bool(docstring), has_type_annotations=bool(type_annotations),
            has_decorators=bool(decorators)
        )
        symbols.append({
            "symbol_type": SYMBOL_NODE_TYPES[node.type], "symbol_name": name,
            "file_path": file_path, "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1, "code_snippet": code_snippet,
            "docstring": docstring or "", "type_annotations": type_annotations,
            "decorators": decorators, "importance_score": importance_score
        })
    for child in node.children:
        _extract_enhanced_symbols(child, file_path, code_bytes, symbols)
    return symbols


# ==============================================================================
# calculate_importance_score —— 符号重要性打分
# ==============================================================================
# 公式：base_weight × doc_bonus × type_annotation_bonus × decorator_bonus × name_penalty
# 最终范围限制在 0.5 ~ 2.0
def calculate_importance_score(node_type, name, has_docstring, has_type_annotations, has_decorators) -> float:
    score = 1.0
    type_weights = {
        "class": 1.5, "function": 1.3, "async_function": 1.3, "decorated": 1.4,
        "variable": 1.0, "import": 0.8, "import_from": 0.8,
        "context_manager": 1.1, "try_block": 1.1
    }
    if node_type in type_weights:
        score *= type_weights[node_type]
    if has_docstring:        score *= 1.2
    if has_type_annotations: score *= 1.15
    if has_decorators:       score *= 1.1
    if name and name.isupper():                        score *= 0.9      # 常量
    elif name and name.startswith("_"):
        score *= 0.8 if name.startswith("__") else 0.9  # 私有变量
    return min(max(score, 0.5), 2.0)


# ==============================================================================
# CodeSymbol 数据类 —— 统一的代码符号表示
# ==============================================================================
@dataclass
class CodeSymbol:
    symbol_type: str       # function / class / method / import / variable ...
    symbol_name: str       # 符号名（如 "read_file"）
    file_path: str         # 相对工作区的文件路径
    line_start: int        # 起始行号
    line_end: int          # 结束行号
    code_snippet: str = ""      # 代码片段（用于向量检索的上下文）
    docstring: str = ""         # 文档字符串
    type_annotations: Dict[str, str] = None    # 类型注解
    decorators: List[str] = None              # 装饰器列表
    importance_score: float = 1.0             # 重要性分数

    def __post_init__(self):
        if self.type_annotations is None: self.type_annotations = {}
        if self.decorators is None: self.decorators = []

    def to_embedding_text(self) -> str:
        """
        生成用于向量嵌入的拼接文本。
        这是决定检索质量的关键——文本内容直接影响向量相似度。
        格式：文件:xxx | 类型:function | 名称:login | 行号:10-25 | 代码:...| 文档:...
        """
        base = f"文件:{self.file_path} | 类型:{self.symbol_type} | 名称:{self.symbol_name} | 行号:{self.line_start}-{self.line_end}"
        if self.code_snippet:   base += f" | 代码:{self.code_snippet}"
        if self.docstring:      base += f" | 文档:{self.docstring[:200]}"
        if self.type_annotations:
            type_str = " ".join([f"{k}:{v}" for k, v in self.type_annotations.items()])
            if type_str: base += f" | 类型注解:{type_str}"
        if self.decorators:     base += f" | 装饰器:{' '.join(self.decorators)}"
        return base


def parse_code_file_to_symbol(file_path, workspace):
    return [CodeSymbol(**sym) for sym in parse_code_file(file_path, workspace)]

def scan_workspace_code_to_symbol(workspace_path):
    return [CodeSymbol(**sym) for sym in scan_workspace_code(workspace_path)]


# ==============================================================================
# hash_index 多语言解析桥接 —— 非 Python 文件的解析方法
# ==============================================================================
def _parse_non_python_file(file_path: str, workspace: str) -> List[Dict[str, Any]]:
    """用 hash_index 解析 C++/Java/JS/TS/Go 等非 Python 文件"""
    rel_path = os.path.relpath(file_path, workspace)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [{"symbol_type": "file", "symbol_name": os.path.basename(file_path),
                 "file_path": rel_path, "line_start": 1, "line_end": 1,
                 "code_snippet": f"[读取失败] {rel_path}",
                 "docstring": "", "type_annotations": {}, "decorators": [], "importance_score": 1.0}]
    if not content.strip():
        return [{"symbol_type": "file", "symbol_name": os.path.basename(file_path),
                 "file_path": rel_path, "code_snippet": f"[空文件] {rel_path}",
                 "line_start": 1, "line_end": 1, "docstring": "",
                 "type_annotations": {}, "decorators": [], "importance_score": 1.0}]

    hash_index = generate_hash_index(content, file_path)
    symbols = []
    for block in hash_index.get("blocks", []):
        _block_to_symbols(block, rel_path, symbols)
    if not symbols:
        symbols.append({"symbol_type": "file", "symbol_name": os.path.basename(file_path),
                        "file_path": rel_path, "line_start": 1, "line_end": 1,
                        "code_snippet": f"[未识别结构] {rel_path}",
                        "docstring": "", "type_annotations": {}, "decorators": [], "importance_score": 0.8})
    symbols.append({"symbol_type": "file_meta", "symbol_name": f"file:{os.path.basename(file_path)}",
                    "file_path": rel_path, "line_start": 1, "line_end": 1,
                    "code_snippet": f"文件: {rel_path} | 项目中的位置",
                    "docstring": "", "type_annotations": {}, "decorators": [], "importance_score": 0.8})
    return symbols


def _block_to_symbols(block: dict, file_path: str, symbols: List[Dict[str, Any]]):
    """递归转换 hash_index block → RAG 符号"""
    symbol_type = BLOCK_TYPE_TO_SYMBOL.get(block.get("type", ""), "global")
    symbols.append({
        "symbol_type": symbol_type, "symbol_name": block.get("name", ""),
        "file_path": file_path, "line_start": block.get("start", 1), "line_end": block.get("end", 1),
        "code_snippet": block.get("content", "")[:2000],
        "docstring": "", "type_annotations": {}, "decorators": [],
        "importance_score": _calc_block_importance(block)
    })
    for sub in block.get("sub_blocks", []):
        _block_to_symbols(sub, file_path, symbols)


def _calc_block_importance(block: dict) -> float:
    type_weights = {"class": 1.5, "function": 1.3, "method": 1.2, "import": 0.8,
                    "macro": 0.8, "decl": 1.0, "global": 1.0, "decorator": 0.9, "annotation": 0.7}
    return type_weights.get(block.get("type", ""), 1.0)
