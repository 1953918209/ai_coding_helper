"""
================================================================================
哈希索引系统 —— 多语言代码块精确定位与批量修改的基础设施
================================================================================

【核心功能】
1. 将代码文件解析为带哈希指纹的块（用于精确修改定位）
2. 支持 edit_file_batch 的 hash-based 定位修改
3. 多语言支持：Python(tree-sitter AST) + C家族/Web/SQL(正则解析器)

【为什么用 Hash 而不是行号】
行号在修改后会变化：插入2行代码 → 后面所有行号+2 → 后续修改定位错乱
Hash 基于内容生成：只要代码块不变，hash 就不变 → 不受行号偏移影响

【HASH INDEX 结构】
generate_hash_index("def login():\\n    pass\\n") 返回:
  {
    "lines": {
      "a3f2c1d8": {"line_num": 1, "content": "def login():", "offset": 0, "length": 13},
      "b4e5f6a7": {"line_num": 2, "content": "    pass",    "offset": 14, "length": 10}
    },
    "blocks": [
      {"type": "function", "name": "login", "hash": "c1d2e3f4a5b6",
       "offset": 0, "length": 24, "start": 1, "end": 2, "content": "def login():\\n    pass\\n"}
    ]
  }

edit_file_batch 通过 block["hash"] 定位到 offset/length 来精确替换。

【多语言支持】
  _parse_python():     Python AST 解析器 (tree-sitter 替代，纯标准库)
  _parse_brace_lang(): C家族语言解析器 (C/C++/Java/C#/JS/TS/Go/Rust 等)
  _parse_sql():        SQL DDL 解析器 (CREATE TABLE/VIEW/INDEX/TRIGGER 等)
  _parse_html/css():    HTML/CSS 简单解析

未知语言 → _detect_language 返回 None → 仅生成行级 hash，无块级结构
（edit_file_batch 仍可定位到具体行，但无法定位到代码块）

【面试高频题 —— Hash Index 设计】
Q21: 为什么用 Hash 而不是行号来定位代码块？
A:  行号在修改后会变化。插入2行 → 后面所有行号+2 → 后续编辑定位错乱。
    Hash 基于内容生成：只要代码块内容不变，hash 就不变，不受行号偏移影响。

Q22: edit_file_batch 为什么"从下往上"应用修改？
A:  如果从上往下应用，前面的修改会改变后续代码的字节偏移量(offset)，
    导致后续 hash 定位错误。从下往上(offset 从大到小)则前面的 offset 不变。

Q23: Python 解析用了 ast 模块，为什么其他语言用正则？
A:  Python 用 ast（标准库，零依赖 AST 解析）是 tree-sitter 的纯标准库替代。
    其他语言用正则：虽然不完美但覆盖大多数 C 家族语言结构，零依赖。
    tree-sitter 在 code_parser.py 中用于 Python 文件的更精确解析。
"""
import ast, hashlib, logging, os, re

logger = logging.getLogger(__name__)

LINE_HASH_LEN = 8     # 行级 hash 长度（MD5 前8位）
BLOCK_HASH_LEN = 12   # 块级 hash 长度（SHA256 前12位）


def _line_hash(line_num: int, line_content: str) -> str:
    """行级 Hash：用 MD5 生成 8 位指纹。包含行号确保不同位置的相同行不同 hash。"""
    raw = f"{line_num}:{line_content}"
    return hashlib.md5(raw.encode()).hexdigest()[:LINE_HASH_LEN]


def _block_hash(block_content: str) -> str:
    """块级 Hash：用 SHA256 生成 12 位指纹。比 MD5 更抗碰撞。"""
    return hashlib.sha256(block_content.encode()).hexdigest()[:BLOCK_HASH_LEN]


def _calc_offset(lines: list, start_line: int, end_line: int) -> int:
    """计算从文件开头到第 start_line 行的字节偏移量。
    offset 用于编辑时精确定位写入位置。"""
    offset = 0
    for i in range(start_line - 1):
        offset += len(lines[i]) + 1   # +1 是换行符
    return offset


# ==============================================================================
# 多语言支持基础设施 —— 语言检测 + 解析器注册
# ==============================================================================
# depth 跟踪括号嵌套层数。遇到 { 加1，遇到 } 减1，回到 0 表示块结束
LANGUAGE_EXT_MAP = {
    '.py': 'python',
    '.java': 'java',
    '.cpp': 'cpp', '.cxx': 'cpp', '.cc': 'cpp', '.c': 'c',
    '.h': 'cpp', '.hpp': 'cpp', '.hh': 'cpp', '.hxx': 'cpp',
    '.cs': 'csharp',
    '.js': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.sql': 'sql',
    '.html': 'html', '.htm': 'html',
    '.css': 'css',
    '.go': 'go',
    '.rs': 'rust',
    '.kt': 'kotlin', '.kts': 'kotlin',
    '.swift': 'swift',
    '.rb': 'ruby',
    '.php': 'php',
    '.pro': 'qmake', '.pri': 'qmake',
    '.qml': 'qml',
    '.ui': 'xml', '.qrc': 'xml',
    '.cmake': 'cmake',
}

_PARSERS = {}

def register_parser(language: str, func):
    _PARSERS[language] = func


def _find_brace_end(lines: list, start_line: int) -> int:
    """查找花括号块结束行号。用 depth 计数器跟踪 { } 嵌套。"""
    depth = 0
    started = False
    for i in range(start_line - 1, len(lines)):
        s = lines[i]
        stripped = s.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            if not started:
                continue
        depth += s.count('{') - s.count('}')
        if not started and '{' in s:
            started = True
        if started and depth == 0:
            return i + 1
    return len(lines)


# 函数匹配时要跳过的控制流关键字（if/for/while 不是函数定义）
_SKIP_KEYWORDS = {'if', 'for', 'while', 'switch', 'catch', 'else', 'do', 'return', 'try', 'case'}


def _detect_language(file_path: str):
    """检测文件语言，未知扩展名返回None（跳过解析器，仅保留行级hash）"""
    if not file_path:
        return 'python'
    ext = os.path.splitext(file_path)[1].lower()
    return LANGUAGE_EXT_MAP.get(ext)


def _make_lines_result(content: str) -> tuple:
    lines = content.split('\n')
    result = {"lines": {}, "blocks": []}
    pos = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            h = _line_hash(i + 1, line)
            result["lines"][h] = {"line_num": i + 1, "content": line, "offset": pos, "length": len(line)}
        pos += len(line) + 1
    return lines, result


def _parse_python(content, lines, result):
    """Python AST 解析器：利用 ast 模块提取 import/function/class/decorator。
    是 tree-sitter 的纯标准库替代方案。"""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                line = lines[node.lineno - 1]
                if not line.strip(): continue
                h = _line_hash(node.lineno, line)
                result["blocks"].append({"type": "import", "name": alias.name, "hash": h,
                    "offset": _calc_offset(lines, node.lineno, node.lineno), "length": len(line),
                    "start": node.lineno, "end": node.lineno, "content": line})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                ds, de = dec.lineno, (dec.end_lineno if hasattr(dec, 'end_lineno') and dec.end_lineno else dec.lineno)
                dc = '\n'.join(lines[ds-1:de])
                result["blocks"].append({"type": "decorator", "name": dc.strip(), "hash": _block_hash(dc),
                    "offset": _calc_offset(lines, ds, de), "length": len(dc), "start": ds, "end": de, "content": dc})
            bc = '\n'.join(lines[node.lineno-1:node.end_lineno])
            result["blocks"].append({"type": "function", "name": node.name, "hash": _block_hash(bc),
                "offset": _calc_offset(lines, node.lineno, node.end_lineno), "length": len(bc),
                "start": node.lineno, "end": node.end_lineno, "content": bc})
        elif isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                ds, de = dec.lineno, (dec.end_lineno if hasattr(dec, 'end_lineno') and dec.end_lineno else dec.lineno)
                dc = '\n'.join(lines[ds-1:de])
                result["blocks"].append({"type": "decorator", "name": dc.strip(), "hash": _block_hash(dc),
                    "offset": _calc_offset(lines, ds, de), "length": len(dc), "start": ds, "end": de, "content": dc})
            cc = '\n'.join(lines[node.lineno-1:node.end_lineno])
            cb = {"type": "class", "name": node.name, "hash": _block_hash(cc),
                "offset": _calc_offset(lines, node.lineno, node.end_lineno), "length": len(cc),
                "start": node.lineno, "end": node.end_lineno, "content": cc, "sub_blocks": []}
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in item.decorator_list:
                        ds, de = dec.lineno, (dec.end_lineno if hasattr(dec, 'end_lineno') and dec.end_lineno else dec.lineno)
                        dc = '\n'.join(lines[ds-1:de])
                        result["blocks"].append({"type": "decorator", "name": dc.strip(), "hash": _block_hash(dc),
                            "offset": _calc_offset(lines, ds, de), "length": len(dc), "start": ds, "end": de, "content": dc})
                    mc = '\n'.join(lines[item.lineno-1:item.end_lineno])
                    cb["sub_blocks"].append({"type": "method", "name": f"{node.name}.{item.name}",
                        "hash": _block_hash(mc), "offset": _calc_offset(lines, item.lineno, item.end_lineno),
                        "length": len(mc), "start": item.lineno, "end": item.end_lineno, "content": mc})
            result["blocks"].append(cb)
        else:
            line = lines[node.lineno - 1]
            if not line.strip(): continue
            end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else node.lineno
            bc = '\n'.join(lines[node.lineno - 1:end_line])
            if node.lineno == end_line:
                h, btype = _line_hash(node.lineno, line), "global"
            else:
                h, btype = _block_hash(bc), "global"
            result["blocks"].append({"type": btype, "name": line.strip()[:30], "hash": h,
                "offset": _calc_offset(lines, node.lineno, end_line), "length": len(bc),
                "start": node.lineno, "end": end_line, "content": bc})


# ==============================================================================
# C 家族语言解析器 —— 正则匹配 class/function/typedef/macro/annotation
# ==============================================================================
# 支持: C/C++/Java/C#/JavaScript/TypeScript/Go/Rust/Kotlin/Swift 等

_BRACE_CLASS_RE = re.compile(
    r'(?:(?:public|private|protected|static|abstract|sealed|internal|partial)\s+)?\s*'
    r'(class|interface|struct|enum|union|record)\s+(\w+)'
    r'(?:\s*:\s*([^{]+)\s*)?'
)
_BRACE_FUNC_RE = re.compile(
    r'(?:(?:public|private|protected|static|abstract|virtual|override|sealed|inline|const|explicit|constexpr|noexcept)\s+)*'
    r'(?:[\w<>*&:,]+\s+)*'
    r'(?:\w+(?:<[^>(;]*>)?\s*::\s*)?'
    r'(\w+)\s*\(', re.MULTILINE)
_BRACE_ANNOT_RE = re.compile(r'^@(\w+)')
_BRACE_MACRO_RE = re.compile(r'#\s*define\s+(\w+)')


def _parse_brace_lang(content, lines, result):
    """花括号语言解析器：逐行扫描，按优先级匹配。
    优先级: import > #define > typedef > 注解 > class/struct > 函数 > JS箭头函数 > 全局变量"""
    lines_len = len(lines)
    i = 0
    while i < lines_len:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith('/*'):
            i += 1
            continue
        if stripped.startswith('*') or stripped.startswith('//'):
            i += 1
            continue
        if stripped in ('{', '}', '};', ''):
            i += 1
            continue
        # 1. import/using/include/package
        if stripped.startswith(('import ', 'using ', '#include', 'package ', '#import ')):
            result["blocks"].append({"type": "import", "name": stripped[:60], "hash": _line_hash(i + 1, line),
                "offset": _calc_offset(lines, i + 1, i + 1), "length": len(line),
                "start": i + 1, "end": i + 1, "content": line})
            i += 1
            continue
        # 2. #define 宏定义（支持多行 \ 续行）
        mac = _BRACE_MACRO_RE.match(stripped)
        if mac:
            end_line = i + 1
            for j in range(i, lines_len):
                if not lines[j].rstrip().endswith('\\'):
                    end_line = j + 1
                    break
            mc = '\n'.join(lines[i:end_line])
            result["blocks"].append({"type": "macro", "name": mac.group(1), "hash": _block_hash(mc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(mc),
                "start": i + 1, "end": end_line, "content": mc})
            i = end_line
            continue
        # 3. typedef 类型定义
        if stripped.startswith('typedef'):
            if re.match(r'typedef\s+(struct|union|enum)\s', stripped):
                end_line = _find_brace_end(lines, i + 1)
                bc = '\n'.join(lines[i:end_line])
                name_m = re.search(r'}\s*(\w+)', lines[min(end_line, lines_len) - 1])
                name = name_m.group(1) if name_m else stripped[:50]
            else:
                end_line = i + 1
                for j in range(i, lines_len):
                    if ';' in lines[j]:
                        end_line = j + 1
                        break
                bc = '\n'.join(lines[i:end_line])
                if end_line == i + 1:
                    name_m = re.search(r'typedef\s+.+?(\w+)\s*;', stripped)
                else:
                    name_m = re.search(r'(\w+)\s*;\s*$', lines[min(end_line, lines_len) - 1].strip())
                name = name_m.group(1) if name_m else stripped[:50]
            result["blocks"].append({"type": "global", "name": f"typedef {name}", "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc})
            i = end_line
            continue
        # 4. 注解（Java/C#）
        am = _BRACE_ANNOT_RE.match(stripped)
        if am:
            result["blocks"].append({"type": "annotation", "name": stripped[:50], "hash": _line_hash(i + 1, line),
                "offset": _calc_offset(lines, i + 1, i + 1), "length": len(line),
                "start": i + 1, "end": i + 1, "content": line})
            i += 1
            continue
        # 5. class/struct/union/enum/interface（支持 export 前缀）
        check_stripped = re.sub(r'^export\s+(?:default\s+)?', '', stripped)
        cm = _BRACE_CLASS_RE.match(check_stripped)
        if cm:
            name = cm.group(2)
            parent = cm.group(3)
            display_name = f"{name}:{parent.strip()}" if parent else name
            end_line = _find_brace_end(lines, i + 1)
            bc = '\n'.join(lines[i:end_line])
            cb = {"type": "class", "name": display_name, "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc, "sub_blocks": []}
            for j in range(i, end_line):
                inner = lines[j].strip()
                if not inner or inner.startswith(('//', '*', '@', 'import', 'using')):
                    continue
                if inner in ('{', '}', ''):
                    continue
                if ';' in inner and '{' not in inner:
                    continue
                fm = _BRACE_FUNC_RE.match(inner)
                if fm and fm.group(1) not in _SKIP_KEYWORDS:
                    m_end = _find_brace_end(lines, j + 1)
                    mc = '\n'.join(lines[j:m_end])
                    cb["sub_blocks"].append({"type": "method", "name": f"{name}.{fm.group(1)}",
                        "hash": _block_hash(mc), "offset": _calc_offset(lines, j + 1, m_end),
                        "length": len(mc), "start": j + 1, "end": m_end, "content": mc})
            result["blocks"].append(cb)
            i = end_line
            continue
        # 6. 函数定义或声明
        fm = _BRACE_FUNC_RE.match(stripped)
        if fm and fm.group(1) not in _SKIP_KEYWORDS:
            has_body = '{' in stripped
            is_decl = stripped.endswith(';')
            if not has_body and not is_decl:
                for la in range(1, min(5, lines_len - i)):
                    nxt = lines[i + la].strip()
                    if '{' in nxt:
                        has_body = True
                        break
                    if ';' in nxt:
                        is_decl = True
                        break
            if has_body:
                end_line = _find_brace_end(lines, i + 1)
                fc = '\n'.join(lines[i:end_line])
                result["blocks"].append({"type": "function", "name": fm.group(1), "hash": _block_hash(fc),
                    "offset": _calc_offset(lines, i + 1, end_line), "length": len(fc),
                    "start": i + 1, "end": end_line, "content": fc})
                i = end_line
            elif is_decl:
                end_line = i + 1
                for j in range(i, lines_len):
                    if ';' in lines[j]:
                        end_line = j + 1
                        break
                fc = '\n'.join(lines[i:end_line])
                result["blocks"].append({"type": "decl", "name": fm.group(1), "hash": _block_hash(fc),
                    "offset": _calc_offset(lines, i + 1, end_line), "length": len(fc),
                    "start": i + 1, "end": end_line, "content": fc})
                i = end_line
            else:
                i += 1
            continue
        # 7. JS/TS 特定：箭头函数、对象字面量、原型赋值
        js_decl = re.match(r'(?:export\s+(?:default\s+)?)?(const|let|var)\s+', stripped)
        if js_decl:
            arrow_m = re.match(
                r'(?:export\s+(?:default\s+)?)?(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|\w+)\s*=>\s*\{',
                stripped)
            if arrow_m:
                end_line = _find_brace_end(lines, i + 1)
                fc = '\n'.join(lines[i:end_line])
                result["blocks"].append({"type": "function", "name": arrow_m.group(1), "hash": _block_hash(fc),
                    "offset": _calc_offset(lines, i + 1, end_line), "length": len(fc),
                    "start": i + 1, "end": end_line, "content": fc})
                i = end_line
                continue
            obj_m = re.match(
                r'(?:export\s+(?:default\s+)?)?(?:const|let|var)\s+(\w+)\s*=\s*\{',
                stripped)
            if obj_m:
                end_line = _find_brace_end(lines, i + 1)
                oc = '\n'.join(lines[i:end_line])
                result["blocks"].append({"type": "global", "name": obj_m.group(1), "hash": _block_hash(oc),
                    "offset": _calc_offset(lines, i + 1, end_line), "length": len(oc),
                    "start": i + 1, "end": end_line, "content": oc})
                i = end_line
                continue
        proto_m = re.match(r'(\w+)\.prototype\.(\w+)\s*=\s*function', stripped)
        if proto_m:
            end_line = _find_brace_end(lines, i + 1)
            pc = '\n'.join(lines[i:end_line])
            result["blocks"].append({"type": "method", "name": f"{proto_m.group(1)}.{proto_m.group(2)}",
                "hash": _block_hash(pc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(pc),
                "start": i + 1, "end": end_line, "content": pc})
            i = end_line
            continue
        # 8. 全局变量/其他顶层语句
        if stripped.endswith(';'):
            result["blocks"].append({"type": "global", "name": stripped[:60], "hash": _line_hash(i + 1, line),
                "offset": _calc_offset(lines, i + 1, i + 1), "length": len(line),
                "start": i + 1, "end": i + 1, "content": line})
            i += 1
            continue
        i += 1


# ==============================================================================
# SQL 解析器 —— DDL 语句块提取 (CREATE TABLE/ALTER/DROP)
# ==============================================================================
# _SQL_NAME 支持四种引用方式: `name`, "name", [name], name

_SQL_NAME = r'(?:`(\w+)`|"(\w+)"|\[(\w+)\]|(\w+))'
_SQL_CREATE_RE = re.compile(
    r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?'
    r'(?:UNIQUE\s+|FULLTEXT\s+|SPATIAL\s+|CLUSTERED\s+|NONCLUSTERED\s+)?'
    r'(TABLE|VIEW|PROCEDURE|FUNCTION|INDEX|TRIGGER|EVENT|DATABASE|SCHEMA|SEQUENCE|TYPE|DOMAIN)'
    r'\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`?\w+`?\.)?' + _SQL_NAME,
    re.IGNORECASE)
_SQL_ALTER_RE = re.compile(
    r'ALTER\s+(TABLE|VIEW|PROCEDURE|FUNCTION|INDEX|DATABASE|SCHEMA|SEQUENCE)'
    r'\s+(?:`?\w+`?\.)?' + _SQL_NAME,
    re.IGNORECASE)
_SQL_DROP_RE = re.compile(
    r'DROP\s+(?:TABLE|VIEW|INDEX|PROCEDURE|FUNCTION|TRIGGER|SEQUENCE|TYPE|DOMAIN|SCHEMA|DATABASE|EVENT)'
    r'\s+(?:IF\s+EXISTS\s+)?(?:`?\w+`?\.)?' + _SQL_NAME,
    re.IGNORECASE)
_SQL_TRUNCATE_RE = re.compile(
    r'TRUNCATE\s+(?:TABLE\s+)?(?:`?\w+`?\.)?' + _SQL_NAME,
    re.IGNORECASE)
_SQL_DDL_KEYWORDS = ('CREATE', 'ALTER', 'DROP', 'TRUNCATE')
_MULTILINE_DDL = ('procedure', 'function', 'trigger')


def _extract_sql_name(m):
    for g in range(1, 5):
        val = m.group(g + 1)
        if val is not None:
            return val
    return 'unknown'


def _find_next_ddl(lines: list, start: int) -> int:
    for j in range(start, len(lines)):
        s = lines[j].strip().upper()
        if any(s.startswith(kw) for kw in _SQL_DDL_KEYWORDS):
            return j
    return len(lines)


def _find_sql_block_end(lines: list, start: int) -> int:
    """找到SQL块的结束位置（下一个 ; ）"""
    for j in range(start, len(lines)):
        if ';' in lines[j]:
            return j + 1
    return len(lines)


def _parse_sql(content, lines, result):
    """SQL DDL 解析：CREATE/ALTER/DROP 语句 → block。每句至分号结束。"""
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith(('--', '/*', '*', '//')):
            i += 1
            continue
        # 1. CREATE 语句（含 MATERIALIZED VIEW, SEQUENCE, TYPE 等）
        crm = _SQL_CREATE_RE.match(stripped)
        if crm:
            obj_type = crm.group(1).lower()
            name = _extract_sql_name(crm)
            is_materialized = 'materialized' in stripped[:30].lower()
            type_map = {'table': 'class', 'view': 'class', 'procedure': 'function',
                        'function': 'function', 'index': 'class', 'trigger': 'function',
                        'sequence': 'global', 'type': 'global', 'domain': 'global',
                        'schema': 'class'}
            btype = type_map.get(obj_type, 'class')
            if is_materialized:
                name = f"mv_{name}"
            # PROCEDURE/FUNCTION/TRIGGER 需跨越 BEGIN...END，用 DDL 边界
            if obj_type in _MULTILINE_DDL:
                end_line = _find_next_ddl(lines, i + 1)
            else:
                end_line = _find_sql_block_end(lines, i)
            bc = '\n'.join(lines[i:end_line])
            result["blocks"].append({"type": btype, "name": name, "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc})
            i = end_line
            continue
        # 2. ALTER 语句
        alm = _SQL_ALTER_RE.match(stripped)
        if alm:
            name = _extract_sql_name(alm)
            end_line = _find_sql_block_end(lines, i)
            bc = '\n'.join(lines[i:end_line])
            result["blocks"].append({"type": "global", "name": f"alter {name}", "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc})
            i = end_line
            continue
        # 3. DROP 语句
        drm = _SQL_DROP_RE.match(stripped)
        if drm:
            name = _extract_sql_name(drm)
            end_line = _find_sql_block_end(lines, i)
            bc = '\n'.join(lines[i:end_line])
            result["blocks"].append({"type": "global", "name": f"drop {name}", "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc})
            i = end_line
            continue
        # 4. TRUNCATE 语句
        trm = _SQL_TRUNCATE_RE.match(stripped)
        if trm:
            name = _extract_sql_name(trm)
            end_line = _find_sql_block_end(lines, i)
            bc = '\n'.join(lines[i:end_line])
            result["blocks"].append({"type": "global", "name": f"truncate {name}", "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc})
            i = end_line
            continue
        # 5. DML 语句（INSERT / UPDATE / DELETE / SELECT）作为 global
        dml_keywords = ('INSERT ', 'UPDATE ', 'DELETE ', 'SELECT ')
        if stripped.upper().startswith(dml_keywords):
            end_line = _find_sql_block_end(lines, i)
            bc = '\n'.join(lines[i:end_line])
            name = stripped[:60]
            result["blocks"].append({"type": "global", "name": name, "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc})
            i = end_line
            continue
        i += 1


# ========== CSS 解析器 ==========

_CSS_AT_IMPORT_RE = re.compile(r'@import\s+(?:url\(["\']?(.+?)["\']?\)|["\'](.+?)["\'])', re.IGNORECASE)
_CSS_AT_RULE_RE = re.compile(r'@(media|supports|keyframes|font-face|page|counter-style|property|layer|viewport)', re.IGNORECASE)
_CSS_KEYFRAME_RE = re.compile(r'@keyframes\s+(\w+)', re.IGNORECASE)


def _parse_css(content, lines, result):
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith(('/*', '*')) or stripped.startswith('//'):
            i += 1
            continue
        # 1. @import
        if stripped.startswith('@import'):
            im = _CSS_AT_IMPORT_RE.match(stripped)
            name = im.group(1) or im.group(2) if im else stripped[:50]
            end_line = i + 1
            for j in range(i, len(lines)):
                if ';' in lines[j]:
                    end_line = j + 1
                    break
            bc = '\n'.join(lines[i:end_line])
            result["blocks"].append({"type": "import", "name": name, "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc})
            i = end_line
            continue
        # 2. @media / @supports / @keyframes / @font-face 等
        arm = _CSS_AT_RULE_RE.match(stripped)
        if arm:
            rule_type = arm.group(1).lower()
            brace_line = i
            has_brace = '{' in stripped
            if not has_brace:
                for j in range(i + 1, min(i + 30, len(lines))):
                    if '{' in lines[j]:
                        brace_line = j
                        has_brace = True
                        break
            if has_brace:
                end_line = _find_brace_end(lines, brace_line)
                bc = '\n'.join(lines[i:end_line])
                name_part = stripped[len(f'@{rule_type}'):].strip().rstrip('{').strip()
                name = f"@{rule_type} {name_part}" if name_part else f"@{rule_type}"
                result["blocks"].append({"type": "class", "name": name, "hash": _block_hash(bc),
                    "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                    "start": i + 1, "end": end_line, "content": bc})
                i = end_line
                continue
        # 3. 选择器块（包含 { 的行）
        if '{' in stripped:
            has_open = '{' in stripped
            has_close = '}' in stripped
            is_multiline = not has_close or stripped.index('{') > stripped.rindex('}')
            if is_multiline:
                # 多行选择器，找到 { 所在行
                brace_line = i
                if not stripped.rstrip().endswith('{'):
                    for j in range(i + 1, min(i + 20, len(lines))):
                        if '{' in lines[j]:
                            brace_line = j
                            break
                end_line = _find_brace_end(lines, brace_line)
            else:
                end_line = i + 1  # 单行块
            bc = '\n'.join(lines[i:end_line])
            first_sel = stripped.split('{')[0].strip().rstrip(',').strip()
            name = first_sel[:60]
            result["blocks"].append({"type": "class", "name": name, "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, end_line), "length": len(bc),
                "start": i + 1, "end": end_line, "content": bc})
            i = end_line
            continue
        i += 1


# ========== HTML 解析器 ==========

_HTML_TAGS = {'html', 'head', 'body', 'header', 'footer', 'main', 'nav', 'aside',
    'section', 'article', 'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'table', 'thead', 'tbody', 'tfoot',
    'tr', 'td', 'th', 'caption', 'colgroup', 'col', 'title',
    'form', 'fieldset', 'legend', 'label', 'select', 'optgroup', 'option',
    'textarea', 'button', 'datalist', 'output', 'progress', 'meter',
    'details', 'summary', 'dialog', 'figure', 'figcaption', 'picture', 'source',
    'video', 'audio', 'canvas', 'iframe', 'embed', 'object', 'param',
    'template', 'slot', 'blockquote', 'pre', 'code', 'address',
    'search', 'mark', 'time', 'data', 'ins', 'del'}
_HTML_VOID_TAGS = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
    'link', 'meta', 'param', 'source', 'track', 'wbr'}


def _parse_html(content, lines, result):
    _parse_html_blocks(lines, 0, len(lines), result["blocks"], 0)


_HTML_ALWAYS_BLOCK = {'header', 'main', 'footer', 'nav', 'aside',
    'section', 'article', 'form', 'blockquote', 'pre', 'code',
    'figure', 'details', 'dialog', 'fieldset', 'search', 'table'}
_HTML_CONTAINER = {'html', 'head', 'body'}


def _find_tag_end(lines, start, tag_name, limit):
    depth = 0
    for j in range(start, min(limit, len(lines))):
        l = lines[j]
        depth += len(re.findall(rf'<{tag_name}(?:\s[^>]*)?>', l))
        depth += len(re.findall(rf'<{tag_name}(?:\s[^>]*)?/>', l))
        depth -= l.count(f'</{tag_name}>')
        if depth <= 0 and j > start:
            return j + 1
    return limit


def _parse_html_blocks(lines, start, end, result, depth):
    i = start
    while i < end and i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith('<!--'):
            for j in range(i, min(end, len(lines))):
                if '-->' in lines[j]:
                    i = j + 1
                    break
            else:
                i += 1
            continue
        if stripped.startswith(('<!DOCTYPE', '<?', '<%')):
            i += 1
            continue
        if re.match(r'\s*</?(?:' + '|'.join(_HTML_CONTAINER) + r')\s*>', stripped, re.IGNORECASE):
            i += 1
            continue
        ss_match = re.match(r'\s*<(script|style)([^>]*)>', stripped)
        if ss_match:
            tag_name = ss_match.group(1)
            tag_end = i + 1
            for j in range(i + 1, min(end, len(lines))):
                if f'</{tag_name}>' in lines[j]:
                    tag_end = j + 1
                    break
            if tag_end > end:
                tag_end = end
            bc = '\n'.join(lines[i:tag_end])
            id_m = re.search(r'\sid\s*=\s*["\'](\w+)["\']', stripped)
            name = id_m.group(1) if id_m else tag_name
            result.append({"type": "function", "name": name, "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, tag_end), "length": len(bc),
                "start": i + 1, "end": tag_end, "content": bc})
            i = tag_end
            continue
        tmpl_match = re.match(r'\s*<template([^>]*)>', stripped)
        if tmpl_match:
            tag_end = i + 1
            for j in range(i + 1, min(end, len(lines))):
                if '</template>' in lines[j]:
                    tag_end = j + 1
                    break
            if tag_end > end:
                tag_end = end
            bc = '\n'.join(lines[i:tag_end])
            id_m = re.search(r'\sid\s*=\s*["\'](\w+)["\']', stripped)
            name = id_m.group(1) if id_m else 'template'
            result.append({"type": "class", "name": f"template#{name}", "hash": _block_hash(bc),
                "offset": _calc_offset(lines, i + 1, tag_end), "length": len(bc),
                "start": i + 1, "end": tag_end, "content": bc})
            i = tag_end
            continue
        title_match = re.match(r'\s*<title([^>]*)>(.*?)</title>\s*$', stripped, re.IGNORECASE)
        if title_match:
            title_text = title_match.group(2).strip()
            btype = 'global'
            result.append({"type": btype, "name": f"title#{title_text}" if title_text else "title",
                "hash": _line_hash(i + 1, line),
                "offset": _calc_offset(lines, i + 1, i + 1), "length": len(line),
                "start": i + 1, "end": i + 1, "content": line})
            i += 1
            continue
        void_match = re.match(r'\s*<(\w+)([^>]*?)\s*/?>', stripped)
        if void_match and void_match.group(1) in _HTML_VOID_TAGS:
            tag_name = void_match.group(1)
            btype = 'import' if tag_name in ('link', 'meta', 'base') else 'global'
            name_attr = re.search(r'(?:name|rel|href|src|action)="([^"]+)"', stripped)
            display = name_attr.group(1) if name_attr else tag_name
            result.append({"type": btype, "name": f"{tag_name}#{display}", "hash": _line_hash(i + 1, line),
                "offset": _calc_offset(lines, i + 1, i + 1), "length": len(line),
                "start": i + 1, "end": i + 1, "content": line})
            i += 1
            continue
        tag_match = re.match(r'\s*<(\w+)([^>]*)>', stripped)
        if tag_match:
            tag_name = tag_match.group(1)
            if tag_name not in _HTML_TAGS or tag_name in _HTML_CONTAINER:
                i += 1
                continue
            has_id = bool(re.search(r'\sid\s*=\s*["\'](\w+)["\']', stripped))
            if tag_name in _HTML_ALWAYS_BLOCK or has_id:
                tag_end = _find_tag_end(lines, i, tag_name, end)
                if tag_end > end:
                    tag_end = end
                bc = '\n'.join(lines[i:tag_end])
                id_m = re.search(r'\sid\s*=\s*["\'](\w+)["\']', stripped)
                cls_m = re.search(r'\sclass\s*=\s*["\'](\w+)["\']', stripped)
                name = id_m.group(1) if id_m else (cls_m.group(1) if cls_m else tag_name)
                sub_blocks = []
                if tag_end - i > 1:
                    _parse_html_blocks(lines, i + 1, tag_end - 1, sub_blocks, depth + 1)
                block = {"type": "class", "name": f"{tag_name}#{name}", "hash": _block_hash(bc),
                    "offset": _calc_offset(lines, i + 1, tag_end), "length": len(bc),
                    "start": i + 1, "end": tag_end, "content": bc}
                if sub_blocks:
                    block["sub_blocks"] = sub_blocks
                result.append(block)
                i = tag_end
                continue
        i += 1


register_parser('python', _parse_python)
register_parser('java', _parse_brace_lang)
register_parser('cpp', _parse_brace_lang)
register_parser('c', _parse_brace_lang)
register_parser('csharp', _parse_brace_lang)
register_parser('javascript', _parse_brace_lang)
register_parser('typescript', _parse_brace_lang)
register_parser('go', _parse_brace_lang)
register_parser('rust', _parse_brace_lang)
register_parser('kotlin', _parse_brace_lang)
register_parser('swift', _parse_brace_lang)
register_parser('ruby', _parse_brace_lang)
register_parser('php', _parse_brace_lang)
register_parser('sql', _parse_sql)
register_parser('html', _parse_html)
register_parser('css', _parse_css)


# ==============================================================================
# 对外 API —— file_tool.py 调用的接口
# ==============================================================================

def generate_hash_index(content: str, file_path: str = '') -> dict:
    """
    主入口：生成代码文件的 HASH INDEX。
    1. 检测语言 → 2. 按行生成 hash → 3. 调用语言解析器提取块 → 4. 返回完整索引
    """
    lang = _detect_language(file_path)
    lines, result = _make_lines_result(content)
    parser = _PARSERS.get(lang)
    if parser:
        parser(content, lines, result)
    return result


def format_hash_index(hash_index: dict, file_path: str) -> str:
    """
    将 HASH INDEX 格式化为可读的注释文本。
    read_file 返回的文件内容末尾会附加这段文本，供 LLM 和 edit_file_batch 使用。

    输出格式示例:
      # ========== HASH INDEX [core/agent.py] ==========
      # a3b2c1d4e5f6 | functi | L108-L156 | create_agent_instance
      # b4c5d6e7f8a9 | class  | L28-L97   | smart_context_trimmer
      # 8f7e6d5c4b3a | line   | L15       | import os
      # ===========================================
    """
    parts = [f"\n# ========== HASH INDEX [{file_path}] =========="]
    parts.append("# 格式: hash值 | 类型 | 位置 | 名称")
    parts.append("# 使用 edit_file_batch 进行批量修改，填入对应的hash值")

    if hash_index["blocks"]:
        for block in hash_index["blocks"]:
            _format_block(block, parts)
    else:
        parts.append("# （该文件未识别到结构块，仅支持按行内容hash编辑——使用方括号 [] 包裹的行hash）")
        for h, info in sorted(hash_index.get("lines", {}).items(), key=lambda x: x[1]["line_num"]):
            parts.append(f"# [{h}] L{info['line_num']} | {info['content'][:60]}")

    parts.append("# ===========================================")
    return '\n'.join(parts)


def _format_block(block, parts, indent=0):
    """递归格式化块（含子块如 class 中的 method）"""
    prefix = '  ' * indent
    if block['start'] == block['end']:
        loc = f"L{block['start']}"
    else:
        loc = f"L{block['start']}-L{block['end']}"
    btype = block['type'].ljust(6)
    parts.append(f"# {block['hash']} | {btype} | {loc} | {prefix}{block['name']}")
    for sub in block.get("sub_blocks", []):
        _format_block(sub, parts, indent + 1)


def _search_block(block, target_hash):
    """递归在块（含子块）中查找指定 hash"""
    if block["hash"] == target_hash:
        return block
    for sub in block.get("sub_blocks", []):
        result = _search_block(sub, target_hash)
        if result:
            return result
    return None


def resolve_block_by_hash(hash_index: dict, target_hash: str) -> dict:
    """
    根据 hash 查找对应的代码块/行信息。
    先在 blocks 中查，未找到再到 lines 中查。
    返回 None 表示 hash 不存在于索引中。
    """
    for block in hash_index["blocks"]:
        result = _search_block(block, target_hash)
        if result:
            return result

    if target_hash in hash_index.get("lines", {}):
        info = hash_index["lines"][target_hash]
        return {
            "type": "line",
            "name": info["content"][:30],
            "hash": target_hash,
            "offset": info["offset"],
            "length": info["length"],
            "start": info["line_num"],
            "end": info["line_num"],
            "content": info["content"]
        }

    return None


def validate_batch_changes(changes: list, hash_index: dict) -> tuple:
    """
    校验批量修改的合法性。
    检查项：
    1. 修改列表非空
    2. 每个 change 的 hash 在索引中存在
    3. 多个 change 的 offset 范围不重叠（防止互相覆盖）

    返回 (is_valid, error_msg)
    """
    if not changes:
        return False, "修改列表为空"

    ranges = []
    for i, c in enumerate(changes):
        target_hash = c.get("hash")
        if not target_hash:
            return False, f"第{i + 1}项缺少hash值"

        block = resolve_block_by_hash(hash_index, target_hash)
        if block is None:
            return False, f"第{i + 1}项的hash '{target_hash}' 未找到"

        # 检测范围重叠
        r = (block["offset"], block["offset"] + block["length"])
        for j, existing in enumerate(ranges):
            a_s, a_e = r
            b_s, b_e = existing
            if max(a_s, b_s) < min(a_e, b_e):
                return False, f"第{i + 1}项与第{j + 1}项范围重叠"
        ranges.append(r)

    return True, ""


def apply_batch_changes(content: str, changes: list, hash_index: dict) -> tuple:
    """
    批量应用修改。关键：**从下往上应用**。
    原因：如果从上往下，前面的修改会改变后续行的 offset，
    导致后面的 hash 定位错误。从下往上则互不干扰。

    返回 (new_content, error_msg)
    """
    items = []
    for c in changes:
        block = resolve_block_by_hash(hash_index, c["hash"])
        if block is None:
            return None, f"hash '{c['hash']}' 未找到"
        items.append({
            "hash": c["hash"],
            "offset": block["offset"],
            "old_len": block["length"],
            "old_content": block["content"],
            "new_content": c.get("new_content", ""),
            "mode": c.get("mode", "replace"),
            "name": block["name"]
        })

    # 按 offset 从大到小排序（从下往上应用）
    items.sort(key=lambda x: x["offset"], reverse=True)

    result = content
    for item in items:
        start = item["offset"]
        end = start + item["old_len"]

        # 校验位置上的内容是否一致
        actual = result[start:end]
        if actual != item["old_content"]:
            return None, f"'{item['name']}' 位置内容校验失败，文件可能已被其他方式修改"

        new_content = item["new_content"]
        mode = item["mode"]

        if mode == "replace":
            if new_content == "" and item["old_len"] > 0:
                result = result[:start] + result[end:]
            else:
                result = result[:start] + new_content + result[end:]

        elif mode == "insert_before":
            result = result[:start] + new_content + '\n' + result[start:]

        elif mode == "insert_after":
            result = result[:start] + result[start:end] + '\n' + new_content + result[end:]

    return result, None
