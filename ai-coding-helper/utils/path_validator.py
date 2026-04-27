"""
================================================================================
路径安全校验 —— Agent 文件操作的安全网关
================================================================================

【模块职责】
所有 Agent 发起的文件/文件夹操作都必须通过本模块的安全校验。
防止：
  1. 越权访问工作区外的路径（路径穿越攻击）
  2. 操作系统级敏感目录（.git、.svn 等版本控制目录）
  3. 操作符号链接（可能指向外部路径）
  4. 操作不支持的文件类型（图片、二进制等）

【校验流程】
check_file_safety(file_path, workspace):
  1. 路径标准化（posixpath.normpath，消除 ../ 穿越）
  2. 绝对路径拼接并检查是否在工作区内
  3. 目录检查（文件操作不允许传目录）
  4. 路径分段检查（每段是否有禁止目录）
  5. 符号链接检查
  6. 文件名白名单（CMakeLists.txt 等特殊文件）
  7. 后缀黑名单 × 白名单双重校验

【iter_workspace_files 工具函数】
遍历工作区所有代码文件，自动过滤禁止目录和禁止后缀。
被 handler.py/file_tool.py 等多处复用。
"""
import os
import posixpath
from ui.config import ALLOWED_EXT, FORBIDDEN_DIR, FORBIDDEN_EXT, ALLOWED_FILENAMES

# ==============================================================================
# check_file_safety —— 文件路径安全校验（防越权访问）
# ==============================================================================
def check_file_safety(file_path: str, workspace: str) -> tuple[bool, str]:
    """
    对 Agent 请求的单个文件路径做安全校验。

    返回 (is_safe, message)：
      - (True, "安全") → 可以通过
      - (False, "原因") → 拒绝并返回原因给 Agent

    攻击场景示例：
      用户输入: ../../etc/passwd
      posixpath.normpath 消除 ../ 后 → etc/passwd
      join(workspace, "etc/passwd") → /workspace/etc/passwd
      workspace = /workspace → 在工作区内 → 安全检查不拦截
      （文件系统仍会拒绝，因为 /workspace/etc/passwd 不存在）
    """
    if not workspace:
        return False, "未设置工作区域"

    try:
        # 路径标准化：消除 ../ 和 // 等冗余路径
        file_path = posixpath.normpath(file_path)
        workspace_abs = os.path.abspath(workspace)
        full_path = os.path.abspath(os.path.join(workspace_abs, file_path))
    except Exception as e:
        return False, f"路径解析失败：{str(e)}"

    # 越权检查：最终绝对路径必须在工作区路径以内
    if not full_path.startswith(workspace_abs):
        return False, f"禁止越权访问：{full_path} 不在工作区 {workspace_abs} 内"

    # 目录检查：文件工具只能操作文件，不能操作目录
    if os.path.isdir(full_path):
        return False, f"{file_path} 是目录，禁止操作"

    # 路径片段检查：每层目录都不能在禁止列表中
    path_parts = full_path.split(os.sep)
    for part in path_parts:
        if part in FORBIDDEN_DIR:
            return False, f"禁止操作系统/版本目录：{part}"

    # 符号链接检查：防止通过符号链接越权访问工作区外
    if os.path.islink(full_path):
        return False, f"{file_path} 是符号链接，禁止操作"

    # 文件名白名单优先：CMakeLists.txt 等特殊文件名直接放行
    basename = os.path.basename(full_path)
    if basename in ALLOWED_FILENAMES:
        return True, "安全"

    # 后缀双重校验：先查黑名单，再查白名单
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in FORBIDDEN_EXT:
        return False, f"禁止操作文档/配置文件：{ext}"
    if ext not in ALLOWED_EXT:
        return False, f"不支持的文件类型: {ext}（支持类型：{','.join(ALLOWED_EXT)}）"
    return True, "安全"


# ==============================================================================
# check_folder_safety —— 文件夹路径安全校验
# ==============================================================================
# 与文件校验的区别：不检查"是目录"（文件夹本身就是目录），其他规则相同
def check_folder_safety(folder_path: str, workspace: str) -> tuple[bool, str]:
    """对 Agent 请求的文件夹路径做安全校验"""
    if not workspace:
        return False, "未设置工作区域"
    try:
        folder_path = posixpath.normpath(folder_path)
        workspace_abs = os.path.abspath(workspace)
        full_path = os.path.abspath(os.path.join(workspace_abs, folder_path))
    except Exception as e:
        return False, f"路径解析失败：{str(e)}"

    if not full_path.startswith(workspace_abs):
        return False, f"禁止越权访问：{full_path} 不在工作区 {workspace_abs} 内"

    path_parts = full_path.split(os.sep)
    for part in path_parts:
        if part in FORBIDDEN_DIR:
            return False, f"禁止操作系统/版本目录：{part}"

    if os.path.islink(full_path):
        return False, f"{folder_path} 是符号链接，禁止操作"
    return True, "安全"


# ==============================================================================
# iter_workspace_files —— 工作区文件遍历工具
# ==============================================================================
# 被 handler.py（快照/DiffBlock/文件树/文件计数）和 file_tool.py（list_files）复用。
# 自动跳过禁止目录（dirs[:] = ... 原地过滤，os.walk 不再进入这些子目录）。
def iter_workspace_files(workspace: str, forbidden_dirs: set = None, allowed_exts: set = None):
    """
    遍历工作区内所有代码文件，自动过滤禁止目录和后缀。

    Yields: (rel_path: str, abs_path: str)
      例: ("core/agent.py", "/workspace/core/agent.py")
    """
    from ui.config import ALLOWED_EXT as _ALLOWED_EXT, FORBIDDEN_DIR as _FORBIDDEN_DIR
    if forbidden_dirs is None:
        forbidden_dirs = _FORBIDDEN_DIR
    if allowed_exts is None:
        allowed_exts = _ALLOWED_EXT

    for root, dirs, files in os.walk(workspace):
        # 原地过滤：修改 dirs 列表后 os.walk 自动跳过这些目录
        dirs[:] = [d for d in dirs if d not in forbidden_dirs]
        for f in files:
            ext = os.path.splitext(f)[-1].lower()
            if ext in allowed_exts:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, workspace)
                yield rel_path, abs_path
