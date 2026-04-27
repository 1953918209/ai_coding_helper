"""
================================================================================
文件工具集合 —— Agent 的"手脚"，所有文件操作的能力来源
================================================================================

【模块职责】
为 LangChain Agent 提供 14 个文件操作工具 + 5 个文件夹操作工具，包括：
  - 读取：read_file（附带 HASH INDEX，供精确批量修改定位）
  - 写入：edit_file（全量覆盖）、edit_file_batch（基于HASH的精确批量修改）
  - 创建/删除：create_file、delete_file
  - 复制/移动/重命名：copy_file、move_file、rename_file
  - 遍历：list_files
  - 文件夹：create_folder、delete_folder、rename_folder、move_folder、copy_folder

每个工具内部自动完成：路径安全校验 → 版本快照 → 执行操作 → RAG索引同步

【LangChain @tool 装饰器原理（深入）】
@tool 装饰器将普通 Python 函数注册为 LLM 可调用的"工具"：
1. 提取函数签名 → 生成 JSON Schema（参数名、类型、是否必填）
2. args_schema=XXXInput 用 Pydantic BaseModel 提供更丰富的参数描述
   - Field(description="...") → LLM 看这个描述来判断参数含义
   - Field(examples=[...]) → 给 LLM 举例说明参数格式
3. 函数的 docstring → 工具描述（LLM 据此决定是否调用此工具）
4. 生成 Tool 对象包含：.name / .description / .args_schema / .invoke()
5. LLM 推理时：看到工具列表 → 输出 tool_calls JSON → LangChain 解析 → 调用 .invoke()

配合 args_schema=XXXInput 用 Pydantic 模型定义参数（类型+描述），
LLM 根据参数描述自动生成符合类型的 tool_calls JSON。

【HASH INDEX 机制（edit_file_batch 的基础设施）】
read_file 读取文件时，系统自动生成 HASH INDEX：
  - 行级hash：每行代码的 MD5 指纹（用于定位）
  - 块级hash：每个代码块（函数/类/import等）的 SHA256 指纹（用于精确替换）

edit_file_batch 使用 hash 值精确定位要修改的代码块，而不是传行号。
优势：行号可能因其他修改而偏移，hash 值不受影响。
实现细节见 utils/hash_index.py。

【版本快照机制】
每次写入操作前，create_version_snapshot 自动备份当前文件到
.workspace_versions/ 目录。用户可通过 UI 的"版本回退"功能恢复。

【RAG 索引同步】
所有写入操作完成后自动调用 rag_manager.update_file_index()，
确保向量检索的数据与磁盘文件保持同步。

【面试高频题 —— 工具/Tool 设计】
Q8: LangChain 中如何定义工具？你们有19个工具怎么管理的？
A:  每个工具通过 @tool(args_schema=XXXInput) 装饰一个内部函数定义。
    所有工具用闭包捕获 workspace 和 app_root，在 get_tools() 中返回列表。
    最后 tool list 传给 create_agent() 注册为 Agent 可用的工具。

Q9: 为什么 edit_file 和 edit_file_batch 要分开？
A:  edit_file 是全量覆盖（适合新建文件/大改），必须传完整内容。
    edit_file_batch 基于 HASH INDEX 精确修改（适合改一个函数/插入代码），
    优势：只改目标代码块，不需要传整个文件内容，节省 token 且避免意外覆盖。

Q10: 工具的返回结果为什么要加标记（如 [EDIT_FILE_RESULT]）？
A:  这些标记供 handlers.py 的 _run_agent_stream 解析 AI 输出使用。
    UI 可以根据不同的标记类型做不同的展示（如 DiffBlock 只对 EDIT_FILE 生成）。
"""
import os, shutil, logging, tkinter as tk
from tkinter import messagebox
from langchain_core.tools import tool              # LangChain 工具装饰器
from pydantic import BaseModel, Field               # 入参模型校验
from utils.path_validator import check_file_safety, check_folder_safety
from utils.version_manager import create_version_snapshot
from core.rag.rag_manager import get_rag_manager
from utils.hash_index import (
    generate_hash_index, format_hash_index,
    resolve_block_by_hash, validate_batch_changes, apply_batch_changes
)

logger = logging.getLogger(__name__)

# ---- 安全限制：最大支持读取 50MB 的文件 ----
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ---- 工具返回结果标记 ----
# 每个工具返回的字符串以特定标记开头，供 UI 层识别解析
READ_FILE_MARK = "[READ_FILE_CONTENT]"
EDIT_FILE_MARK = "[EDIT_FILE_RESULT]"
DELETE_FILE_MARK = "[DELETE_FILE_RESULT]"
RENAME_FILE_MARK = "[RENAME_FILE_RESULT]"
COPY_FILE_MARK = "[COPY_FILE_RESULT]"
MOVE_FILE_MARK = "[MOVE_FILE_RESULT]"
CREATE_FILE_MARK = "[CREATE_FILE_RESULT]"
LIST_FILE_MARK = "[LIST_FILE_RESULT]"
CREATE_FOLDER_MARK = "[CREATE_FOLDER_RESULT]"
DELETE_FOLDER_MARK = "[DELETE_FOLDER_RESULT]"
RENAME_FOLDER_MARK = "[RENAME_FOLDER_RESULT]"
MOVE_FOLDER_MARK = "[MOVE_FOLDER_RESULT]"
COPY_FOLDER_MARK = "[COPY_FOLDER_RESULT]"

# ---- HASH INDEX 缓存 ----
# read_file 时写入，edit_file_batch 时读取，操作完成后清除
_hash_index_cache: dict[str, dict] = {}


# ==============================================================================
# 通用工具：自动生成不重复的文件路径
# ==============================================================================
# 当目标文件已存在时，自动追加 _1, _2 后缀，避免覆盖。
# 用于 copy/move/rename 操作（类似 Windows "复制 - 副本" 的行为）。
def get_unique_file_path(full_path: str) -> tuple[str, str]:
    """返回 (最终唯一路径, 提示信息)"""
    if not os.path.exists(full_path):
        return full_path, "目标文件不存在，可直接操作"
    dir_name, file_name = os.path.split(full_path)
    name, ext = os.path.splitext(file_name)
    counter = 1
    while True:
        new_path = os.path.join(dir_name, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path, f"目标文件已存在，自动生成副本：{os.path.basename(new_path)}"
        counter += 1


# ==============================================================================
# Pydantic 入参模型 —— 定义每个工具的参数结构
# ==============================================================================
# Field 的 description 是给 LLM 看的"使用说明"，
# examples 帮助 LLM 理解参数格式。

class EditFileInput(BaseModel):
    file_path: str = Field(description="代码文件相对工作区的路径", examples=["core/agent.py"])
    new_content: str = Field(description="文件完整新内容", examples=["# 完整代码"])
    reason: str = Field(description="修改理由", examples=["修复BUG"])

class DeleteFileInput(BaseModel):
    file_path: str = Field(description="待删除文件路径", examples=["temp.py"])

class ReadFileInput(BaseModel):
    file_path: str = Field(description="待读取文件路径", examples=["test.py"])

class EditChange(BaseModel):
    """edit_file_batch 的单次修改单元"""
    hash: str = Field(description="目标代码块的hash值（来自HASH INDEX）")
    new_content: str = Field(description="新代码内容，空字符串=删除该块")
    mode: str = Field(default="replace", description="replace=替换, insert_before=前插, insert_after=后插")

class EditFileBatchInput(BaseModel):
    file_path: str = Field(description="代码文件相对工作区的路径", examples=["core/agent.py"])
    changes: list[EditChange] = Field(description="批量修改列表")
    reason: str = Field(description="修改理由", examples=["批量修复"])

class RenameFileInput(BaseModel):
    old_file_path: str = Field(description="原文件路径")
    new_file_path: str = Field(description="新文件路径")
    reason: str = Field(description="重命名理由")

class CopyFileInput(BaseModel):
    source_file: str = Field(description="源文件路径")
    target_file: str = Field(description="目标文件路径")
    reason: str = Field(description="复制理由")

class MoveFileInput(BaseModel):
    source_file: str = Field(description="源文件路径")
    target_file: str = Field(description="目标文件路径")
    reason: str = Field(description="移动理由")

class CreateFileInput(BaseModel):
    file_path: str = Field(description="新建文件路径")
    reason: str = Field(description="创建理由")

class CreateFolderInput(BaseModel):
    folder_path: str = Field(description="新建文件夹路径（相对工作区）")
    reason: str = Field(description="创建理由")

class DeleteFolderInput(BaseModel):
    folder_path: str = Field(description="待删除文件夹路径")

class RenameFolderInput(BaseModel):
    old_folder_path: str = Field(description="原文件夹路径")
    new_folder_path: str = Field(description="新文件夹路径")
    reason: str = Field(description="重命名理由")

class MoveFolderInput(BaseModel):
    source_folder: str = Field(description="源文件夹路径")
    target_folder: str = Field(description="目标文件夹路径")
    reason: str = Field(description="移动理由")

class CopyFolderInput(BaseModel):
    source_folder: str = Field(description="源文件夹路径")
    target_folder: str = Field(description="目标文件夹路径")
    reason: str = Field(description="复制理由")


# ==============================================================================
# get_tools() —— 注册所有文件工具（含文件夹工具）
# ==============================================================================
# workspace 参数是工作区绝对路径，main_root 是 Tkinter 主窗口（用于弹窗）。
# 所有工具函数通过闭包捕获 workspace 和 app_root，避免全局变量污染。
def get_tools(workspace: str, main_root: tk.Tk):
    app_root = main_root

    # ------------------------------------------------------------------
    # read_file —— 读取文件（附带 HASH INDEX）
    # ------------------------------------------------------------------
    # 这是 Agent 使用最频繁的工具。返回值包含两部分：
    #   1. 文件完整内容（供 LLM 分析）
    #   2. HASH INDEX（供后续 edit_file_batch 精确定位代码块）
    #
    # HASH INDEX 写入 _hash_index_cache，edit_file_batch 从缓存中读取。
    # 缓存生命周期：read_file 写入 → edit_file_batch 消费后清除。
    @tool(args_schema=ReadFileInput)
    def read_file(**kwargs):
        """读取工作区内代码文件完整内容，附带HASH INDEX供edit_file_batch定位"""
        try:
            file_path = kwargs["file_path"]
            print(f"【工具调用】read_file | {file_path}")
            logger.info(f"【工具调用】read_file | {file_path}")

            # 安全校验：路径必须在工作区内 + 后缀允许 + 非符号链接
            safe, msg = check_file_safety(file_path, workspace)
            if not safe:
                return f"❌ 安全校验失败：{msg}"

            full_path = os.path.join(workspace, file_path)
            if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                return f"❌ 文件超过{MAX_FILE_SIZE_MB}MB"

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 生成 HASH INDEX 并缓存
            hash_index = generate_hash_index(content, file_path)
            _hash_index_cache[file_path] = hash_index
            hash_text = format_hash_index(hash_index, file_path)

            return f"{READ_FILE_MARK}\n✅ 读取成功\n文件：{file_path}\n内容：\n{content}\n{hash_text}"
        except Exception as e:
            logger.exception(f"读取失败：{str(e)}")
            return f"❌ 读取失败：{str(e)}"

    # ------------------------------------------------------------------
    # edit_file —— 全量覆盖写入
    # ------------------------------------------------------------------
    # 必须传入文件的完整新内容（不是 diff），这是 LLM 直接生成替换文本的方式。
    # 如果文件不存在 → 自动创建父目录 + 写入 + 创建初始快照。
    # 写入后同步：更新 RAG 索引 + 清除 HASH INDEX 缓存（内容变了，旧hash失效）。
    @tool(args_schema=EditFileInput)
    def edit_file(**kwargs):
        """全量覆盖修改文件，必须传入完整内容"""
        try:
            file_path = kwargs["file_path"]
            new_content = kwargs["new_content"]
            reason = kwargs["reason"]
            print(f"【工具调用】edit_file | {file_path} | {reason}")
            logger.info(f"【工具调用】edit_file | {file_path} | {reason}")

            safe, msg = check_file_safety(file_path, workspace)
            if not safe:
                return f"❌ 安全校验失败：{msg}"
            if not new_content.strip():
                return "❌ 禁止写入空内容"

            full_path = os.path.join(workspace, file_path)
            if os.path.exists(full_path):
                create_version_snapshot(full_path, workspace)  # 自动备份
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                _hash_index_cache.pop(file_path, None)  # 内容变了，旧hash失效
                result_msg = f"{EDIT_FILE_MARK}\n✅ 修改成功\n文件：{file_path}\n理由：{reason}"
            else:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                create_version_snapshot(full_path, workspace)
                result_msg = f"{EDIT_FILE_MARK}\n✅ 文件不存在，已自动创建并写入内容\n文件：{file_path}\n理由：{reason}"

            rag_manager = get_rag_manager()
            if rag_manager:
                rag_manager.update_file_index(file_path)
            return result_msg
        except Exception as e:
            logger.exception(f"修改失败：{str(e)}")
            return f"❌ 修改失败：{str(e)}"

    # ------------------------------------------------------------------
    # edit_file_batch —— 基于 HASH 的精确批量修改（推荐优先使用）
    # ------------------------------------------------------------------
    # 流程：从 _hash_index_cache 获取目标文件的 HASH INDEX
    #      → validate_batch_changes 校验所有 hash 有效性
    #      → apply_batch_changes 按 offset 从后往前应用（防止偏移干扰）
    #      → 写入磁盘 + 更新 RAG + 清除缓存
    #
    # 三种 mode：
    #   replace:       用 new_content 替换 hash 对应的代码块
    #   insert_before: 在 hash 对应的代码块前面插入 new_content
    #   insert_after:  在 hash 对应的代码块后面插入 new_content
    #   new_content="" 且 mode=replace → 删除该代码块
    @tool(args_schema=EditFileBatchInput)
    def edit_file_batch(**kwargs):
        """【推荐】基于HASH INDEX批量修改文件，支持替换/前插/后插"""
        try:
            file_path = kwargs["file_path"]
            changes = kwargs["changes"]
            reason = kwargs["reason"]
            print(f"【工具调用】edit_file_batch | {file_path} | {reason} | {len(changes)}项")
            logger.info(f"【工具调用】edit_file_batch | {file_path} | {reason} | {len(changes)}项")

            safe, msg = check_file_safety(file_path, workspace)
            if not safe:
                return f"❌ 安全校验失败：{msg}"

            # 从缓存中获取 HASH INDEX
            hash_index = _hash_index_cache.get(file_path)
            if not hash_index:
                return f"❌ 未找到 {file_path} 的hash索引，请先使用 read_file 读取该文件"

            changes_dicts = [c.model_dump() if hasattr(c, 'model_dump') else c for c in changes]

            # 校验所有 change 的 hash 是否有效
            valid, err_msg = validate_batch_changes(changes_dicts, hash_index)
            if not valid:
                return f"❌ 批量修改校验失败：{err_msg}"

            full_path = os.path.join(workspace, file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                current_content = f.read()

            # 应用修改（offset 从后往前，防止前面的修改影响后面的定位）
            new_content, apply_err = apply_batch_changes(current_content, changes_dicts, hash_index)
            if apply_err:
                return f"❌ 批量修改应用失败：{apply_err}"

            create_version_snapshot(full_path, workspace)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            _hash_index_cache.pop(file_path, None)  # 消费后清除缓存

            rag_manager = get_rag_manager()
            if rag_manager:
                rag_manager.update_file_index(file_path)

            # 构建简要的修改说明
            detail_lines = []
            for c in changes_dicts:
                block = resolve_block_by_hash(hash_index, c["hash"])
                name = block["name"] if block else c["hash"]
                mode = c.get("mode", "replace")
                detail_lines.append(f"  - [{mode}] {name}")

            return f"{EDIT_FILE_MARK}\n✅ 批量修改成功\n文件：{file_path}\n理由：{reason}\n修改项：\n" + "\n".join(detail_lines)
        except Exception as e:
            logger.exception(f"批量修改失败：{str(e)}")
            return f"❌ 批量修改失败：{str(e)}"

    # ------------------------------------------------------------------
    # rename_file —— 重命名文件
    # ------------------------------------------------------------------
    # 自动处理重名（追加 _1 后缀）。
    # 旧路径的索引删除，新路径重新索引。
    @tool(args_schema=RenameFileInput)
    def rename_file(**kwargs):
        """重命名文件，自动处理重名，生成副本"""
        try:
            old_file_path = kwargs["old_file_path"]
            new_file_path = kwargs["new_file_path"]
            reason = kwargs["reason"]
            print(f"【工具调用】rename_file | {old_file_path} → {new_file_path}")
            logger.info(f"【工具调用】rename_file | {old_file_path} → {new_file_path}")

            safe1, msg1 = check_file_safety(old_file_path, workspace)
            safe2, msg2 = check_file_safety(new_file_path, workspace)
            if not safe1 or not safe2:
                return f"❌ 校验失败：{msg1 if not safe1 else msg2}"

            old_full = os.path.join(workspace, old_file_path)
            new_full = os.path.join(workspace, new_file_path)
            if not os.path.exists(old_full):
                return "❌ 原文件不存在"

            final_new_full, tip = get_unique_file_path(new_full)
            final_new_path = os.path.relpath(final_new_full, workspace)

            create_version_snapshot(old_full, workspace)
            os.rename(old_full, final_new_full)
            _hash_index_cache.pop(old_file_path, None)

            rag_manager = get_rag_manager()
            if rag_manager:
                rag_manager.remove_file_from_index(old_file_path)
                rag_manager.update_file_index(final_new_path)

            return f"{RENAME_FILE_MARK}\n✅ 重命名成功\n{old_file_path} → {final_new_path}\n提示：{tip}\n理由：{reason}"
        except Exception as e:
            logger.exception(f"重命名失败：{str(e)}")
            return f"❌ 重命名失败：{str(e)}"

    # ------------------------------------------------------------------
    # copy_file / move_file —— 复制和移动文件
    # ------------------------------------------------------------------
    # 逻辑类似：校验 → 防重名 → 操作 → RAG索引同步

    @tool(args_schema=CopyFileInput)
    def copy_file(**kwargs):
        """复制文件，自动处理重名"""
        try:
            source_file = kwargs["source_file"]
            target_file = kwargs["target_file"]
            reason = kwargs["reason"]
            safe_s, msg_s = check_file_safety(source_file, workspace)
            safe_t, msg_t = check_file_safety(target_file, workspace)
            if not safe_s or not safe_t:
                return f"❌ 校验失败：{msg_s if not safe_s else msg_t}"
            s_full = os.path.join(workspace, source_file)
            t_full = os.path.join(workspace, target_file)
            if not os.path.exists(s_full):
                return "❌ 源文件不存在"
            final_t_full, tip = get_unique_file_path(t_full)
            final_t_path = os.path.relpath(final_t_full, workspace)
            shutil.copy2(s_full, final_t_full)  # copy2 保留文件元数据
            create_version_snapshot(final_t_full, workspace)
            rag_manager = get_rag_manager()
            if rag_manager:
                rag_manager.update_file_index(final_t_path)
            return f"{COPY_FILE_MARK}\n✅ 复制成功\n{source_file} → {final_t_path}\n提示：{tip}\n理由：{reason}"
        except Exception as e:
            return f"❌ 复制失败：{str(e)}"

    @tool(args_schema=MoveFileInput)
    def move_file(**kwargs):
        """移动/剪切文件"""
        try:
            source_file = kwargs["source_file"]
            target_file = kwargs["target_file"]
            reason = kwargs["reason"]
            safe_s, msg_s = check_file_safety(source_file, workspace)
            safe_t, msg_t = check_file_safety(target_file, workspace)
            if not safe_s or not safe_t:
                return f"❌ 校验失败：{msg_s if not safe_s else msg_t}"
            s_full = os.path.join(workspace, source_file)
            t_full = os.path.join(workspace, target_file)
            if not os.path.exists(s_full):
                return "❌ 源文件不存在"
            final_t_full, tip = get_unique_file_path(t_full)
            final_t_path = os.path.relpath(final_t_full, workspace)
            create_version_snapshot(s_full, workspace)
            os.rename(s_full, final_t_full)
            _hash_index_cache.pop(source_file, None)
            rag_manager = get_rag_manager()
            if rag_manager:
                rag_manager.remove_file_from_index(source_file)
                rag_manager.update_file_index(final_t_path)
            return f"{MOVE_FILE_MARK}\n✅ 移动成功\n{source_file} → {final_t_path}\n提示：{tip}\n理由：{reason}"
        except Exception as e:
            return f"❌ 移动失败：{str(e)}"

    # ------------------------------------------------------------------
    # create_file —— 创建空白文件
    # ------------------------------------------------------------------
    @tool(args_schema=CreateFileInput)
    def create_file(**kwargs):
        """创建空白代码文件"""
        try:
            file_path = kwargs["file_path"]
            reason = kwargs["reason"]
            safe, msg = check_file_safety(file_path, workspace)
            if not safe:
                return f"❌ 安全校验失败：{msg}"
            full_path = os.path.join(workspace, file_path)
            if os.path.exists(full_path):
                return "❌ 文件已存在"
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write("")
            create_version_snapshot(full_path, workspace)
            rag_manager = get_rag_manager()
            if rag_manager:
                rag_manager.update_file_index(file_path)
            return f"{CREATE_FILE_MARK}\n✅ 创建成功\n文件：{file_path}\n理由：{reason}"
        except Exception as e:
            return f"❌ 创建失败：{str(e)}"

    # ------------------------------------------------------------------
    # list_files —— 列出工作区所有代码文件
    # ------------------------------------------------------------------
    @tool
    def list_files():
        """列出工作区内所有代码文件（递归）"""
        try:
            print("【工具调用】list_files")
            from utils.path_validator import iter_workspace_files
            file_list = sorted(rel for rel, _ in iter_workspace_files(workspace))
            return f"{LIST_FILE_MARK}\n✅ 工作区文件列表：\n" + "\n".join(file_list)
        except Exception as e:
            return f"❌ 列出文件失败：{str(e)}"

    # ------------------------------------------------------------------
    # delete_file —— 删除文件（带系统确认弹窗）
    # ------------------------------------------------------------------
    # 弹窗在主线程中通过 Tkinter askyesno 阻塞，等待用户确认。
    @tool(args_schema=DeleteFileInput)
    def delete_file(**kwargs):
        """删除文件，本地自动弹出确认窗口"""
        try:
            file_path = kwargs["file_path"]
            safe, msg = check_file_safety(file_path, workspace)
            if not safe:
                return f"❌ 安全校验失败：{msg}"
            full_path = os.path.join(workspace, file_path)
            if not os.path.exists(full_path):
                return "❌ 文件不存在"

            confirm_result = messagebox.askyesno(
                title="删除确认",
                message=f"确定要删除文件吗？\n\n{file_path}",
                parent=app_root
            )
            if not confirm_result:
                return f"用户取消了删除操作：{file_path}"

            create_version_snapshot(full_path, workspace)
            os.remove(full_path)
            _hash_index_cache.pop(file_path, None)
            rag_manager = get_rag_manager()
            if rag_manager:
                rag_manager.remove_file_from_index(file_path)
            return f"{DELETE_FILE_MARK}\n✅ 删除成功\n文件：{file_path}"
        except Exception as e:
            return f"❌ 删除失败：{str(e)}"

    # ------------------------------------------------------------------
    # create_folder —— 创建文件夹
    # ------------------------------------------------------------------
    @tool(args_schema=CreateFolderInput)
    def create_folder(**kwargs):
        """创建文件夹（支持多级递归）"""
        try:
            folder_path = kwargs["folder_path"]
            reason = kwargs["reason"]
            safe, msg = check_folder_safety(folder_path, workspace)
            if not safe:
                return f"❌ 安全校验失败：{msg}"
            full_path = os.path.join(workspace, folder_path)
            if os.path.exists(full_path):
                return "❌ 文件夹已存在"
            os.makedirs(full_path, exist_ok=True)
            return f"{CREATE_FOLDER_MARK}\n✅ 创建成功\n文件夹：{folder_path}\n理由：{reason}"
        except Exception as e:
            return f"❌ 创建文件夹失败：{str(e)}"

    # ------------------------------------------------------------------
    # delete_folder —— 删除文件夹（递归 + 确认弹窗）
    # ------------------------------------------------------------------
    # 删除前先弹确认窗。删除后遍历索引中所有受影响的文件，逐一从RAG索引中移除。
    @tool(args_schema=DeleteFolderInput)
    def delete_folder(**kwargs):
        """删除文件夹（递归删除所有内容，带确认弹窗）"""
        try:
            folder_path = kwargs["folder_path"]
            safe, msg = check_folder_safety(folder_path, workspace)
            if not safe:
                return f"❌ 安全校验失败：{msg}"
            full_path = os.path.join(workspace, folder_path)
            if not os.path.exists(full_path):
                return "❌ 文件夹不存在"

            confirm_result = messagebox.askyesno(
                title="删除确认",
                message=f"确定要删除文件夹及其所有内容吗？\n\n{folder_path}",
                parent=app_root
            )
            if not confirm_result:
                return f"用户取消了删除操作：{folder_path}"

            # 删除前扫描子文件，同步清理RAG索引
            rag_manager = get_rag_manager()
            if rag_manager:
                for root, dirs, files in os.walk(full_path):
                    for f in files:
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, workspace)
                        rag_manager.remove_file_from_index(rel)

            shutil.rmtree(full_path)
            return f"{DELETE_FOLDER_MARK}\n✅ 删除成功\n文件夹：{folder_path}"
        except Exception as e:
            return f"❌ 删除文件夹失败：{str(e)}"

    # ------------------------------------------------------------------
    # rename_folder / move_folder / copy_folder —— 文件夹操作
    # ------------------------------------------------------------------
    # 操作逻辑：遍历子文件 → 逐个更新RAG索引 → 执行目录级操作

    @tool(args_schema=RenameFolderInput)
    def rename_folder(**kwargs):
        """重命名文件夹"""
        try:
            old_folder_path = kwargs["old_folder_path"]
            new_folder_path = kwargs["new_folder_path"]
            reason = kwargs["reason"]
            safe1, msg1 = check_folder_safety(old_folder_path, workspace)
            safe2, msg2 = check_folder_safety(new_folder_path, workspace)
            if not safe1 or not safe2:
                return f"❌ 校验失败：{msg1 if not safe1 else msg2}"
            old_full = os.path.join(workspace, old_folder_path)
            new_full = os.path.join(workspace, new_folder_path)
            if not os.path.exists(old_full):
                return "❌ 原文件夹不存在"

            # 遍历旧文件夹下所有文件，从RAG索引中删除
            rag_manager = get_rag_manager()
            old_files = []
            if rag_manager:
                for root, dirs, files in os.walk(old_full):
                    for f in files:
                        old_files.append(os.path.relpath(os.path.join(root, f), workspace))
                        rag_manager.remove_file_from_index(old_files[-1])

            os.rename(old_full, new_full)

            # 重新索引新路径下的所有文件
            if rag_manager:
                for root, dirs, files in os.walk(new_full):
                    for f in files:
                        new_rel = os.path.relpath(os.path.join(root, f), workspace)
                        rag_manager.update_file_index(new_rel)

            return f"{RENAME_FOLDER_MARK}\n✅ 重命名成功\n{old_folder_path} → {new_folder_path}\n理由：{reason}"
        except Exception as e:
            return f"❌ 重命名文件夹失败：{str(e)}"

    @tool(args_schema=MoveFolderInput)
    def move_folder(**kwargs):
        """移动文件夹"""
        try:
            source_folder = kwargs["source_folder"]
            target_folder = kwargs["target_folder"]
            reason = kwargs["reason"]
            safe_s, msg_s = check_folder_safety(source_folder, workspace)
            safe_t, msg_t = check_folder_safety(target_folder, workspace)
            if not safe_s or not safe_t:
                return f"❌ 校验失败"
            s_full = os.path.join(workspace, source_folder)
            t_full = os.path.join(workspace, target_folder)
            if not os.path.exists(s_full):
                return "❌ 源文件夹不存在"
            if os.path.exists(t_full):
                return "❌ 目标已存在"

            rag_manager = get_rag_manager()
            if rag_manager:
                for root, dirs, files in os.walk(s_full):
                    for f in files:
                        rag_manager.remove_file_from_index(os.path.relpath(os.path.join(root, f), workspace))

            shutil.move(s_full, t_full)

            if rag_manager:
                for root, dirs, files in os.walk(t_full):
                    for f in files:
                        rag_manager.update_file_index(os.path.relpath(os.path.join(root, f), workspace))

            return f"{MOVE_FOLDER_MARK}\n✅ 移动成功\n{source_folder} → {target_folder}\n理由：{reason}"
        except Exception as e:
            return f"❌ 移动文件夹失败：{str(e)}"

    @tool(args_schema=CopyFolderInput)
    def copy_folder(**kwargs):
        """复制文件夹（递归）"""
        try:
            source_folder = kwargs["source_folder"]
            target_folder = kwargs["target_folder"]
            reason = kwargs["reason"]
            safe_s, msg_s = check_folder_safety(source_folder, workspace)
            safe_t, msg_t = check_folder_safety(target_folder, workspace)
            if not safe_s or not safe_t:
                return "❌ 校验失败"

            s_full = os.path.join(workspace, source_folder)
            t_full = os.path.join(workspace, target_folder)
            if not os.path.exists(s_full):
                return "❌ 源文件夹不存在"
            if os.path.exists(t_full):
                return "❌ 目标已存在"

            shutil.copytree(s_full, t_full)

            rag_manager = get_rag_manager()
            if rag_manager:
                for root, dirs, files in os.walk(t_full):
                    for f in files:
                        rag_manager.update_file_index(os.path.relpath(os.path.join(root, f), workspace))

            return f"{COPY_FOLDER_MARK}\n✅ 复制成功\n{source_folder} → {target_folder}\n理由：{reason}"
        except Exception as e:
            return f"❌ 复制文件夹失败：{str(e)}"

    # 返回所有工具（Agent 可调用的能力集合）
    return [
        read_file, edit_file, edit_file_batch, rename_file,
        copy_file, move_file, create_file,
        list_files, delete_file,
        create_folder, delete_folder, rename_folder, move_folder, copy_folder
    ]
