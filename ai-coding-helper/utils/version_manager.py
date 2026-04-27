"""
================================================================================
版本快照管理 —— 文件修改前的自动备份与回退机制
================================================================================

【设计目标】
每次 Agent 修改文件前自动创建快照，用户可随时回退到之前版本。
想象一下 Git 的 commit history，但完全自动化，无需用户手动操作。

【快照存储】
.workspace_versions/                  ← 隐藏目录（.gitignore 自动排除）
  ├── core_agent.py.1700000000000.snap  ← 时间戳格式的快照文件
  ├── core_agent.py.1700000001000.snap
  └── ...

快照文件名格式：{文件key}.{时间戳ms}.snap
文件key = 相对路径用 _ 替换路径分隔符（如 core/agent.py → core_agent.py）

【并发安全】
所有写操作通过 threading.Lock() 保护，防止 Agent 多线程操作时快照冲突。

【清理策略】
每个文件最多保留 MAX_HISTORY_VERSIONS（默认2）个快照。
创建新快照时自动删除超出数量的旧快照。
"""
import os, time, threading, shutil, logging
from typing import Optional, List
from ui.config import MAX_HISTORY_VERSIONS

logger = logging.getLogger(__name__)

version_lock = threading.Lock()              # 全局互斥锁，保护快照文件并发安全
VERSION_DIR_NAME = ".workspace_versions"     # 快照存储目录名


def get_version_dir(workspace: str) -> str:
    """获取工作区的版本快照目录路径"""
    return os.path.join(workspace, VERSION_DIR_NAME)


def _init_version_dir_internal(version_dir: str) -> None:
    """内部初始化：创建版本目录 + .gitignore（不带锁，由调用方持有锁）"""
    os.makedirs(version_dir, exist_ok=True)
    # 确保版本目录不会被 Git 跟踪
    gitignore_path = os.path.join(version_dir, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("*")   # 忽略目录下所有文件


def init_version_dir(workspace: str) -> None:
    """外部接口：初始化版本目录（带锁）"""
    version_dir = get_version_dir(workspace)
    with version_lock:
        _init_version_dir_internal(version_dir)


def get_file_version_key(file_path: str, workspace: str) -> str:
    """
    生成文件的版本存储key。
    将文件相对路径中的路径分隔符替换为 _，避免子目录创建问题。
    例: core/agent.py → core_agent.py
    """
    rel_path = os.path.relpath(os.path.abspath(file_path), os.path.abspath(workspace))
    return rel_path.replace(os.sep, "_").replace("/", "_").replace("\\", "_")


# ==============================================================================
# create_version_snapshot —— 创建文件快照
# ==============================================================================
# 这是最重要接口：每次 Agent 写入文件前由 file_tool.py 调用。
# 流程：加锁 → 初始化目录 → 复制文件 → 清理旧快照 → 解锁
def create_version_snapshot(file_path: str, workspace: str) -> bool:
    """
    创建文件的版本快照。
    返回 True/False 表示是否成功（失败不影响 Agent 继续执行）。
    """
    if not os.path.exists(file_path):
        logger.warning(f"文件[{file_path}]不存在，跳过版本快照")
        return False
    try:
        with version_lock:
            version_dir = get_version_dir(workspace)
            _init_version_dir_internal(version_dir)       # 确保目录存在

            file_key = get_file_version_key(file_path, workspace)
            timestamp = str(int(time.time() * 1000))       # 毫秒级时间戳
            snapshot_path = os.path.join(version_dir, f"{file_key}.{timestamp}.snap")

            # copy2 保留文件元数据（修改时间等），完整复制文件内容
            shutil.copy2(file_path, snapshot_path)
            logger.info(f"文件[{file_path}]版本快照创建成功")

            # 清理超出数量限制的旧快照
            _clean_expired_versions_internal(version_dir, file_key)
            return True
    except Exception as e:
        logger.error(f"创建版本快照失败：{str(e)}")
        return False


def _clean_expired_versions_internal(version_dir: str, file_key: str) -> None:
    """
    清理过期快照：保留最近的 MAX_HISTORY_VERSIONS 个，删除其余。
    按时间戳降序排列后，截断尾部。
    """
    try:
        snap_files = [f for f in os.listdir(version_dir) if f.startswith(file_key) and f.endswith(".snap")]
        snap_files.sort(key=lambda x: int(x.split(".")[-2]), reverse=True)  # 最新在前
        if len(snap_files) > MAX_HISTORY_VERSIONS:
            for snap in snap_files[MAX_HISTORY_VERSIONS:]:
                os.remove(os.path.join(version_dir, snap))
                logger.info(f"清理过期版本快照：{snap}")
    except Exception as e:
        logger.error(f"清理过期版本失败：{str(e)}")


# ==============================================================================
# get_file_history_versions —— 获取文件历史版本列表
# ==============================================================================
def get_file_history_versions(file_path: str, workspace: str) -> List[dict]:
    """返回文件的历史快照列表（按时间倒序）"""
    history = []
    try:
        version_dir = get_version_dir(workspace)
        if not os.path.exists(version_dir):
            return history

        file_key = get_file_version_key(file_path, workspace)
        snap_files = [f for f in os.listdir(version_dir) if f.startswith(file_key) and f.endswith(".snap")]
        snap_files.sort(key=lambda x: int(x.split(".")[-2]), reverse=True)

        for snap in snap_files:
            timestamp = int(snap.split(".")[-2])
            create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp / 1000))
            history.append({
                "snap_file": snap,
                "snap_path": os.path.join(version_dir, snap),
                "create_time": create_time,
                "timestamp": timestamp
            })
        return history
    except Exception as e:
        logger.error(f"获取历史版本失败：{str(e)}")
        return []


# ==============================================================================
# rollback_to_version —— 回退文件到指定快照
# ==============================================================================
# 回退前会自动创建"回退前快照"（类似 Git 的 reflog），防止回退操作不可逆。
def rollback_to_version(file_path: str, workspace: str, snap_path: Optional[str] = None) -> tuple[bool, str]:
    """回退文件到指定版本"""
    try:
        with version_lock:
            if not snap_path:
                history = get_file_history_versions(file_path, workspace)
                if not history:
                    return False, "该文件无可用的历史版本"
                snap_path = history[0]["snap_path"]   # 默认回退到最新快照

            if not os.path.exists(snap_path):
                return False, "快照文件不存在，回退失败"

            # 回退前先给当前状态做个快照（防后悔）
            if os.path.exists(file_path):
                version_dir = get_version_dir(workspace)
                _init_version_dir_internal(version_dir)
                file_key = get_file_version_key(file_path, workspace)
                timestamp = str(int(time.time() * 1000))
                backup_snap = os.path.join(version_dir, f"{file_key}.{timestamp}.snap")
                shutil.copy2(file_path, backup_snap)

            # 执行回退：用快照覆盖当前文件
            shutil.copy2(snap_path, file_path)
            logger.info(f"文件[{file_path}]回退成功")
            return True, "回退成功"
    except Exception as e:
        return False, f"回退失败：{str(e)}"


def rollback_last_modify(file_path: str, workspace: str) -> tuple[bool, str]:
    """快捷回退到上一个修改版本"""
    return rollback_to_version(file_path, workspace, None)


# ==============================================================================
# clean_workspace_versions —— 退出时清理所有历史版本
# ==============================================================================
def clean_workspace_versions(workspace: str) -> None:
    """
    程序退出时调用：删除整个版本目录，释放磁盘空间。
    仅清理当前工作区的版本目录。
    """
    if not workspace or not os.path.exists(workspace):
        return
    version_dir = get_version_dir(workspace)
    if os.path.exists(version_dir):
        shutil.rmtree(version_dir, ignore_errors=True)
        logger.info(f"已清理工作区历史版本：{version_dir}")
