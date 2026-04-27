"""
================================================================================
Agent 编排层 —— 整个 AI 编程助手的中枢神经系统
================================================================================

【架构定位】
本模块是整个项目的"大脑"，负责：

1. LLM 实例化与配置（温度、超时、重试等）
2. 上下文管理（智能修剪，防止 token 超限）
3. 工具注册（文件工具 + RAG 检索工具）
4. Agent 创建（基于 LangChain create_agent，使用 ReAct 模式）
5. 会话记忆管理（LangGraph MemorySaver 检查点机制）
6. System Prompt 注入（工具说明 + 行为规则 + 项目上下文）

【核心概念 —— LangChain Agent 原理】
- LangChain Agent：LLM 通过"思考→调用工具→观察结果→继续思考"的循环来完成任务
  这种模式叫做 ReAct (Reasoning + Acting)
- LangGraph：在 LangChain 之上提供状态图管理，MemorySaver 是它的检查点机制
- Checkpoint：每次 LLM 调用和工具调用后自动保存消息快照，支持断点续传
- Context Window：LLM 能"记住"的最大 token 数。DeepSeek 是 128K tokens
  但文件内容（ToolMessage）可能占几万行，必须修剪，否则很快就超出窗口

【ChatOpenAI — 为什么 DeepSeek 能用 OpenAI 的客户端】
DeepSeek、通义千问等都实现了 OpenAI 兼容的 API 格式，包括：
- /v1/chat/completions 端点（Chat Completions API）
- 相同的请求体格式：{"model":"...", "messages":[...], "temperature":...}
- 相同的响应格式：{"choices":[{"message":{"content":"..."}}]}
所以用 ChatOpenAI(base_url="https://api.deepseek.com") 就能直接调用 DeepSeek。

【@tool 装饰器原理】
@tool 是 LangChain 的核心装饰器，将普通 Python 函数转换为 LLM 可调用的 Tool：
1. 提取函数签名（参数名+类型注解）→ 生成 Tool Schema (JSON Schema)
2. args_schema=PydanticModel 提供更丰富的参数描述（description + examples）
3. 函数 docstring 作为工具描述（LLM 据此决定何时调用此工具）
4. 生成的 Tool 包含 .name / .description / .args_schema / .invoke() 等属性
LLM 看到工具列表 → 输出 tool_calls JSON → LangChain 解析 → 调用 .invoke()

【面试高频题 —— Agent/LLM 调用链】
Q5: ChatOpenAI 的 temperature 参数是什么？设多少合适？
A:  temperature 控制输出的随机性，范围 0~2。越低越确定，越高越有"创意"。
    代码任务设为 0.1（需要精确输出），写文章可设 0.7~0.9。
    本项目用 0.1 + top_p=0.85 确保代码修改准确可控。

Q6: 流式输出(stream)和非流式(invoke)有什么区别？
A:  invoke: 等 LLM 生成全部内容后一次性返回（等待时间长，用户体验差）
    stream: 每生成一个 token 就 yield，UI 实时更新（打字机效果，本项目使用）
    底层都是 SSE (Server-Sent Events) 协议。

Q7: Pydantic 在 LangChain 中扮演什么角色？
A:  所有工具的参数校验都通过 Pydantic BaseModel 实现：
    - Field(description="...") 定义参数描述（给 LLM 看）
    - 类型注解（str/int/float）自动校验输入类型
    - .model_dump() 转为 dict 传给工具函数
    本项目每个工具都有对应的 Input 类（如 EditFileInput, ReadFileInput 等）。
"""
from langchain.agents import create_agent          # LangChain 的 Agent 工厂函数
from langchain_openai import ChatOpenAI             # OpenAI 兼容的 LLM 客户端（DeepSeek 也用这个）
from core.file_tool import get_tools                # 文件操作工具集（read/edit/create/delete...）
from langgraph.checkpoint.memory import MemorySaver  # LangGraph 内存检查点（保存会话历史）
from langchain_core.runnables import RunnableConfig  # Agent 运行配置（thread_id, recursion_limit）
from langchain_core.messages import (
    SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
)
from typing_extensions import TypedDict
from typing import Annotated, Sequence, List, Optional, Any
import operator
import logging
import tkinter as tk
from core.rag.rag_tool import get_rag_tools
from ui.config import AUTO_TRIM_CONTEXT

logger = logging.getLogger(__name__)


# ==============================================================================
# AgentState —— LangGraph 状态定义
# ==============================================================================
# LangGraph 用 TypedDict 定义 Agent 的状态结构。
# messages 字段用 operator.add 做 Annotated，意思是：新的消息会"追加"到已有列表，
# 而不是覆盖。这样每一轮对话都会累加历史消息。
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ==============================================================================
# smart_context_trimmer —— 智能上下文修剪器
# ==============================================================================
# 【为什么需要修剪】
# LLM 的上下文窗口有限（DeepSeek 128K tokens），但文件内容可能占几万 tokens。
# 如果不修剪，几轮对话后就会超过窗口限制，导致：
#   1. 最早的消息被截断（模型"失忆"）
#   2. API 调用失败（token 超限）
#   3. 响应变慢（处理大量无用内容）
#
# 【修剪策略 —— "摘要化"而非"丢弃"】
# 核心思想：历史轮次中，完全删除文件内容（ToolMessage），但用操作摘要代替。
# 这样模型知道"我做过什么"，但不需要记住"文件里具体是什么"。
#
# 轮次划分规则：每个 HumanMessage 开始新的一轮
def smart_context_trimmer(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """
    智能上下文修剪器：压缩历史轮次的消息，保留最新轮的完整内容。

    三条铁律：
    1. SystemMessage 永远是第一条，且只保留一个
    2. 历史轮次：删除 ToolMessage（文件内容），替换为操作摘要 + 保留 AI 总结
    3. 最新轮次：完整保留所有消息（包括 ToolMessage）

    修剪前后对比示例：
    ─────────────────────────────────────────────
    修剪前（3轮对话）:
      [System: "你是代码助手..."]
      [Human: "优化登录"]                          ← 第1轮
      [AI(tool_calls): read_file(login.py)]       ← 第1轮工具调用
      [ToolMessage: "login.py 完整3000行内容..."]  ← 第1轮文件内容（几万tokens）
      [AI: "已优化登录，添加了验证码逻辑"]          ← 第1轮总结
      [Human: "加个注册功能"]                       ← 第2轮
      [AI(tool_calls): create_file(register.py)]  ← 第2轮工具调用
      [ToolMessage: "创建成功"]                     ← 第2轮工具结果
      [AI: "注册功能已添加"]                        ← 第2轮总结
      [Human: "检查bug"]                           ← 第3轮（最新）

    修剪后:
      [System: "你是代码助手..."]
      [Human: "优化登录"]                          ← 第1轮用户问题保留
      [AI: "已执行：read_file(login.py) | 优化登录验证"]
      [AI: "已优化登录，添加了验证码逻辑"]          ← 第1轮总结保留
      [Human: "加个注册功能"]                       ← 第2轮用户问题保留
      [AI: "已执行：create_file(register.py) | 新增注册模块"]
      [AI: "注册功能已添加"]                        ← 第2轮总结保留
      [Human: "检查bug"]                           ← 第3轮完整保留
      （如果有 ToolMessage 也完整保留）
    ─────────────────────────────────────────────
    """
    if not AUTO_TRIM_CONTEXT:
        return list(messages)   # 开关关闭时不做任何修剪

    trimmed_messages: List[BaseMessage] = []

    # ---- 步骤1: 提取 SystemMessage ----
    # System prompt 定义了 Agent 的行为规则，必须保留且只在第一位。
    # 只取第一个（正常情况下只有一个），多余的忽略。
    system_message: Optional[SystemMessage] = None
    for msg in messages:
        if isinstance(msg, SystemMessage):
            if system_message is None:
                system_message = msg
            break   # 找到第一个就停止
    if system_message:
        trimmed_messages.append(system_message)

    # ---- 步骤2: 按轮次分组 ----
    # LangChain 的消息流中，每个 HumanMessage 标志着一轮对话的开始。
    # 我们把消息按 HumanMessage 为分隔符切分成 rounds。
    rounds: List[List[BaseMessage]] = []
    current_round: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue   # SystemMessage 已在上面处理
        if isinstance(msg, HumanMessage):
            if current_round:
                rounds.append(current_round)
            current_round = [msg]   # 新轮次从 HumanMessage 开始
        else:
            current_round.append(msg)
    if current_round:
        rounds.append(current_round)

    # ---- 步骤3: 处理历史轮次 + 保留最新轮 ----
    if len(rounds) > 0:
        history_rounds = rounds[:-1]     # 除最后一轮外都是历史
        latest_round = rounds[-1]        # 最后一轮完整保留

        for round_msgs in history_rounds:
            user_msg: Optional[HumanMessage] = None
            operations: List[str] = []        # 收集工具调用信息
            last_ai_text: Optional[str] = None # 只保留最后一个 AI 文本（总结）

            for msg in round_msgs:
                if isinstance(msg, HumanMessage):
                    user_msg = msg   # 用户原始问题保留
                elif isinstance(msg, AIMessage):
                    # 提取工具调用信息（含文件名和理由）
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            parts = [tool_call.get("name", "")]
                            tool_args = tool_call.get("args", {})
                            # 兼容所有工具的文件路径参数名
                            file_path = (
                                tool_args.get("file_path") or tool_args.get("source_file")
                                or tool_args.get("old_file_path") or tool_args.get("target_file")
                                or tool_args.get("new_file_path") or tool_args.get("source_folder")
                                or tool_args.get("folder_path")
                            )
                            if file_path:
                                parts.append(f"({file_path})")
                            reason = tool_args.get("reason", "")
                            if reason:
                                parts.append(f" | {reason}")  # 保留操作理由
                            operations.append("".join(parts))
                    # 收集 AI 的文本回复（后面的覆盖前面的，保留最后一个=总结）
                    if msg.content and msg.content.strip():
                        last_ai_text = msg.content.strip()

            if user_msg:
                trimmed_messages.append(user_msg)
            if operations:
                ops_str = " → ".join(operations)
                trimmed_messages.append(AIMessage(content=f"已执行：{ops_str}"))
            if last_ai_text:
                truncated = last_ai_text[:400] + ("..." if len(last_ai_text) > 400 else "")
                trimmed_messages.append(AIMessage(content=truncated))

        # 最新一轮：完整保留，不修剪
        trimmed_messages.extend(latest_round)

    logger.info(
        f"上下文修剪：原始{len(messages)}条 → 修剪后{len(trimmed_messages)}条"
    )
    return trimmed_messages


# ==============================================================================
# TrimmedChatOpenAI —— 带上下文修剪的 LLM 客户端
# ==============================================================================
# 【设计模式：装饰器/代理模式】
# 继承 ChatOpenAI，在每次 LLM 调用前自动执行 smart_context_trimmer。
# 对上层代码完全透明 —— handlers.py 只看到"我调用了 LLM"，不需要关心修剪逻辑。
#
# 重写了 4 个方法（同步+异步 × 调用+流式）：
#   invoke()  → 同步，返回完整响应
#   stream()  → 同步，逐 token 流式输出（本项目主要使用这种方式）
#   ainvoke() → 异步版本（当前未使用，保留给未来扩展）
#   astream() → 异步流式（当前未使用，保留给未来扩展）
#
# 【流式输出 vs 非流式】
# 流式(stream)：每生成一个 token 就 yield，UI 可以实时显示（打字机效果）
# 非流式(invoke)：等待全部生成完毕再返回，适合不需要实时反馈的场景
class TrimmedChatOpenAI(ChatOpenAI):
    """ChatOpenAI 的子类，在每次 LLM 调用前自动执行上下文修剪。"""

    def invoke(self, input_messages, config=None, **kwargs):
        """同步调用：修剪 → 调用父类 invoke → 返回"""
        if isinstance(input_messages, dict) and "messages" in input_messages:
            input_messages["messages"] = smart_context_trimmer(input_messages["messages"])
        elif isinstance(input_messages, list):
            input_messages = smart_context_trimmer(input_messages)
        return super().invoke(input_messages, config=config, **kwargs)

    def stream(self, input_messages, config=None, **kwargs):
        """流式调用：修剪 → 调用父类 stream → 逐 chunk yield"""
        if isinstance(input_messages, dict) and "messages" in input_messages:
            input_messages["messages"] = smart_context_trimmer(input_messages["messages"])
        elif isinstance(input_messages, list):
            input_messages = smart_context_trimmer(input_messages)
        yield from super().stream(input_messages, config=config, **kwargs)

    async def ainvoke(self, input_messages, config=None, **kwargs):
        """异步调用（预留）"""
        if isinstance(input_messages, dict) and "messages" in input_messages:
            input_messages["messages"] = smart_context_trimmer(input_messages["messages"])
        elif isinstance(input_messages, list):
            input_messages = smart_context_trimmer(input_messages)
        return await super().ainvoke(input_messages, config=config, **kwargs)

    async def astream(self, input_messages, config=None, **kwargs):
        """异步流式（预留）"""
        if isinstance(input_messages, dict) and "messages" in input_messages:
            input_messages["messages"] = smart_context_trimmer(input_messages["messages"])
        elif isinstance(input_messages, list):
            input_messages = smart_context_trimmer(input_messages)
        async for chunk in super().astream(input_messages, config=config, **kwargs):
            yield chunk


# ==============================================================================
# create_agent_instance —— Agent 实例工厂
# ==============================================================================
# 【LangChain Agent 原理】
# LangChain 的 create_agent 内部使用 ReAct 循环：
#   1. LLM 分析用户输入 → 决定调用哪个工具
#   2. 执行工具 → 获取结果
#   3. 将结果反馈给 LLM → LLM 决定下一步（继续调工具 or 回复用户）
#   4. 循环直到 LLM 决定"任务完成"（不再调工具，直接回复文本）
#
# 【system_prompt 的作用】
# 这是发给 LLM 的第一条消息，定义了 Agent 的"人设"和行为约束。
# 内容包括：
#   - 可用工具列表及用法
#   - 执行流程规则（规划→分析→执行→总结）
#   - 工具调用规则（什么时候可以/不可以调哪个工具）
#   - 项目上下文（文件树、RAG 检索结果）
#   等
def create_agent_instance(model_type: str, workspace: str, model_config: dict, checkpointer, main_root: tk.Tk, file_tree: str = ""):
    """
    创建 Agent 实例。

    参数:
        model_type:   模型标识（如 "deepseek"）
        workspace:    工作区路径
        model_config: 模型配置字典 {api_key, base_url, model_name}
        checkpointer: MemorySaver 实例，负责保存/恢复会话状态
        main_root:    Tkinter 主窗口（用于弹窗确认）
        file_tree:    工作区文件树字符串，注入到 system_prompt 中

    返回:
        (agent_runnable, tools)：可运行的 Agent 对象 + 工具列表
    """
    logger.info(f"创建Agent实例，模型：{model_type}，工作区：{workspace}")

    # ========== LLM 超参数配置 ==========
    # temperature=0.1: 低温度让输出更确定、更可控（代码任务不需要创意）
    # top_p=0.85:     核采样，只考虑累计概率前85%的token
    # timeout=600:    API 调用超时 600 秒
    # max_tokens=None: 不限制输出长度（让模型自己决定）
    common_kwargs = {
        "temperature": 0.1,
        "top_p": 0.85,
        "timeout": 600,
        "max_retries": 3,
        "max_tokens": None,
        "presence_penalty": 0.1,
    }

    # ========== 创建 LLM 实例 ==========
    # 使用 TrimmedChatOpenAI（带上下文修剪）而非原生 ChatOpenAI
    # base_url 指向 DeepSeek 的 API 地址（兼容 OpenAI 接口格式）
    trimmed_llm = TrimmedChatOpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        model=model_config["model_name"],
        **common_kwargs
    )

    # ========== 注册工具 ==========
    # 工具是 Agent 的"手脚"，通过 @tool 装饰器注册为 LangChain 工具
    # file_tools: read_file, edit_file, edit_file_batch, create_file,
    #             rename_file, copy_file, move_file, delete_file, list_files,
    #             create_folder, delete_folder, rename_folder, move_folder, copy_folder
    file_tools = get_tools(workspace, main_root)
    # rag_tools: retrieve_related_files（基于向量检索找相关代码文件）
    rag_tools = get_rag_tools()
    tools = file_tools + rag_tools

    # ========== System Prompt（角色设定+行为规则）==========
    # 这是整个 Agent 行为的"宪法"，定义了所有规则
    system_prompt = f"""你是专业的代码助手智能体，严格遵循以下规则。
        ## 一、你的能力与可用工具
        你拥有以下工具来完成任务，工具都能够批量调用，必须根据场景选择正确的工具：
        ### 1. 文件读取类
        - **read_file(file_path)**
          - 用途：读取指定文件的最新完整内容（附带HASH INDEX，供edit_file_batch精确定位代码块）
          - 注意：每次只能读一个文件
        ### 2. 文件写入类
        - **edit_file_batch(file_path, changes, reason)**
          - 用途：基于HASH INDEX进行批量精确修改（**推荐优先使用**）
          - 注意：
            * 必须先调用 read_file 获取 HASH INDEX，再使用其中的 hash 值定位代码块
            * 每个 change 需指定 hash（来自 HASH INDEX）、new_content（新代码）、mode（replace/insert_before/insert_after）
            * mode=replace 替换代码块；insert_before 在块前插入；insert_after 在块后插入
            * new_content 传空字符串表示删除该代码块
            * 支持一次性批量修改多个代码块，系统自动按正确顺序应用
        - **edit_file(file_path, new_content, reason)**
          - 用途：全量覆盖修改文件内容（**大幅修改或新增文件时使用**）
          - 注意：
            * 必须传入完整的新内容，不能只传修改部分
            * 禁止写入空内容
            * 修改一个文件立即保存一个，不要等全部处理完
          - 自动机制：系统会自动创建版本快照，支持回退
        - **create_file(file_path, reason)**
          - 用途：创建新的空白代码文件
        ### 3. 文件操作类
        - **rename_file(old_file_path, new_file_path, reason)** ：重命名文件，自动处理重名
        - **copy_file(source_file, target_file, reason)** ：复制文件，自动处理重名
        - **move_file(source_file, target_file, reason)** ：移动/剪切文件
        - **delete_file(file_path)** ：删除文件（本地自动弹确认窗口）
        - **copy_folder(source_folder, target_folder, reason)** ：复制文件夹（递归）
        ### 4. 信息获取类
        - **list_files()** ：列出工作区内所有代码文件
        ### 5. 文件夹操作类
        - **create_folder / delete_folder / rename_folder / move_folder**
        ## 二、核心执行流程（必须严格遵守）
        0. **规划阶段（每轮首次必做）**：接到任务后，先回复一段简短的规划文字，逐项列出待完成的步骤清单。
        1. **分析阶段**：理解用户需求，确定需要操作的文件
        2. **检索阶段**：系统已自动为你检索了相关文件（见下方【检索结果】）
        3. **执行阶段**：严格按照「工具调用黄金规则」调用工具，一次性批量调用所有需要的工具
        4. **总结阶段**：对照规划中的清单，逐项确认完成状态，然后做简短总结
        ## 三、工具调用黄金规则（强制执行，无例外）
        1. **单独编辑现有文件 → 必须先调用 read_file**，确保使用文件最新版本
        2. **新建并写入文件（create+edit）→ 禁止调用 read_file**，直接连续执行
        3. **删除/重命名/复制/移动文件 → 禁止调用 read_file**，直接执行
        7. **所有 edit_file/edit_file_batch 操作 → 必须基于 read_file 的最新版本进行操作**
        8. **批量修改优先用 edit_file_batch**
        9. **批量调用**：在同一条消息中包含多个 tool_calls 会依次执行，无需分开发送
        10. **当工具返回【任务终止】标记时，立即结束任务**
        11. **文件夹操作 → 禁止调用任何文件工具**，专用文件夹工具操作
        ## 四、关键注意事项
        1. **路径要求**：所有文件路径必须是相对工作区的路径（如 core/agent.py）
        2. **安全限制**：禁止访问工作区外的文件，禁止操作禁止目录/文件类型
        3. **不要猜测**：不确定文件路径时，先调用 list_files 查看
        4. **代码质量**：保持与现有代码一致的风格
        5. **信任工具返回结果**：工具已做校验，不需要额外验证
        6. **任务完成即停止**：对照清单确认完成后直接总结，不要反复循环
        7. **闲聊模式**：如果系统提示「当前是聊天模式」，禁止调用任何工具，像普通聊天AI一样回复
        ## 四点五、Qt/C++ 项目规则（.pro 文件同步，强制执行）
        如果工作区中存在 `.pro` 文件（qmake 项目），创建/删除/重命名源文件后必须同步更新 .pro 的 SOURCES/HEADERS/FORMS/RESOURCES 列表。
        CMake 项目同理，需更新 CMakeLists.txt 的 add_executable/target_sources。
        ## 五、项目上下文
        【当前工作区文件树】
        {file_tree}
        ## 六、系统自动检索结果
        （系统会在此处插入自动检索到的相关文件列表）
        现在，开始执行任务。"""

    # ========== 创建 LangChain Agent ==========
    # create_agent 是 LangChain 的高层 API，内部封装了：
    #   1. ReAct 循环逻辑
    #   2. 工具绑定（bind_tools）
    #   3. 与 LangGraph 的集成（checkpointer）
    # debug=True 会在控制台输出详细的工具调用日志
    agent_runnable = create_agent(
        model=trimmed_llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        debug=True
    )
    return agent_runnable, tools


# ==============================================================================
# create_agent_executor —— 创建完整的 Agent 执行环境
# ==============================================================================
# 这是外部调用的入口函数（被 handlers.py 的 _set_workspace 调用）。
# 每切换一次工作区或模型，都会调用此函数创建新的 Agent 实例。
#
# 【MemorySaver 原理】
# MemorySaver 是 LangGraph 的内存级检查点存储。
# 每次 LLM 调用完成或工具执行完成，LangGraph 自动保存当前 messages 快照。
# 下次调用时，通过 thread_id 找到对应的历史状态，实现"会话记忆"。
# 缺点：进程重启后丢失（不持久化）。
#
# 【thread_id 的作用】
# 不同会话用不同的 thread_id 隔离：
#   会话A: thread_id = "abc-123"  → MemorySaver 存着 A 的对话历史
#   会话B: thread_id = "def-456"  → MemorySaver 存着 B 的对话历史
# 同一个 thread_id 的多次调用共享历史，实现多轮对话。
def create_agent_executor(model_type, workspace, sid, model_config, main_root, file_tree=""):
    """
    创建 Agent 执行环境，返回 (agent, checkpointer, config)。

    参数:
        model_type:  模型标识
        workspace:   工作区路径
        sid:         会话 ID（UUID），用作 thread_id 来隔离不同会话
        model_config: 模型配置字典
        main_root:   Tkinter 主窗口
        file_tree:   文件树字符串

    返回:
        agent:         可运行的 Agent 对象
        checkpointer:  MemorySaver 实例
        agent_config:  运行配置（包含 thread_id, recursion_limit 等）
    """
    # 创建内存检查点 —— 每次 LLM/工具调用后自动保存状态
    checkpointer = MemorySaver()

    # 创建 Agent 实例
    agent, tools = create_agent_instance(
        model_type, workspace, model_config, checkpointer, main_root, file_tree
    )

    # ========== 配置 Agent 运行参数 ==========
    # thread_id = sid：用会话 ID 隔离不同会话的记忆
    # recursion_limit = 100：最多允许 100 次工具调用循环（防止死循环）
    #   例如：用户提问 → read_file → edit_file → read_file → edit_file → ...
    #   每调用一次工具算一次递归，100 次足够了
    agent_config = RunnableConfig(
        configurable={"thread_id": sid},
        tags=["code_assistant", f"session_{sid}"],
        recursion_limit=100,
    )
    return agent, checkpointer, agent_config

"""
================================================================================
【面试要点速查】

Q: 这个项目的核心技术栈是什么？
A: LangChain + LangGraph 做 Agent 编排，ChromaDB 做向量检索(RAG)，
   DeepSeek 做 LLM 后端，Tkinter 做桌面 UI。

Q: 上下文管理是怎么做的？
A: 通过 smart_context_trimmer 对历史消息做"摘要化"处理：
   - 删除历史轮次的 ToolMessage（文件内容占几千tokens）
   - 替换为操作摘要（"read_file(xxx) | 理由"）
   - 保留最新轮完整内容
   - 通过 TrimmedChatOpenAI 透明拦截所有 LLM 调用

Q: 用什么模式调用 LLM？
A: stream() 流式输出，逐 token yield，前端显示打字机效果。

Q: 会话记忆怎么实现的？
A: LangGraph MemorySaver，基于 thread_id 的内存检查点。
   每次 LLM/工具调用后自动保存状态，下次同一 thread_id 恢复历史。

Q: Agent 是怎么执行任务的？
A: ReAct 循环：思考(tool_calls) → 执行工具 → 观察结果 → 继续思考 → ... → 输出文本。
   LangChain 的 create_agent 封装了这个循环。
================================================================================
"""
