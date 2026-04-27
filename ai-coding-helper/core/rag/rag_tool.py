"""
================================================================================
RAG 工具链 —— 智能代码检索的完整管道
================================================================================

【架构定位】
本模块是 AI 编程助手的信息检索层。当用户提问时，系统自动从工作区找到最相关
的代码文件，节省 Agent 盲目遍历文件的时间。

【RAG 完整管道】
用户查询 "优化登录验证逻辑"
    │
    ▼
Step1: _analyze_query_intent()     →  意图分析（代码搜索/配置搜索/文件搜索...）
    │
    ▼
Step2: _query_rewrite()            →  查询重写（同义词扩展："优化"→"编辑edit变更"）
    │
    ▼
Step3: ChromaDB 向量检索           →  在向量空间中找 Top-N 相似的代码符号
    │
    ▼
Step4: _rerank_results()           →  规则 Reranking（4个信号+1个惩罚重新打分）
    │      ├─ S1: 符号名称匹配
    │      ├─ S2: 文档字符串重叠
    │      ├─ S3: 符号类型对齐
    │      └─ S4: 文件路径匹配
    │
    ▼
Step5: 文件级去重                   →  同一个文件多个符号命中，取最高分
    │
    ▼
Step6: _determine_relevant_count()  →  动态阈值决定返回几个文件
    │
    ▼
返回 Top-K 相关文件列表 [{file_path, final_score, symbol_name, code_snippet}]

【面试高频题 —— RAG 检索增强生成】
Q11: 什么是 Embedding（嵌入/向量化）？
A:  将文本（如"def login(user, pwd):"）转换为一组高维浮点数向量（如1024维）。
    向量之间的距离代表语义相似度。"login函数"和"authenticate方法"的向量很近，
    "login函数"和"print输出"的向量很远。ChromaDB 用余弦距离衡量相似度。

Q12: 为什么需要 Reranking？向量检索不够吗？
A:  向量检索只看语义相似度，忽略了结构化信息（符号名匹配、类型对齐等）。
    例如查询"找到User类"，向量检索可能返回包含"user"变量的文件排第一，
    而 Reranking 通过规则信号（类型对齐加分、路径匹配加分）让 User 类排到前面。

Q13: 为什么使用 DashScope text-embedding-v3 而不是本地模型？
A:  云端嵌入模型质量更高（阿里云自研），且不需要本地 GPU。
    但需要网络请求，每次检索会有网络延迟。trade-off：质量 vs 速度。

Q14: 动态阈值 _determine_relevant_count 是什么原理？
A:  不同查询的分数分布差异大：
    - 精准查询 [0.93, 0.50, 0.45] → 第一名远超，返回1个
    - 全局查询 [0.84, 0.83, 0.83] → 分数聚类，全返回
    - 模糊查询 [0.52, 0.51, 0.49] → 都不高，但差不大
    用标准差(std)检测聚类 + 绝对阈值(0.40) + Elbow拐点截断 + Safety cap(10)。
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.tools import tool            # LangChain 工具装饰器
from pydantic import BaseModel, Field             # 工具入参校验
from core.rag.rag_manager import get_rag_manager
import os

logger = logging.getLogger(__name__)


# ==============================================================================
# 工具入参模型 —— 定义 retrieve_related_files 接受的参数
# ==============================================================================
class RetrieveRelatedFilesInput(BaseModel):
    query: str = Field(description="用户提问/检索关键词")
    top_k: int = Field(default=-1, description="手动指定返回数量，-1表示自动")


# ==============================================================================
# _determine_relevant_count —— 动态阈值决定返回多少个文件
# ==============================================================================
# 【为什么需要动态阈值】
# 不同查询的分数分布差异很大：
#   - 精准查询: [0.93, 0.50, 0.45, ...] → 第一名远超第二名，返回1个就行
#   - 全局查询: [0.84, 0.83, 0.83, ...] → 全挤在一起（分数聚类），应该全返回
#   - 模糊查询: [0.52, 0.51, 0.49, ...] → 都不高，但也都差不多
#
# 固定 TopK=5 对精准查询太多（浪费 token），对全局查询太少（遗漏文件）
def _determine_relevant_count(scores: List[float], query_intent: Dict[str, bool] = None) -> int:
    """
    动态决定返回多少个文件。

    决策流程：
    1. 分数聚类检测（std < 0.06）→ 全返回（全局分析任务）
    2. 极致置信快速路径（第一名远超第二名）→ 返回1个
    3. 动态阈值过线统计
    4. Elbow 拐点截断
    5. Safety cap=10 保底
    """
    if not scores:
        return 5

    n = len(scores)
    if n == 1:
        return 1

    SAFETY_CAP = 10
    scores_np = np.array(scores)

    mean_score = float(np.mean(scores_np))
    top_score = float(scores_np[0])
    std = float(np.std(scores_np))

    # ---- 分数聚类检测：如果所有分数都挤在一起（标准差极小），全返回 ----
    # 这通常发生在"分析整个项目"这类全局查询上
    if std < 0.06 and mean_score > 0.5 and n >= 5:
        return min(n, SAFETY_CAP)

    # ---- 正常阈值计算 ----
    floor = 0.40
    if query_intent:
        if query_intent.get("is_config_search") or query_intent.get("is_file_search"):
            floor = 0.35  # 配置文件/文件搜索降低门槛
    # 动态阈值 = max(绝对下限, 跟随均值, 防单点拉升)
    threshold = max(floor, mean_score * 0.80, top_score * 0.60)

    # ---- 极致置信快速路径 ----
    # 如果第一名 > 0.90 且领先第二名 > 0.35 → 只需返回这一个文件
    if n >= 2:
        second_score = float(scores_np[1])
        if top_score > 0.90 and (top_score - second_score) > 0.35:
            return 1
        if top_score > 0.85 and (top_score - second_score) > 0.30:
            return 1

    # ---- 过线统计 ----
    above = scores_np >= threshold
    count = int(np.sum(above))

    if count == 0:
        logger.warning(f"最高分={top_score:.3f} < 阈值={threshold}，仅推荐最佳候选")
        return 1   # 保底：至少返回1个

    # ---- Elbow 拐点截断 ----
    # 在分数下降曲线中找到"拐点"（下降速度突然加快的位置），在拐点处截断
    if count >= 3 and n >= 4:
        above_last = min(count, n - 1)
        drops = [float(scores_np[i] - scores_np[i + 1]) for i in range(above_last)]
        max_drop_idx = int(np.argmax(drops))
        max_drop = drops[max_drop_idx]
        avg_drop = float(np.mean(drops))
        if max_drop > avg_drop * 2.5 and max_drop > 0.10:
            elbow_k = max_drop_idx + 1
            count = min(count, elbow_k)

    return min(count, SAFETY_CAP, n)


# ==============================================================================
# _analyze_query_intent —— 查询意图分析
# ==============================================================================
# 通过关键词匹配判断用户想做什么，为后续的 Reranking 提供依据。
# 例如：用户说"找一下class定义" → is_class_search=True → 类符号加分
def _analyze_query_intent(query: str) -> Dict[str, bool]:
    """分析查询意图，为检索策略和 Reranking 提供依据"""
    query_lower = query.lower()

    intent = {
        "is_code_search": False,
        "is_config_search": False,
        "is_file_search": False,
        "is_class_search": False,
        "is_function_search": False,
        "is_implementation_search": False,
        "is_api_search": False
    }

    code_keywords = ["代码", "函数", "类", "方法", "变量", "import", "def ", "class ", "实现", "源码", "源代码"]
    if any(keyword in query_lower for keyword in code_keywords):
        intent["is_code_search"] = True

    impl_keywords = ["实现", "逻辑", "算法", "如何", "怎么", "步骤", "流程", "处理", "解析"]
    if any(keyword in query_lower for keyword in impl_keywords):
        intent["is_implementation_search"] = True

    file_keywords = ["文件", "路径", ".py", ".json", "目录", "文件夹", "脚本", "模块", "包"]
    if any(keyword in query_lower for keyword in file_keywords):
        intent["is_file_search"] = True

    class_keywords = ["class", "类", "继承", "基类", "父类", "子类", "对象", "实例", "属性"]
    if any(keyword in query_lower for keyword in class_keywords):
        intent["is_class_search"] = True

    function_keywords = ["函数", "function", "def ", "方法", "调用", "参数", "返回值", "返回", "签名", "定义"]
    if any(keyword in query_lower for keyword in function_keywords):
        intent["is_function_search"] = True

    api_keywords = ["api", "接口", "路由", "endpoint", "请求", "响应", "http", "rest"]
    if any(keyword in query_lower for keyword in api_keywords):
        intent["is_api_search"] = True

    config_keywords = ["配置", "config", "设置", "参数", "环境变量", "选项", "开关", "启用", "禁用"]
    if any(keyword in query_lower for keyword in config_keywords):
        intent["is_config_search"] = True

    return intent


# ==============================================================================
# _query_rewrite —— 查询重写
# ==============================================================================
# 对用户查询做同义词扩展，利用技术术语的多种表达方式提升召回率。
# 例如："修改" → "修改 编辑 edit 变更"，让向量检索能从更多维度匹配
def _query_rewrite(original_query: str) -> str:
    """查询重写：同义词扩展，提升向量检索的召回率"""
    query = original_query.lower()

    synonyms = {
        "怎么": "如何", "怎样": "如何", "哪里": "位置 路径", "哪个": "哪个文件",
        "代码": "源码 源代码 实现", "函数": "方法 def", "类": "class 对象",
        "文件": "路径 目录 文件夹", "配置": "设置 参数 config",
        "读取": "打开 open load", "写入": "保存 write dump",
        "删除": "移除 remove delete", "修改": "编辑 edit 变更",
        "创建": "新建 make create", "移动": "剪切 move", "复制": "拷贝 copy",
    }

    rewritten = original_query
    for key, value in synonyms.items():
        if key in query:
            rewritten += " " + value

    tech_terms = ["python", "代码", "函数", "类", "模块", "包", "导入", "变量", "常量"]
    for term in tech_terms:
        if term not in rewritten.lower():
            if any(word in query for word in ["如何", "怎么", "怎样", "实现", "使用"]):
                rewritten += " " + term
                break

    return rewritten.strip()


# ==============================================================================
# Reranking 管道 —— 4个正向信号 + 1个惩罚 → 自适应缩放 → 三级Cap
# ==============================================================================
# ChromaDB 的向量相似度只是基础分，我们在上面叠加规则信号重新打分。
# 【为什么需要 Reranking】
# 向量检索只看语义相似度，忽略了：
#   - 符号名称是否直接匹配查询词（名称匹配应该加分）
#   - 符号类型是否符合用户意图（找class时function应该降低权重）
#   - 文件路径是否包含查询中的关键词
# Reranking 就是把这些"人类常识"加回去。

def _name_match_bonus(symbol_name: str, query: str) -> float:
    """信号1(S1)：符号名称命中查询词（0~0.20，最强信号）
    例如：查询"login" → 符号名"loginUser" → 精确命中 → +0.15"""
    if not symbol_name:
        return 0.0
    query_lower = query.lower().strip()
    name_lower = symbol_name.lower().strip()
    if not query_lower or not name_lower:
        return 0.0
    if query_lower == name_lower:                    return 0.20   # 完全匹配
    if query_lower in name_lower:                    return 0.15   # 查询是符号名的子串
    if name_lower in query_lower:                    return 0.08   # 符号名是查询的子串
    # Token 级重叠匹配
    query_tokens = set(q for q in query_lower.replace('_', ' ').replace('-', ' ').split() if len(q) > 1)
    name_tokens = set(n for n in name_lower.replace('_', ' ').replace('-', ' ').split() if len(n) > 1)
    overlap = query_tokens & name_tokens
    if overlap:
        return min(0.15, 0.05 * len(overlap))
    return 0.0


def _docstring_match_bonus(docstring: str, query: str) -> float:
    """信号2(S2)：文档字符串与查询重叠（0~0.12）"""
    if not docstring:
        return 0.0
    query_lower = query.lower()
    doc_lower = docstring.lower()
    if query_lower in doc_lower:                      return 0.10
    # Token 匹配
    query_terms = [t.strip() for t in query_lower.replace('_', ' ').replace('-', ' ').split() if len(t.strip()) > 1]
    matched_terms = sum(1 for t in query_terms if t in doc_lower)
    if matched_terms >= 2:                            return 0.10
    if matched_terms == 1 and len(query_terms) >= 2:  return 0.06
    if doc_lower in query_lower:                      return 0.06
    # 去停用词后的 token 重叠率
    stop_words = {"的", "了", "是", "在", "和", "与", "或", "一个", "这个",
                  "the", "a", "an", "is", "are", "in", "to", "for", "of", "and", "that", "it", "on", "at", "by"}
    query_tokens = set(t for t in query_lower.split() if t not in stop_words and len(t) > 1)
    doc_tokens = set(t for t in doc_lower.split() if t not in stop_words and len(t) > 1)
    if not query_tokens:                              return 0.0
    overlap = len(query_tokens & doc_tokens)
    ratio = overlap / len(query_tokens)
    if ratio > 0.5:                                   return 0.12
    elif ratio > 0.3:                                 return 0.08
    elif ratio > 0.1:                                 return 0.04
    return 0.0


def _type_alignment_bonus(symbol_type: str, query_intent: Dict[str, bool]) -> float:
    """信号3(S3)：符号类型与查询意图对齐（0~0.10）
    找class → class符号加分，找function → function符号加分"""
    if not query_intent:
        return 0.0
    preferences = {
        "is_class_search":             {"class": 0.10},
        "is_function_search":          {"function": 0.10, "method": 0.08},
        "is_config_search":            {"variable": 0.08, "declaration": 0.06, "macro": 0.06},
        "is_api_search":               {"function": 0.08, "class": 0.06},
        "is_implementation_search":    {"function": 0.06, "method": 0.06, "class": 0.04},
        "is_code_search":              {"function": 0.04, "method": 0.04, "class": 0.04, "import": 0.02},
    }
    for intent_key, type_map in preferences.items():
        if query_intent.get(intent_key):
            if symbol_type in type_map:
                return type_map[symbol_type]
    return 0.0


def _filepath_match_bonus(file_path: str, query: str) -> float:
    """信号4(S4)：文件路径匹配（0~0.12）
    查询包含文件路径关键词时加分"""
    if not file_path or not query:
        return 0.0
    query_lower = query.lower().strip()
    path_lower = file_path.lower()
    path_clean = path_lower.replace('\\', '/').replace('/', ' ').replace('_', ' ').replace('-', ' ')
    path_tokens = set(t for t in path_clean.split() if len(t) > 1)
    query_tokens = set(t for t in query_lower.replace('_', ' ').replace('-', ' ').split() if len(t) > 1)
    if not query_tokens or not path_tokens:
        return 0.0
    overlap = query_tokens & path_tokens
    if overlap:
        return min(0.12, 0.06 * len(overlap))
    return 0.0


def _type_mismatch_penalty(symbol_type: str, query_intent: Dict[str, bool]) -> float:
    """惩罚项(P1)：符号类型与查询意图冲突时扣分（-0.05~0）
    找class → import符号不应该排在前面 → 扣分"""
    if not query_intent:
        return 0.0
    total_penalty = 0.0
    if query_intent.get("is_class_search"):
        if symbol_type not in ("class", "declaration"):
            total_penalty -= 0.05
    if query_intent.get("is_config_search"):
        if symbol_type not in ("variable", "declaration", "macro"):
            total_penalty -= 0.03
    if query_intent.get("is_function_search") and not query_intent.get("is_code_search"):
        if symbol_type not in ("function", "method"):
            total_penalty -= 0.03
    return total_penalty


def _adaptive_bonus_scale(base_score: float, raw_bonus: float) -> float:
    """自适应缩放：基础分越高，bonus 衰减越多（防分数膨胀到1.0以上）"""
    if base_score > 0.90:      scale = 0.3
    elif base_score > 0.80:    scale = 0.6
    elif base_score < 0.40:    scale = 1.3
    else:                      scale = 1.0
    return raw_bonus * scale


def _tiered_cap(bonus: float) -> float:
    """三级Cap：平滑分段上限，防止单个信号拉分太多"""
    if bonus <= 0.20:          return bonus
    elif bonus <= 0.35:        return 0.20 + (bonus - 0.20) * 0.5
    else:                      return min(0.30, 0.275 + (bonus - 0.35) * 0.2)


def _rerank_results(details, query, query_intent):
    """
    Reranking 主函数：对 ChromaDB 初排结果做规则重排。

    流程：base_score + (S1+S2+S3+S4+P1) → 自适应缩放 → 三级Cap → final_score
    最终按 final_score 降序排列。
    """
    for d in details:
        base = d.get("weighted_score", 0.0)

        s1 = _name_match_bonus(d.get("symbol_name", ""), query)
        s2 = _docstring_match_bonus(d.get("docstring", ""), query)
        s3 = _type_alignment_bonus(d.get("symbol_type", ""), query_intent)
        s4 = _filepath_match_bonus(d.get("file_path", ""), query)
        p1 = _type_mismatch_penalty(d.get("symbol_type", ""), query_intent)

        raw_bonus = s1 + s2 + s3 + s4 + p1
        scaled_bonus = _adaptive_bonus_scale(base, raw_bonus)
        final_bonus = _tiered_cap(scaled_bonus)

        d["rerank_bonus"] = final_bonus
        d["rerank_detail"] = {
            "name_match": round(s1, 3), "doc_match": round(s2, 3),
            "type_align": round(s3, 3), "path_match": round(s4, 3),
            "penalty": round(p1, 3), "raw": round(raw_bonus, 3),
            "scaled": round(scaled_bonus, 3),
        }
        d["final_score"] = base + final_bonus

    details.sort(key=lambda x: x["final_score"], reverse=True)
    return details


# ==============================================================================
# retrieve_related_files_structured —— 结构化检索（外部可调用）
# ==============================================================================
# 这是非 tool 包装的检索函数，返回结构化数据而非 Markdown 字符串。
# handlers.py 的 _augment_prompt_with_rag 调用此函数获取结构化结果后自行格式化。
def retrieve_related_files_structured(query: str, top_k_input: int = -1) -> List[Dict[str, Any]]:
    """完整的 RAG 检索管道，返回结构化结果"""
    rag_manager = get_rag_manager()
    if not rag_manager:
        return []

    query_intent = _analyze_query_intent(query)
    rewritten_query = _query_rewrite(query)

    # 从 ChromaDB 获取 Top-50 候选（扩大范围供 Reranking 筛选）
    details = rag_manager.search_related_files_with_details(rewritten_query, top_k=50)
    if not details:
        return []

    # Reranking: 在向量相似度基础上叠加规则信号
    details = _rerank_results(details, query, query_intent)

    # 文件级去重：同一文件多个符号命中 → 取最高分
    file_best: Dict[str, dict] = {}
    for d in details:
        fp = d["file_path"]
        if fp not in file_best or d["final_score"] > file_best[fp]["final_score"]:
            file_best[fp] = d

    file_paths = list(file_best.keys())
    scores = [file_best[fp]["final_score"] for fp in file_paths]

    # 动态阈值决定返回数量
    if top_k_input <= 0:
        top_k = _determine_relevant_count(scores, query_intent=query_intent)
    else:
        top_k = min(top_k_input, len(file_paths))

    result = []
    for i, fp in enumerate(file_paths[:top_k]):
        best = file_best[fp]
        result.append({
            "file_path": fp,
            "final_score": best["final_score"],
            "symbol_name": best.get("symbol_name", ""),
            "symbol_type": best.get("symbol_type", ""),
            "code_snippet": best.get("code_snippet", ""),
            "rerank_bonus": best.get("rerank_bonus", 0),
        })
    return result


# ==============================================================================
# get_rag_tools —— 注册为 LangChain 工具的 RAG 检索
# ==============================================================================
# 这个函数返回的 retrieve_related_files 是 @tool 装饰的 LangChain 工具，
# Agent 可以像调用 read_file 一样调用它来检索相关文件。
def get_rag_tools():
    @tool(args_schema=RetrieveRelatedFilesInput)
    def retrieve_related_files(**kwargs):
        """检索与用户提问相关的代码文件路径"""
        try:
            query = kwargs["query"]
            top_k_input = kwargs.get("top_k", -1)

            # 调用结构化检索管道
            results = retrieve_related_files_structured(query, top_k_input)
            if not results:
                return "未检索到相关代码文件"

            query_intent = _analyze_query_intent(query)
            rewritten_query = _query_rewrite(query)
            top_k = len(results)
            scores = [r["final_score"] for r in results]

            # 格式化为 Markdown 返回给 Agent
            result_parts = ["## 智能代码检索结果"]
            result_parts.append(f"**原始查询**: {query}")
            if rewritten_query != query:
                result_parts.append(f"**重写查询**: {rewritten_query}")
            result_parts.append("")

            for i, r in enumerate(results):
                fp = r["file_path"]
                score = r["final_score"]
                rerank_info = ""
                if r.get("rerank_bonus", 0) > 0:
                    rerank_info = f" [+{r['rerank_bonus']:.2f} Rerank]"

                if score > 0.85:       score_str = f"火 {score:.3f} (极高置信度{rerank_info})"
                elif score > 0.7:      score_str = f"星 {score:.3f} (高置信度{rerank_info})"
                elif score > 0.55:     score_str = f"✓ {score:.3f} (中等置信度{rerank_info})"
                else:                  score_str = f"• {score:.3f} (低置信度{rerank_info})"

                result_parts.append(f"### {i+1}. [{score_str}] {fp}")
                result_parts.append(f"**匹配符号**: `{r['symbol_name']}` ({r['symbol_type']})")
                snippet = r.get("code_snippet", "")
                if snippet:
                    result_parts.append(f"```\n{snippet}\n```")
                result_parts.append(f"**建议**: 使用 `read_file('{fp}')` 读取完整文件")
                result_parts.append("")

            if len(scores) > 0:
                mean_score = float(np.mean(scores))
                result_parts.append(f"**平均综合分**: {mean_score:.3f}")
                if mean_score > 0.75:
                    result_parts.append("**质量评估**: 检索结果质量很高")
                elif mean_score > 0.6:
                    result_parts.append("**质量评估**: 结果质量中等，建议更具体的查询词")
                else:
                    result_parts.append("**质量评估**: 相似度较低，建议修改查询词")

            return "\n".join(result_parts)

        except Exception as e:
            return f"检索失败：{str(e)}"

    return [retrieve_related_files]


"""
================================================================================
【面试要点速查】

Q: RAG 管道是怎么工作的？
A: 6步管道：意图分析 → 查询重写 → ChromaDB向量检索 → 规则Reranking(4信号+1惩罚)
   → 文件去重 → 动态阈值筛选。最终返回 Top-K 个最相关的代码文件。

Q: 为什么需要 Reranking？
A: 向量相似度只看语义，忽略了符号名称、类型、路径的结构化信息。
   Reranking 把这些规则信号叠加上去，提高检索精度。

Q: 动态阈值 vs 固定 TopK 的区别？
A: 固定 TopK 对精准查询返回太多（浪费token），对全局查询返回太少（遗漏文件）。
   动态阈值根据分数分布自动调整：聚类→全返回，高分独占→返回1个，正常→正常阈值。

Q: 查询重写做了什么？
A: 同义词扩展。"修改"→"修改 编辑 edit 变更"，让向量检索匹配到更多相关符号。

Q: 系统用的向量数据库是？
A: ChromaDB，开源、轻量、支持本地持久化，用阿里云 DashScope text-embedding-v3 做嵌入。
================================================================================
"""
