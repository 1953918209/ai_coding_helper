"""
================================================================================
向量存储层 —— 基于 ChromaDB 的代码符号向量索引
================================================================================

【技术选型】
ChromaDB：开源、轻量级向量数据库，支持本地持久化，无需独立部署服务。
嵌入模型：阿里云 DashScope text-embedding-v3

【核心概念 —— Embedding 嵌入向量】
- Embedding（嵌入/向量化）：将文本（如函数名+代码片段）转换成高维向量（浮点数组）
  例如 "def login()" → [0.23, -0.45, 0.78, ..., 0.12] (通常1024维)
- Similarity Search（相似度搜索）：在向量空间中找与查询最接近的向量
  度量方式：余弦距离(Cosine Distance)，范围[0,2]，越小越相似
- ChromaDB collection：类似关系数据库的"表"，存放所有代码符号的向量+元数据
- Persist Directory：持久化目录 .code_rag_index/，底层用 SQLite + Parquet 双存储

【ChromaDB 内部原理】
底层使用 hnswlib（Hierarchical Navigable Small World）索引：
- 近似最近邻搜索(ANN)，在精度和速度间取平衡
- 构建多层图结构，搜索时从高层快速逼近目标区域
- 比暴力搜索快几个数量级（O(log N) vs O(N)）

【存储结构】
每个代码符号存为一条记录：
  - 文本(text)：用于生成向量的拼接文本（文件路径+符号类型+符号名+代码片段）
  - 元数据(metadata)：{file_path, symbol_name, symbol_type, line_start, ...}
  - 向量(embedding)：DashScope 自动生成的 n 维浮点向量

【检索流程】
用户查询 "登录验证"
  → DashScope 将查询转为向量 Q
  → ChromaDB 在 .code_rag_index 中找与 Q 最接近的 Top-N 个向量
  → 返回对应的文件路径 + 元数据 + 相似度分数

【面试高频题 —— 向量数据库/ChromaDB】
Q15: 为什么选 ChromaDB 而不是 Pinecone/Weaviate/Milvus？
A:  ChromaDB 是嵌入式向量数据库（Python 库直接引入），无需独立部署，
    适合本地桌面应用。Pinecone 是云服务（需付费），Milvus 需要 Docker 部署。
    对于个人项目/中小规模（<100万向量），ChromaDB 是最简单的选择。

Q16: 向量检索的相似度(1-distance/2)和加权分(0.7*sim+0.3*imp)分别是什么？
A:  ChromaDB 返回的是 L2 距离的平方（距离越近越相似），范围 0~2。
    转相似度：similarity = 1.0 - (distance / 2.0)
    加权分：weighted_score = 0.7 * similarity + 0.3 * importance_score
    把代码的重要性（class>function>import）融入排序，让重要符号排名更靠前。
"""
import os
import logging
import uuid
import gc
import json
import numpy as np
from typing import List, Tuple, Dict, Any
from langchain_chroma import Chroma                       # LangChain 的 ChromaDB 封装
from langchain_core.embeddings import Embeddings           # 嵌入函数抽象接口
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云 DashScope 嵌入
from core.rag.code_parser import CodeSymbol
from ui.config import RAG_COLLECTION_NAME, RAG_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class CodeVectorStore:
    """代码向量存储：封装 ChromaDB 的增删改查操作。

    每个工作区有独立的 .code_rag_index/ 目录存储向量数据。
    """

    def __init__(self, workspace: str):
        self.workspace = workspace
        # API Key 从环境变量读取（安全性考虑，不写死在代码里）
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not self.dashscope_api_key:
            logger.warning("未检测到DASHSCOPE_API_KEY环境变量，RAG功能将不可用")

        # 持久化目录：workspace/.code_rag_index/
        self.persist_dir = os.path.join(workspace, ".code_rag_index")
        os.makedirs(self.persist_dir, exist_ok=True)

        # 嵌入函数：把文本变成高维向量
        self.embeddings: Embeddings = DashScopeEmbeddings(
            dashscope_api_key=self.dashscope_api_key,
            model=RAG_EMBEDDING_MODEL,  # text-embedding-v3
        )

        # 初始化 ChromaDB 连接（collection_name 相当于表名）
        self.vector_store = Chroma(
            collection_name=RAG_COLLECTION_NAME,      # "code_assistant_symbols"
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir         # 数据存哪里
        )
        logger.info(f"Chroma向量库已初始化，持久化目录：{self.persist_dir}")

    # ========================================================================
    # close() —— 优雅关闭
    # ========================================================================
    # ChromaDB 底层使用 SQLite，需要显式关闭释放文件句柄。
    # 但保留所有持久化数据，下次打开时自动恢复。
    def close(self):
        """显式关闭向量库，释放 SQLite 连接和文件句柄"""
        try:
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                if hasattr(self.vector_store, '_client'):
                    client = self.vector_store._client
                    if hasattr(client, '_system'):
                        try:
                            client._system.stop()
                        except Exception:
                            pass
                    if hasattr(client, '_server') and hasattr(client._server, 'close'):
                        try:
                            client._server.close()
                        except Exception:
                            pass
                self.vector_store = None
                gc.collect()   # 强制垃圾回收，确保文件句柄释放
                logger.info("Chroma 向量库已显式关闭")
        except (OSError, IOError) as e:
            logger.warning(f"关闭向量库时出错（可忽略）：{e}")

    # ========================================================================
    # add_symbols() —— 批量写入符号向量
    # ========================================================================
    # 将代码符号列表写入 ChromaDB。每个符号生成：
    #   - texts:      嵌入文本（符号名+文件路径+代码片段拼接）
    #   - metadatas:  结构化元数据（用于过滤和查询）
    #   - ids:        UUID 唯一标识
    def add_symbols(self, symbols: List[CodeSymbol]):
        """批量添加代码符号到向量库"""
        if not symbols:
            return
        texts = []
        metadatas = []
        ids = []
        for symbol in symbols:
            texts.append(symbol.to_embedding_text())  # 生成嵌入文本
            metadatas.append({
                "file_path": symbol.file_path,
                "symbol_name": symbol.symbol_name,
                "symbol_type": symbol.symbol_type,
                "line_start": symbol.line_start,
                "line_end": symbol.line_end,
                "code_snippet": symbol.code_snippet,
                "docstring": symbol.docstring,
                "importance_score": symbol.importance_score,
            })
            ids.append(str(uuid.uuid4()))  # 全局唯一ID

        # 调用 ChromaDB 批量写入
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"添加 {len(symbols)} 个符号到向量库")

    # ========================================================================
    # search_related_files —— 检索相关文件（核心功能）
    # ========================================================================
    # 以下是三种检索接口，从简单到详细：
    #   1. search_related_files()         → 只返回文件路径
    #   2. search_related_files_with_scores() → 返回 (文件路径, 分数)
    #   3. search_related_files_with_details() → 返回完整元数据字典

    def search_related_files(self, query: str, top_k: int = 5) -> List[str]:
        """基础检索：只返回文件路径列表"""
        scored = self.search_related_files_with_scores(query, top_k)
        return [fp for fp, _ in scored]

    def search_related_files_with_scores(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        带分数的检索。
        使用 importance_weight 对结果加权：
          final_score = 0.7 × similarity + 0.3 × importance
        这样重要的符号（如public函数）比次要符号（如import语句）排名更靠前。
        """
        try:
            # 扩大搜索范围（top_k*2），供加权筛选
            raw_results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)

            vector_results = []
            for doc, distance in raw_results:
                file_path = doc.metadata["file_path"]
                # ChromaDB 返回的是 distance（余弦距离），转成 similarity（相似度）
                similarity = 1.0 - (distance / 2.0)

                # 重要性加权
                try:
                    importance = float(doc.metadata.get("importance_score", 1.0))
                    weighted_score = 0.7 * similarity + 0.3 * importance
                except (ValueError, TypeError):
                    weighted_score = similarity

                vector_results.append((file_path, weighted_score))

            # 按加权分排序
            vector_results.sort(key=lambda x: x[1], reverse=True)

            # 文件级去重：同一文件取最高分
            seen_files = set()
            deduplicated = []
            for fp, score in vector_results:
                if fp not in seen_files:
                    seen_files.add(fp)
                    deduplicated.append((fp, score))
                    if len(seen_files) >= top_k:
                        break

            return deduplicated
        except Exception as e:
            logger.warning(f"向量检索失败：{e}")
            return []

    def search_related_files_with_details(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        详细检索：返回完整元数据字典。
        供 RAG 管道的 Reranking 阶段使用。
        """
        try:
            raw_results = self.vector_store.similarity_search_with_score(query, k=top_k * 3)
            detailed_results = []
            for doc, distance in raw_results:
                similarity = 1.0 - (distance / 2.0)
                importance = float(doc.metadata.get("importance_score", 1.0))
                weighted_score = 0.7 * similarity + 0.3 * importance

                detailed_results.append({
                    "file_path": doc.metadata["file_path"],
                    "symbol_name": doc.metadata.get("symbol_name", ""),
                    "symbol_type": doc.metadata.get("symbol_type", ""),
                    "similarity": similarity,
                    "importance": importance,
                    "weighted_score": weighted_score,   # 综合分（用于初排）
                    "code_snippet": doc.metadata.get("code_snippet", ""),
                    "docstring": doc.metadata.get("docstring", "")[:100],
                })

            detailed_results.sort(key=lambda x: x["weighted_score"], reverse=True)
            # 文件去重
            seen = set()
            final = []
            for r in detailed_results:
                fp = r["file_path"]
                if fp not in seen:
                    seen.add(fp)
                    final.append(r)
                    if len(seen) >= top_k:
                        break
            return final
        except Exception as e:
            logger.warning(f"详细检索失败：{e}")
            return []

    # ========================================================================
    # 索引维护方法
    # ========================================================================
    def delete_file_symbols(self, file_path: str):
        """从索引中删除指定文件的所有符号（按 file_path 过滤删除）"""
        try:
            self.vector_store.delete(where={"file_path": file_path})
            logger.info(f"删除文件 {file_path} 的旧符号索引")
        except Exception as e:
            logger.error(f"删除文件符号失败：{str(e)}")

    def clear_index(self):
        """
        清空向量库集合。
        使用 delete_collection() 而非删除目录，更稳定可靠。
        同时清理 ChromaDB 遗留的孤立 UUID 文件夹。
        """
        try:
            self.vector_store.delete_collection()
            self._cleanup_orphan_uuid_folders()
            # 重建 collection
            self.vector_store = Chroma(
                collection_name=RAG_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
            logger.info("向量库索引已清空")
        except Exception as e:
            logger.error(f"清空索引失败：{str(e)}")
            # 异常恢复：确保 vector_store 可用
            try:
                self.vector_store = Chroma(
                    collection_name=RAG_COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_dir
                )
            except Exception:
                pass

    def _cleanup_orphan_uuid_folders(self):
        """清理 ChromaDB 遗留的孤立 UUID 文件夹（如 a1b2c3d4-...）"""
        import re, shutil
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        for name in os.listdir(self.persist_dir):
            full = os.path.join(self.persist_dir, name)
            if os.path.isdir(full) and uuid_pattern.match(name):
                shutil.rmtree(full, ignore_errors=True)
                logger.info(f"已清理旧UUID文件夹：{name}")

    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计（符号总数、类型分布、平均重要性等）"""
        try:
            collection = self.vector_store._client.get_collection(RAG_COLLECTION_NAME)
            count = collection.count()
            results = collection.peek(limit=min(100, count))
            symbol_types = {}
            importance_scores = []
            for i, _ in enumerate(results['documents']):
                meta = results['metadatas'][i]
                st = meta.get('symbol_type', 'unknown')
                symbol_types[st] = symbol_types.get(st, 0) + 1
                try:
                    importance_scores.append(float(meta.get('importance_score', 1.0)))
                except (ValueError, TypeError):
                    pass
            return {
                "total_symbols": count,
                "symbol_type_distribution": symbol_types,
                "avg_importance_score": np.mean(importance_scores) if importance_scores else 0,
            }
        except Exception as e:
            return {"total_symbols": 0, "error": str(e)}
