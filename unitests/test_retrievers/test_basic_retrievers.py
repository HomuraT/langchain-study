"""
基础检索器测试

测试LangChain基础检索器功能，包括：
- VectorStoreRetriever：向量存储检索器的基本使用
- 相似性搜索：基于向量相似性的文档检索
- MMR搜索：最大边际相关性搜索，避免结果冗余
- 阈值搜索：基于相似性分数的过滤检索
- 搜索参数配置：k值、阈值等参数的测试

作者: LinRen
创建时间: 2025年
"""

import unittest
import os
import sys
import time
from typing import List, Dict, Any
import tempfile

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config.api import apis


class TestBasicRetrievers(unittest.TestCase):
    """
    基础检索器测试类
    """
    
    def setUp(self) -> None:
        """
        测试前的初始化设置
        
        Args:
            None
            
        Returns:
            None
        """
        # 获取API配置
        self.config = apis["local"]
        
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"]
        )
        
        # 创建测试文档
        self.test_docs = [
            Document(
                page_content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                metadata={"source": "ai_intro", "topic": "人工智能"}
            ),
            Document(
                page_content="机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
                metadata={"source": "ml_intro", "topic": "机器学习"}
            ),
            Document(
                page_content="深度学习是机器学习的一个分支，使用多层神经网络来处理和分析复杂数据。",
                metadata={"source": "dl_intro", "topic": "深度学习"}
            ),
            Document(
                page_content="自然语言处理是人工智能的一个重要应用领域，涉及计算机与人类语言的交互。",
                metadata={"source": "nlp_intro", "topic": "自然语言处理"}
            ),
            Document(
                page_content="计算机视觉是另一个重要的人工智能应用，使计算机能够理解和解释视觉信息。",
                metadata={"source": "cv_intro", "topic": "计算机视觉"}
            ),
            Document(
                page_content="强化学习是机器学习的一种方法，通过与环境的交互来学习最优行为策略。",
                metadata={"source": "rl_intro", "topic": "强化学习"}
            )
        ]
        
        # 创建向量存储
        self.vectorstore = FAISS.from_documents(self.test_docs, self.embeddings)
        
        # 创建基础检索器
        self.base_retriever = self.vectorstore.as_retriever()
    
    def test_basic_retriever_creation(self) -> None:
        """
        测试基础检索器创建
        
        Args:
            None
            
        Returns:
            None
        """
        # 测试默认检索器创建
        retriever = self.vectorstore.as_retriever()
        self.assertIsNotNone(retriever)
        self.assertIsInstance(retriever, BaseRetriever)
        
        # 测试检索器属性
        self.assertEqual(retriever.search_type, "similarity")
        self.assertEqual(retriever.search_kwargs.get("k", 4), 4)
    
    def test_similarity_search(self) -> None:
        """
        测试相似性搜索功能
        
        Args:
            None
            
        Returns:
            None
        """
        # 创建相似性检索器
        retriever = self.vectorstore.as_retriever(search_type="similarity")
        
        # 测试基础查询
        query = "什么是机器学习？"
        docs = retriever.invoke(query)
        
        # 验证结果
        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)
        self.assertLessEqual(len(docs), 4)  # 默认k=4
        
        # 验证文档内容
        for doc in docs:
            self.assertIsInstance(doc, Document)
            self.assertIsInstance(doc.page_content, str)
            self.assertIsInstance(doc.metadata, dict)
        
        # 检查是否包含相关文档
        contents = [doc.page_content for doc in docs]
        has_ml_content = any("机器学习" in content for content in contents)
        self.assertTrue(has_ml_content, "结果应包含机器学习相关内容")
    
    def test_mmr_search(self) -> None:
        """
        测试MMR（最大边际相关性）搜索
        
        Args:
            None
            
        Returns:
            None
        """
        # 创建MMR检索器
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 6, "lambda_mult": 0.5}
        )
        
        # 测试MMR查询
        query = "人工智能的应用领域"
        docs = retriever.invoke(query)
        
        # 验证结果
        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)
        self.assertLessEqual(len(docs), 3)
        
        # MMR应该返回多样化的结果
        topics = [doc.metadata.get("topic", "") for doc in docs]
        unique_topics = set(topics)
        
        # 如果有多个文档，应该有不同的主题
        if len(docs) > 1:
            self.assertGreater(len(unique_topics), 1, "MMR应该返回多样化的结果")
    
    def test_similarity_score_threshold(self) -> None:
        """
        测试相似性分数阈值搜索
        
        Args:
            None
            
        Returns:
            None
        """
        # 创建阈值检索器
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7}
        )
        
        # 测试高相关性查询
        high_relevance_query = "机器学习算法"
        docs = retriever.invoke(high_relevance_query)
        
        # 应该有结果（高相关性）
        self.assertIsInstance(docs, list)
        
        # 测试低相关性查询
        low_relevance_query = "厨房烹饪技巧"
        docs_low = retriever.invoke(low_relevance_query)
        
        # 低相关性查询可能没有结果
        self.assertIsInstance(docs_low, list)
        # 低相关性查询的结果应该少于高相关性查询
        self.assertLessEqual(len(docs_low), len(docs))
    
    def test_search_kwargs_configuration(self) -> None:
        """
        测试搜索参数配置
        
        Args:
            None
            
        Returns:
            None
        """
        # 测试不同的k值
        k_values = [1, 2, 5]
        for k in k_values:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke("人工智能")
            
            self.assertLessEqual(len(docs), k, f"返回文档数应不超过k={k}")
            self.assertGreater(len(docs), 0, f"k={k}时应有结果")
    
    def test_retriever_invoke_method(self) -> None:
        """
        测试检索器invoke方法
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = self.base_retriever
        
        # 测试字符串查询
        docs = retriever.invoke("深度学习")
        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)
        
        # 测试空查询
        empty_docs = retriever.invoke("")
        self.assertIsInstance(empty_docs, list)
        
        # 测试特殊字符查询
        special_docs = retriever.invoke("@#$%^&*()")
        self.assertIsInstance(special_docs, list)
    
    def test_retriever_performance(self) -> None:
        """
        测试检索器性能
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = self.base_retriever
        query = "自然语言处理技术"
        
        # 测试单次检索性能
        start_time = time.time()
        docs = retriever.invoke(query)
        end_time = time.time()
        
        retrieval_time = end_time - start_time
        
        # 验证结果
        self.assertIsInstance(docs, list)
        self.assertLess(retrieval_time, 10.0, "单次检索应在10秒内完成")
        
        # 测试批量检索性能
        queries = [
            "人工智能应用",
            "机器学习模型", 
            "深度学习网络",
            "强化学习策略"
        ]
        
        start_time = time.time()
        results = []
        for q in queries:
            result = retriever.invoke(q)
            results.append(result)
        end_time = time.time()
        
        batch_time = end_time - start_time
        avg_time = batch_time / len(queries)
        
        self.assertEqual(len(results), len(queries))
        self.assertLess(avg_time, 5.0, "平均检索时间应在5秒内")
    
    def test_document_metadata_preservation(self) -> None:
        """
        测试文档元数据保持
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = self.base_retriever
        docs = retriever.invoke("计算机视觉")
        
        # 验证元数据保持完整
        for doc in docs:
            self.assertIsInstance(doc.metadata, dict)
            if "topic" in doc.metadata:
                self.assertIsInstance(doc.metadata["topic"], str)
            if "source" in doc.metadata:
                self.assertIsInstance(doc.metadata["source"], str)
    
    def test_different_search_types(self) -> None:
        """
        测试不同搜索类型的比较
        
        Args:
            None
            
        Returns:
            None
        """
        query = "人工智能技术"
        
        # 相似性搜索
        similarity_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        similarity_docs = similarity_retriever.invoke(query)
        
        # MMR搜索
        mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.5}
        )
        mmr_docs = mmr_retriever.invoke(query)
        
        # 验证结果
        self.assertIsInstance(similarity_docs, list)
        self.assertIsInstance(mmr_docs, list)
        
        # 两种搜索都应该有结果
        self.assertGreater(len(similarity_docs), 0)
        self.assertGreater(len(mmr_docs), 0)
        
        # 结果可能不同（MMR追求多样性）
        similarity_contents = {doc.page_content for doc in similarity_docs}
        mmr_contents = {doc.page_content for doc in mmr_docs}
        
        # 记录差异（用于分析，不是断言）
        print(f"相似性搜索结果数: {len(similarity_docs)}")
        print(f"MMR搜索结果数: {len(mmr_docs)}")


class TestRetrieverEdgeCases(unittest.TestCase):
    """
    检索器边界情况测试类
    """
    
    def setUp(self) -> None:
        """
        测试前的初始化设置
        """
        self.config = apis["local"]
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"]
        )
        
        # 创建小规模测试数据
        self.small_docs = [
            Document(page_content="测试文档1", metadata={"id": 1}),
            Document(page_content="测试文档2", metadata={"id": 2})
        ]
        self.small_vectorstore = FAISS.from_documents(self.small_docs, self.embeddings)
    
    def test_empty_query(self) -> None:
        """
        测试空查询处理
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = self.small_vectorstore.as_retriever()
        
        # 测试空字符串
        docs = retriever.invoke("")
        self.assertIsInstance(docs, list)
        
        # 测试空白字符
        docs_whitespace = retriever.invoke("   ")
        self.assertIsInstance(docs_whitespace, list)
    
    def test_large_k_value(self) -> None:
        """
        测试k值大于文档数量的情况
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = self.small_vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke("测试")
        
        # 返回的文档数不应超过实际文档数
        self.assertLessEqual(len(docs), len(self.small_docs))
    
    def test_very_high_threshold(self) -> None:
        """
        测试极高阈值的情况
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = self.small_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.99}
        )
        
        # 极高阈值可能没有结果
        docs = retriever.invoke("完全不相关的查询内容xyz123")
        self.assertIsInstance(docs, list)
        # 不强制要求有结果，因为阈值很高


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2) 