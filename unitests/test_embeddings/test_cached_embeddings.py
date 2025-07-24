"""
嵌入模型缓存测试

测试LangChain嵌入模型的缓存功能，包括：
- CacheBackedEmbeddings：缓存支持的嵌入器
- LocalFileStore：本地文件存储缓存
- InMemoryByteStore：内存字节存储缓存
- 与向量存储的集成使用
- 性能对比测试

作者: LinRen
创建时间: 2025年
"""

import unittest
import tempfile
import shutil
import time
import os
import sys
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore, InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

from src.config.api import apis


class TestCachedEmbeddings(unittest.TestCase):
    """
    测试缓存嵌入功能的测试类
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
        
        # 初始化基础嵌入模型
        self.underlying_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"]
        )
        
        # 创建临时目录用于测试文件存储
        self.temp_dir = tempfile.mkdtemp()
        
        # 测试文档
        self.test_documents = [
            "这是第一个测试文档，包含一些基本信息。",
            "第二个文档讨论了人工智能的发展。",
            "第三个文档涉及机器学习和深度学习技术。",
            "最后一个文档描述了自然语言处理的应用。"
        ]
        
        # 测试查询
        self.test_query = "人工智能的应用有哪些？"
    
    def tearDown(self) -> None:
        """
        测试后的清理工作
        
        Args:
            None
            
        Returns:
            None
        """
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_local_file_store_cache(self) -> None:
        """
        测试本地文件存储缓存功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试本地文件存储缓存 ===")
        
        # 创建本地文件存储
        store = LocalFileStore(self.temp_dir)
        
        # 创建缓存嵌入器
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.underlying_embeddings, 
            store, 
            namespace=self.underlying_embeddings.model
        )
        
        # 检查初始状态 - 缓存应该为空
        initial_keys = list(store.yield_keys())
        self.assertEqual(len(initial_keys), 0, "初始缓存应该为空")
        print(f"初始缓存键数量: {len(initial_keys)}")
        
        # 第一次嵌入 - 记录时间
        start_time = time.time()
        embeddings_1 = cached_embedder.embed_documents(self.test_documents)
        first_run_time = time.time() - start_time
        
        # 检查结果
        self.assertEqual(len(embeddings_1), len(self.test_documents))
        self.assertTrue(all(isinstance(emb, list) for emb in embeddings_1))
        print(f"第一次嵌入耗时: {first_run_time:.3f}秒")
        
        # 检查缓存中的键
        cached_keys = list(store.yield_keys())
        self.assertEqual(len(cached_keys), len(self.test_documents))
        print(f"缓存中的键数量: {len(cached_keys)}")
        
        # 第二次嵌入相同文档 - 应该使用缓存
        start_time = time.time()
        embeddings_2 = cached_embedder.embed_documents(self.test_documents)
        second_run_time = time.time() - start_time
        
        # 验证结果一致性
        self.assertEqual(len(embeddings_2), len(embeddings_1))
        for emb1, emb2 in zip(embeddings_1, embeddings_2):
            self.assertEqual(emb1, emb2, "缓存的嵌入结果应该完全一致")
        
        print(f"第二次嵌入耗时: {second_run_time:.3f}秒")
        print(f"性能提升: {first_run_time / second_run_time:.2f}倍")
        
        # 缓存应该显著提高性能
        self.assertLess(second_run_time, first_run_time * 0.5, "缓存应该显著提高性能")
    
    def test_in_memory_store_cache(self) -> None:
        """
        测试内存字节存储缓存功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试内存字节存储缓存 ===")
        
        # 创建内存存储
        store = InMemoryByteStore()
        
        # 创建缓存嵌入器
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.underlying_embeddings,
            store,
            namespace=self.underlying_embeddings.model
        )
        
        # 测试文档嵌入
        embeddings = cached_embedder.embed_documents(self.test_documents[:2])
        
        # 验证结果
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(isinstance(emb, list) for emb in embeddings))
        self.assertTrue(all(len(emb) > 0 for emb in embeddings))
        
        print(f"内存缓存嵌入完成，向量维度: {len(embeddings[0])}")
    
    def test_query_caching(self) -> None:
        """
        测试查询嵌入缓存功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试查询嵌入缓存 ===")
        
        # 创建本地文件存储
        store = LocalFileStore(self.temp_dir)
        
        # 创建支持查询缓存的嵌入器
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.underlying_embeddings,
            store,
            namespace=self.underlying_embeddings.model,
            query_embedding_cache=True  # 启用查询缓存
        )
        
        # 第一次查询嵌入
        start_time = time.time()
        query_embedding_1 = cached_embedder.embed_query(self.test_query)
        first_query_time = time.time() - start_time
        
        # 第二次相同查询 - 应该使用缓存
        start_time = time.time()
        query_embedding_2 = cached_embedder.embed_query(self.test_query)
        second_query_time = time.time() - start_time
        
        # 验证结果
        self.assertEqual(query_embedding_1, query_embedding_2, "查询缓存结果应该一致")
        self.assertLess(second_query_time, first_query_time * 0.5, "查询缓存应该提高性能")
        
        print(f"第一次查询耗时: {first_query_time:.3f}秒")
        print(f"第二次查询耗时: {second_query_time:.3f}秒")
        print(f"查询向量维度: {len(query_embedding_1)}")
    
    def test_faiss_integration(self) -> None:
        """
        测试与FAISS向量存储的集成
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试FAISS向量存储集成 ===")
        
        # 创建缓存嵌入器
        store = LocalFileStore(self.temp_dir)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.underlying_embeddings,
            store,
            namespace=self.underlying_embeddings.model
        )
        
        # 创建文档对象
        documents = [Document(page_content=text) for text in self.test_documents]
        
        # 第一次创建向量存储
        start_time = time.time()
        db1 = FAISS.from_documents(documents, cached_embedder)
        first_creation_time = time.time() - start_time
        
        # 第二次创建相同的向量存储 - 应该使用缓存
        start_time = time.time()
        db2 = FAISS.from_documents(documents, cached_embedder)
        second_creation_time = time.time() - start_time
        
        # 验证向量存储功能
        self.assertEqual(db1.index.ntotal, len(documents))
        self.assertEqual(db2.index.ntotal, len(documents))
        
        # 测试相似性搜索
        results1 = db1.similarity_search(self.test_query, k=2)
        results2 = db2.similarity_search(self.test_query, k=2)
        
        self.assertEqual(len(results1), 2)
        self.assertEqual(len(results2), 2)
        
        print(f"第一次创建向量存储耗时: {first_creation_time:.3f}秒")
        print(f"第二次创建向量存储耗时: {second_creation_time:.3f}秒")
        print(f"性能提升: {first_creation_time / second_creation_time:.2f}倍")
        
        # 显示搜索结果
        print("相似性搜索结果:")
        for i, doc in enumerate(results1):
            print(f"  {i+1}. {doc.page_content}")
    
    def test_namespace_isolation(self) -> None:
        """
        测试命名空间隔离功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试命名空间隔离 ===")
        
        store = LocalFileStore(self.temp_dir)
        
        # 创建两个不同命名空间的嵌入器
        embedder1 = CacheBackedEmbeddings.from_bytes_store(
            self.underlying_embeddings,
            store,
            namespace="model1"
        )
        
        embedder2 = CacheBackedEmbeddings.from_bytes_store(
            self.underlying_embeddings,
            store,
            namespace="model2"
        )
        
        # 使用第一个嵌入器
        embeddings1 = embedder1.embed_documents(self.test_documents[:2])
        keys_after_first = list(store.yield_keys())
        
        # 使用第二个嵌入器（相同文档）
        embeddings2 = embedder2.embed_documents(self.test_documents[:2])
        keys_after_second = list(store.yield_keys())
        
        # 验证命名空间隔离
        self.assertEqual(len(keys_after_first), 2)
        self.assertEqual(len(keys_after_second), 4)  # 每个命名空间各2个
        
        # 验证不同命名空间的键是不同的
        first_namespace_keys = [k for k in keys_after_second if "model1" in k]
        second_namespace_keys = [k for k in keys_after_second if "model2" in k]
        
        self.assertEqual(len(first_namespace_keys), 2)
        self.assertEqual(len(second_namespace_keys), 2)
        self.assertEqual(set(first_namespace_keys) & set(second_namespace_keys), set())
        
        print(f"model1命名空间键数量: {len(first_namespace_keys)}")
        print(f"model2命名空间键数量: {len(second_namespace_keys)}")
        print("命名空间隔离验证成功")
    
    def test_batch_processing(self) -> None:
        """
        测试批处理功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试批处理功能 ===")
        
        store = LocalFileStore(self.temp_dir)
        
        # 创建具有批处理大小的缓存嵌入器
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.underlying_embeddings,
            store,
            namespace=self.underlying_embeddings.model,
            batch_size=2  # 设置批处理大小为2
        )
        
        # 准备更多测试文档
        large_document_set = self.test_documents * 3  # 12个文档
        
        # 进行批处理嵌入
        start_time = time.time()
        embeddings = cached_embedder.embed_documents(large_document_set)
        processing_time = time.time() - start_time
        
        # 验证结果
        self.assertEqual(len(embeddings), len(large_document_set))
        self.assertTrue(all(isinstance(emb, list) for emb in embeddings))
        
        print(f"批处理{len(large_document_set)}个文档耗时: {processing_time:.3f}秒")
        print(f"平均每个文档: {processing_time/len(large_document_set):.3f}秒")
        
        # 验证缓存效果
        cached_keys = list(store.yield_keys())
        # 由于有重复文档，缓存键应该等于唯一文档数量
        unique_docs = len(set(large_document_set))
        self.assertEqual(len(cached_keys), unique_docs)
        print(f"缓存的唯一文档数量: {len(cached_keys)}")


class TestEmbeddingPerformance(unittest.TestCase):
    """
    嵌入性能测试类
    """
    
    def setUp(self) -> None:
        """
        性能测试初始化
        
        Args:
            None
            
        Returns:
            None
        """
        self.config = apis["local"]
        self.underlying_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"]
        )
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建大量测试文档
        self.performance_documents = [
            f"这是第{i}个性能测试文档，包含序号为{i}的内容信息。" 
            for i in range(50)
        ]
    
    def tearDown(self) -> None:
        """
        性能测试清理
        
        Args:
            None
            
        Returns:
            None
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cache_vs_no_cache_performance(self) -> None:
        """
        测试缓存与非缓存的性能对比
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 缓存性能对比测试 ===")
        
        # 无缓存嵌入
        start_time = time.time()
        no_cache_embeddings = self.underlying_embeddings.embed_documents(
            self.performance_documents[:10]
        )
        no_cache_time = time.time() - start_time
        
        # 有缓存嵌入 - 第一次
        store = LocalFileStore(self.temp_dir)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.underlying_embeddings,
            store,
            namespace=self.underlying_embeddings.model
        )
        
        start_time = time.time()
        cached_embeddings_1 = cached_embedder.embed_documents(
            self.performance_documents[:10]
        )
        cached_time_1 = time.time() - start_time
        
        # 有缓存嵌入 - 第二次（使用缓存）
        start_time = time.time()
        cached_embeddings_2 = cached_embedder.embed_documents(
            self.performance_documents[:10]
        )
        cached_time_2 = time.time() - start_time
        
        # 验证结果一致性
        self.assertEqual(len(no_cache_embeddings), len(cached_embeddings_1))
        self.assertEqual(cached_embeddings_1, cached_embeddings_2)
        
        # 性能统计
        print(f"无缓存嵌入耗时: {no_cache_time:.3f}秒")
        print(f"首次缓存嵌入耗时: {cached_time_1:.3f}秒")
        print(f"二次缓存嵌入耗时: {cached_time_2:.3f}秒")
        print(f"缓存性能提升: {no_cache_time / cached_time_2:.2f}倍")
        
        # 缓存应该显著提高性能
        self.assertLess(cached_time_2, no_cache_time * 0.1, "缓存应该大幅提高性能")


if __name__ == '__main__':
    # 创建测试套件
    unittest.main(verbosity=2) 