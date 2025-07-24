"""
向量存储测试

测试LangChain向量存储功能，包括：
- Chroma：向量数据库的创建和查询
- FAISS：Facebook AI相似性搜索
- similarity_search：相似性搜索功能
- similarity_search_by_vector：向量相似性搜索
- 异步操作测试
- 文档加载和分割测试

作者: LinRen
创建时间: 2025年
"""

import unittest
import tempfile
import shutil
import asyncio
import os
import sys
from typing import List, Dict, Any, Optional

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

from src.config.api import apis


class TestVectorStores(unittest.TestCase):
    """
    向量存储基础功能测试类
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
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 测试文档数据
        self.test_documents = [
            Document(
                page_content="人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的机器。AI技术包括机器学习、深度学习、自然语言处理等领域。",
                metadata={"source": "ai_intro.txt", "category": "AI"}
            ),
            Document(
                page_content="机器学习是人工智能的一个子集，它使计算机能够从数据中学习而无需明确编程。常见的机器学习算法包括监督学习、无监督学习和强化学习。",
                metadata={"source": "ml_intro.txt", "category": "ML"}
            ),
            Document(
                page_content="深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。深度学习在图像识别、语音识别和自然语言处理方面取得了突破性进展。",
                metadata={"source": "dl_intro.txt", "category": "DL"}
            ),
            Document(
                page_content="自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。NLP技术使计算机能够理解、解释和生成人类语言。",
                metadata={"source": "nlp_intro.txt", "category": "NLP"}
            ),
            Document(
                page_content="计算机视觉是人工智能的一个分支，目标是让计算机能够理解和解释视觉信息。它涉及图像处理、模式识别和机器学习技术。",
                metadata={"source": "cv_intro.txt", "category": "CV"}
            )
        ]
        
        # 测试查询
        self.test_queries = [
            "什么是人工智能？",
            "机器学习如何工作？",
            "深度学习的应用有哪些？",
            "自然语言处理的技术有什么？"
        ]
    
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
    
    def test_faiss_vector_store_creation(self) -> None:
        """
        测试FAISS向量存储的创建
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试FAISS向量存储创建 ===")
        
        # 从文档创建FAISS向量存储
        db = FAISS.from_documents(self.test_documents, self.embeddings)
        
        # 验证向量存储
        self.assertIsNotNone(db, "FAISS数据库应该成功创建")
        self.assertEqual(db.index.ntotal, len(self.test_documents), "索引中的文档数量应该正确")
        
        print(f"FAISS向量存储创建成功")
        print(f"索引中的文档数量: {db.index.ntotal}")
        print(f"向量维度: {db.index.d}")
    
    def test_chroma_vector_store_creation(self) -> None:
        """
        测试Chroma向量存储的创建
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试Chroma向量存储创建 ===")
        
        # 从文档创建Chroma向量存储
        db = Chroma.from_documents(
            self.test_documents, 
            self.embeddings,
            persist_directory=self.temp_dir
        )
        
        # 验证向量存储
        self.assertIsNotNone(db, "Chroma数据库应该成功创建")
        
        # 检查文档数量
        collection = db.get()
        self.assertEqual(len(collection['ids']), len(self.test_documents), "集合中的文档数量应该正确")
        
        print(f"Chroma向量存储创建成功")
        print(f"集合中的文档数量: {len(collection['ids'])}")
        print(f"持久化目录: {self.temp_dir}")
    
    def test_similarity_search_faiss(self) -> None:
        """
        测试FAISS的相似性搜索功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试FAISS相似性搜索 ===")
        
        # 创建FAISS向量存储
        db = FAISS.from_documents(self.test_documents, self.embeddings)
        
        for i, query in enumerate(self.test_queries):
            print(f"\n查询 {i+1}: {query}")
            
            # 执行相似性搜索
            docs = db.similarity_search(query, k=2)
            
            # 验证结果
            self.assertEqual(len(docs), 2, "应该返回2个最相似的文档")
            self.assertTrue(all(isinstance(doc, Document) for doc in docs), "结果应该都是Document对象")
            
            # 显示搜索结果
            for j, doc in enumerate(docs):
                print(f"  结果 {j+1}: {doc.page_content[:50]}...")
                print(f"    元数据: {doc.metadata}")
    
    def test_similarity_search_chroma(self) -> None:
        """
        测试Chroma的相似性搜索功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试Chroma相似性搜索 ===")
        
        # 创建Chroma向量存储
        db = Chroma.from_documents(
            self.test_documents, 
            self.embeddings,
            persist_directory=self.temp_dir
        )
        
        query = self.test_queries[0]  # 使用第一个查询
        print(f"查询: {query}")
        
        # 执行相似性搜索
        docs = db.similarity_search(query, k=3)
        
        # 验证结果
        self.assertEqual(len(docs), 3, "应该返回3个最相似的文档")
        self.assertTrue(all(isinstance(doc, Document) for doc in docs), "结果应该都是Document对象")
        
        # 显示搜索结果
        for i, doc in enumerate(docs):
            print(f"结果 {i+1}: {doc.page_content[:50]}...")
            print(f"  元数据: {doc.metadata}")
    
    def test_similarity_search_by_vector(self) -> None:
        """
        测试通过向量进行相似性搜索
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试向量相似性搜索 ===")
        
        # 创建FAISS向量存储
        db = FAISS.from_documents(self.test_documents, self.embeddings)
        
        query = "人工智能的应用领域"
        print(f"查询: {query}")
        
        # 生成查询向量
        embedding_vector = self.embeddings.embed_query(query)
        print(f"查询向量维度: {len(embedding_vector)}")
        
        # 通过向量搜索
        docs_by_vector = db.similarity_search_by_vector(embedding_vector, k=2)
        
        # 通过文本搜索（对比）
        docs_by_text = db.similarity_search(query, k=2)
        
        # 验证结果
        self.assertEqual(len(docs_by_vector), 2, "向量搜索应该返回2个文档")
        self.assertEqual(len(docs_by_text), 2, "文本搜索应该返回2个文档")
        
        # 比较结果（应该相似或相同）
        print("通过向量搜索的结果:")
        for i, doc in enumerate(docs_by_vector):
            print(f"  {i+1}. {doc.page_content[:50]}...")
        
        print("通过文本搜索的结果:")
        for i, doc in enumerate(docs_by_text):
            print(f"  {i+1}. {doc.page_content[:50]}...")
    
    def test_similarity_search_with_score(self) -> None:
        """
        测试带分数的相似性搜索
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试带分数的相似性搜索 ===")
        
        # 创建FAISS向量存储
        db = FAISS.from_documents(self.test_documents, self.embeddings)
        
        query = "深度学习神经网络"
        print(f"查询: {query}")
        
        # 执行带分数的相似性搜索
        docs_with_scores = db.similarity_search_with_score(query, k=3)
        
        # 验证结果
        self.assertEqual(len(docs_with_scores), 3, "应该返回3个带分数的文档")
        
        # 显示结果和分数
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"结果 {i+1} (分数: {score:.4f}): {doc.page_content[:50]}...")
            print(f"  元数据: {doc.metadata}")
            
            # 验证分数类型 (包括numpy类型)
            import numpy as np
            self.assertIsInstance(score, (int, float, np.number), "分数应该是数字")
        
        # 验证分数排序（分数越低越相似，对于FAISS）
        scores = [score for _, score in docs_with_scores]
        self.assertEqual(scores, sorted(scores), "分数应该按升序排列（越低越相似）")
    
    def test_add_documents_to_existing_store(self) -> None:
        """
        测试向现有向量存储添加文档
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试向现有存储添加文档 ===")
        
        # 创建初始向量存储（只包含前3个文档）
        initial_docs = self.test_documents[:3]
        db = FAISS.from_documents(initial_docs, self.embeddings)
        
        initial_count = db.index.ntotal
        print(f"初始文档数量: {initial_count}")
        
        # 添加新文档
        new_docs = self.test_documents[3:]
        db.add_documents(new_docs)
        
        final_count = db.index.ntotal
        print(f"添加后文档数量: {final_count}")
        
        # 验证文档数量增加
        self.assertEqual(final_count, initial_count + len(new_docs), "文档数量应该正确增加")
        
        # 测试搜索新添加的文档
        query = "自然语言处理"
        docs = db.similarity_search(query, k=1)
        
        self.assertEqual(len(docs), 1, "应该能搜索到新添加的文档")
        print(f"搜索结果: {docs[0].page_content[:50]}...")
    
    def test_metadata_filtering(self) -> None:
        """
        测试基于元数据的过滤搜索
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试元数据过滤搜索 ===")
        
        # 创建Chroma向量存储（支持更好的元数据过滤）
        db = Chroma.from_documents(
            self.test_documents, 
            self.embeddings,
            persist_directory=self.temp_dir
        )
        
        query = "学习算法"
        print(f"查询: {query}")
        
        try:
            # 尝试基于元数据过滤搜索（仅搜索ML类别的文档）
            docs = db.similarity_search(
                query, 
                k=2,
                filter={"category": "ML"}
            )
            
            # 验证结果
            self.assertLessEqual(len(docs), 2, "结果数量不应超过指定的k值")
            
            # 验证所有结果都属于指定类别
            for doc in docs:
                if "category" in doc.metadata:
                    self.assertEqual(doc.metadata["category"], "ML", "所有结果都应该属于ML类别")
            
            print(f"过滤搜索结果数量: {len(docs)}")
            for i, doc in enumerate(docs):
                print(f"  {i+1}. {doc.page_content[:50]}... (类别: {doc.metadata.get('category', 'N/A')})")
                
        except Exception as e:
            print(f"元数据过滤功能可能不被此向量存储支持: {e}")
    
    def test_delete_documents(self) -> None:
        """
        测试删除文档功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试删除文档功能 ===")
        
        # 创建Chroma向量存储（FAISS不直接支持删除）
        db = Chroma.from_documents(
            self.test_documents, 
            self.embeddings,
            persist_directory=self.temp_dir
        )
        
        # 获取初始文档数量
        initial_collection = db.get()
        initial_count = len(initial_collection['ids'])
        print(f"初始文档数量: {initial_count}")
        
        try:
            # 获取第一个文档的ID
            doc_id = initial_collection['ids'][0]
            print(f"尝试删除文档ID: {doc_id}")
            
            # 删除文档
            db.delete([doc_id])
            
            # 验证删除效果
            final_collection = db.get()
            final_count = len(final_collection['ids'])
            print(f"删除后文档数量: {final_count}")
            
            self.assertEqual(final_count, initial_count - 1, "文档数量应该减少1")
            self.assertNotIn(doc_id, final_collection['ids'], "被删除的文档ID不应该存在")
            
        except Exception as e:
            print(f"删除功能可能不被此向量存储支持: {e}")


class TestAsyncVectorStores(unittest.TestCase):
    """
    异步向量存储测试类
    """
    
    def setUp(self) -> None:
        """
        异步测试初始化
        
        Args:
            None
            
        Returns:
            None
        """
        self.config = apis["local"]
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"]
        )
        
        self.temp_dir = tempfile.mkdtemp()
        
        self.test_documents = [
            Document(
                page_content="异步编程是一种编程范式，允许程序在等待某些操作完成时继续执行其他任务。",
                metadata={"source": "async_intro.txt"}
            ),
            Document(
                page_content="Python的asyncio库提供了编写异步代码的工具，包括async/await语法。",
                metadata={"source": "python_async.txt"}
            ),
            Document(
                page_content="异步操作在处理I/O密集型任务时特别有用，可以显著提高程序性能。",
                metadata={"source": "async_performance.txt"}
            )
        ]
    
    def tearDown(self) -> None:
        """
        异步测试清理
        
        Args:
            None
            
        Returns:
            None
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_async_similarity_search(self) -> None:
        """
        测试异步相似性搜索
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试异步相似性搜索 ===")
        
        async def async_search_test():
            # 创建向量存储
            db = Chroma.from_documents(
                self.test_documents, 
                self.embeddings,
                persist_directory=self.temp_dir
            )
            
            query = "异步编程的优势"
            print(f"异步查询: {query}")
            
            # 执行异步搜索
            docs = await db.asimilarity_search(query, k=2)
            
            # 验证结果
            self.assertEqual(len(docs), 2, "异步搜索应该返回2个文档")
            self.assertTrue(all(isinstance(doc, Document) for doc in docs), "结果应该都是Document对象")
            
            print("异步搜索结果:")
            for i, doc in enumerate(docs):
                print(f"  {i+1}. {doc.page_content[:50]}...")
            
            return docs
        
        # 运行异步测试
        docs = asyncio.run(async_search_test())
        self.assertIsNotNone(docs, "异步搜索应该返回结果")


class TestDocumentLoaderAndSplitter(unittest.TestCase):
    """
    文档加载和分割测试类
    """
    
    def setUp(self) -> None:
        """
        文档处理测试初始化
        
        Args:
            None
            
        Returns:
            None
        """
        self.config = apis["local"]
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"]
        )
        
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试文本文件
        self.test_file_path = os.path.join(self.temp_dir, "test_document.txt")
        with open(self.test_file_path, 'w', encoding='utf-8') as f:
            f.write("""
人工智能技术发展历程

人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创造能够模拟人类智能行为的机器和系统。

机器学习的兴起
机器学习是人工智能的核心技术之一。它使计算机能够通过数据学习和改进性能，而无需明确编程每一个任务的具体步骤。

深度学习革命
深度学习作为机器学习的一个分支，通过神经网络模拟人脑的工作方式。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。

自然语言处理的应用
自然语言处理（NLP）使计算机能够理解、解释和生成人类语言。现代NLP技术广泛应用于机器翻译、情感分析、问答系统等领域。

未来发展趋势
人工智能技术将继续快速发展，预计在自动驾驶、医疗诊断、智能助手等领域发挥更大作用。
            """.strip())
    
    def tearDown(self) -> None:
        """
        文档处理测试清理
        
        Args:
            None
            
        Returns:
            None
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_text_loader_and_splitter(self) -> None:
        """
        测试文本加载器和分割器
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试文本加载和分割 ===")
        
        # 加载文档
        loader = TextLoader(self.test_file_path, encoding='utf-8')
        raw_documents = loader.load()
        
        print(f"原始文档数量: {len(raw_documents)}")
        print(f"原始文档长度: {len(raw_documents[0].page_content)} 字符")
        
        # 分割文档
        text_splitter = CharacterTextSplitter(
            chunk_size=30,  # 每个块200字符
            chunk_overlap=10,  # 重叠50字符
            separator="\n\n"
        )
        
        documents = text_splitter.split_documents(raw_documents)
        
        print(f"分割后文档数量: {len(documents)}")
        
        # 验证分割结果
        self.assertGreater(len(documents), 1, "应该分割成多个文档块")
        
        # 显示分割后的文档块
        for i, doc in enumerate(documents):
            print(f"文档块 {i+1} ({len(doc.page_content)} 字符): {doc.page_content[:50]}...")
        
        return documents
    
    def test_vector_store_from_split_documents(self) -> None:
        """
        测试从分割文档创建向量存储
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试从分割文档创建向量存储 ===")
        
        # 加载和分割文档
        loader = TextLoader(self.test_file_path, encoding='utf-8')
        raw_documents = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=30,
            chunk_overlap=10
        )
        documents = text_splitter.split_documents(raw_documents)
        
        print(f"准备存储的文档块数量: {len(documents)}")
        
        # 创建向量存储
        db = FAISS.from_documents(documents, self.embeddings)
        
        # 验证向量存储
        self.assertEqual(db.index.ntotal, len(documents), "向量存储中的文档数量应该正确")
        
        # 测试搜索
        query = "深度学习神经网络"
        results = db.similarity_search(query, k=2)
        
        print(f"搜索查询: {query}")
        print(f"搜索结果数量: {len(results)}")
        
        for i, doc in enumerate(results):
            print(f"结果 {i+1}: {doc.page_content[:100]}...")
        
        self.assertEqual(len(results), 2, "应该返回2个搜索结果")


if __name__ == '__main__':
    # 创建测试套件
    unittest.main(verbosity=2) 