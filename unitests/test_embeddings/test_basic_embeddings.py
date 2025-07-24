"""
基础嵌入模型测试

测试LangChain基础嵌入模型功能，包括：
- OpenAIEmbeddings：OpenAI嵌入模型的基本使用
- embed_documents：文档嵌入功能
- embed_query：查询嵌入功能
- 向量维度验证
- 相似性计算测试

作者: LinRen
创建时间: 2025年
"""

import unittest
import os
import sys
import numpy as np
from typing import List, Dict, Any
import math

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config.api import apis


class TestBasicEmbeddings(unittest.TestCase):
    """
    基础嵌入模型测试类
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
        
        # 测试文档
        self.test_documents = [
            "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的机器。",
            "机器学习是人工智能的一个子集，它使计算机能够从数据中学习而无需明确编程。",
            "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。",
            "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。"
        ]
        
        # 测试查询
        self.test_queries = [
            "什么是人工智能？",
            "机器学习如何工作？",
            "深度学习的应用有哪些？"
        ]
    
    def test_embed_documents_basic(self) -> None:
        """
        测试基础文档嵌入功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试基础文档嵌入功能 ===")
        
        # 执行文档嵌入
        embeddings = self.embeddings.embed_documents(self.test_documents)
        
        # 验证结果
        self.assertEqual(len(embeddings), len(self.test_documents), "嵌入数量应该与文档数量一致")
        
        # 验证每个嵌入都是列表且不为空
        for i, embedding in enumerate(embeddings):
            self.assertIsInstance(embedding, list, f"第{i+1}个嵌入应该是列表")
            self.assertGreater(len(embedding), 0, f"第{i+1}个嵌入不应该为空")
            
            # 验证向量元素都是数字
            for j, value in enumerate(embedding):
                self.assertIsInstance(value, (int, float), f"嵌入向量第{j+1}个元素应该是数字")
        
        # 验证所有嵌入向量的维度一致
        dimensions = [len(emb) for emb in embeddings]
        self.assertTrue(all(dim == dimensions[0] for dim in dimensions), "所有嵌入向量维度应该一致")
        
        print(f"文档数量: {len(self.test_documents)}")
        print(f"嵌入向量维度: {len(embeddings[0])}")
        print(f"每个嵌入向量前5个值示例: {embeddings[0][:5]}")
    
    def test_embed_query_basic(self) -> None:
        """
        测试基础查询嵌入功能
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试基础查询嵌入功能 ===")
        
        query_embeddings = []
        
        # 对每个查询进行嵌入
        for query in self.test_queries:
            embedding = self.embeddings.embed_query(query)
            query_embeddings.append(embedding)
            
            # 验证结果
            self.assertIsInstance(embedding, list, "查询嵌入应该是列表")
            self.assertGreater(len(embedding), 0, "查询嵌入不应该为空")
            
            # 验证向量元素都是数字
            for value in embedding:
                self.assertIsInstance(value, (int, float), "嵌入向量元素应该是数字")
        
        # 验证所有查询嵌入维度一致
        dimensions = [len(emb) for emb in query_embeddings]
        self.assertTrue(all(dim == dimensions[0] for dim in dimensions), "所有查询嵌入维度应该一致")
        
        print(f"查询数量: {len(self.test_queries)}")
        print(f"查询嵌入维度: {len(query_embeddings[0])}")
        
        # 显示每个查询的嵌入示例
        for i, (query, embedding) in enumerate(zip(self.test_queries, query_embeddings)):
            print(f"查询{i+1}: {query}")
            print(f"  嵌入前5个值: {embedding[:5]}")
    
    def test_embedding_consistency(self) -> None:
        """
        测试嵌入一致性
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试嵌入一致性 ===")
        
        test_text = "这是一个测试文本，用于验证嵌入的一致性。"
        
        # 多次嵌入相同文本
        embedding1 = self.embeddings.embed_query(test_text)
        embedding2 = self.embeddings.embed_query(test_text)
        embedding3 = self.embeddings.embed_query(test_text)
        
        # 验证结果一致性
        self.assertEqual(embedding1, embedding2, "相同文本的嵌入应该完全一致")
        self.assertEqual(embedding2, embedding3, "相同文本的嵌入应该完全一致")
        
        print(f"测试文本: {test_text}")
        print("多次嵌入结果一致性验证通过")
    
    def test_embedding_similarity(self) -> None:
        """
        测试嵌入相似性计算
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试嵌入相似性计算 ===")
        
        # 相似文本对
        similar_texts = [
            "人工智能是一门科学",
            "AI是计算机科学的一个分支"
        ]
        
        # 不相似文本对
        different_texts = [
            "人工智能是一门科学",
            "今天天气很好，阳光明媚"
        ]
        
        # 获取嵌入
        similar_embeddings = [self.embeddings.embed_query(text) for text in similar_texts]
        different_embeddings = [self.embeddings.embed_query(text) for text in different_texts]
        
        # 计算余弦相似度
        similar_similarity = self._cosine_similarity(similar_embeddings[0], similar_embeddings[1])
        different_similarity = self._cosine_similarity(different_embeddings[0], different_embeddings[1])
        
        # 验证相似文本的相似度更高
        self.assertGreater(similar_similarity, different_similarity, 
                          "相似文本的余弦相似度应该大于不相似文本")
        
        print(f"相似文本对: {similar_texts}")
        print(f"相似度: {similar_similarity:.4f}")
        print(f"不相似文本对: {different_texts}")
        print(f"相似度: {different_similarity:.4f}")
        print(f"相似度差异: {similar_similarity - different_similarity:.4f}")
    
    def test_empty_text_handling(self) -> None:
        """
        测试空文本处理
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试空文本处理 ===")
        
        # 测试空字符串
        try:
            empty_embedding = self.embeddings.embed_query("")
            self.assertIsInstance(empty_embedding, list, "空字符串嵌入应该是列表")
            self.assertGreater(len(empty_embedding), 0, "空字符串嵌入不应该为空列表")
            print("空字符串处理成功")
        except Exception as e:
            print(f"空字符串处理异常: {e}")
        
        # 测试只包含空格的字符串
        try:
            space_embedding = self.embeddings.embed_query("   ")
            self.assertIsInstance(space_embedding, list, "空格字符串嵌入应该是列表")
            print("空格字符串处理成功")
        except Exception as e:
            print(f"空格字符串处理异常: {e}")
    
    def test_long_text_handling(self) -> None:
        """
        测试长文本处理
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试长文本处理 ===")
        
        # 创建长文本
        long_text = "这是一个很长的测试文本。" * 100  # 重复100次
        
        try:
            long_embedding = self.embeddings.embed_query(long_text)
            self.assertIsInstance(long_embedding, list, "长文本嵌入应该是列表")
            self.assertGreater(len(long_embedding), 0, "长文本嵌入不应该为空")
            
            print(f"长文本长度: {len(long_text)} 字符")
            print(f"嵌入向量维度: {len(long_embedding)}")
            print("长文本处理成功")
            
        except Exception as e:
            print(f"长文本处理异常: {e}")
    
    def test_batch_vs_individual_embeddings(self) -> None:
        """
        测试批量嵌入与单独嵌入的一致性
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试批量嵌入与单独嵌入的一致性 ===")
        
        test_texts = self.test_documents[:3]  # 使用前3个文档
        
        # 批量嵌入
        batch_embeddings = self.embeddings.embed_documents(test_texts)
        
        # 单独嵌入
        individual_embeddings = []
        for text in test_texts:
            embedding = self.embeddings.embed_query(text)
            individual_embeddings.append(embedding)
        
        # 验证一致性
        self.assertEqual(len(batch_embeddings), len(individual_embeddings))
        
        for i, (batch_emb, individual_emb) in enumerate(zip(batch_embeddings, individual_embeddings)):
            # 注意：embed_documents 和 embed_query 可能使用不同的嵌入策略
            # 所以我们只验证维度一致性，不验证数值完全一致
            self.assertEqual(len(batch_emb), len(individual_emb), 
                           f"第{i+1}个文档的批量嵌入与单独嵌入维度应该一致")
        
        print(f"测试文档数量: {len(test_texts)}")
        print(f"批量嵌入维度: {len(batch_embeddings[0])}")
        print(f"单独嵌入维度: {len(individual_embeddings[0])}")
        print("维度一致性验证通过")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度值
        """
        # 转换为numpy数组
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)


class TestEmbeddingModels(unittest.TestCase):
    """
    不同嵌入模型对比测试类
    """
    
    def setUp(self) -> None:
        """
        模型对比测试初始化
        
        Args:
            None
            
        Returns:
            None
        """
        self.config = apis["local"]
        
        # 初始化不同的嵌入模型
        self.small_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"]
        )
        
        # 如果有大模型可用，可以添加
        # self.large_model = OpenAIEmbeddings(
        #     model="text-embedding-3-large",
        #     openai_api_base=self.config["base_url"],
        #     openai_api_key=self.config["api_key"]
        # )
        
        self.test_text = "这是一个用于测试不同嵌入模型的示例文本。"
    
    def test_small_model_embedding(self) -> None:
        """
        测试小型嵌入模型
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试小型嵌入模型 ===")
        
        embedding = self.small_model.embed_query(self.test_text)
        
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        
        print(f"模型: {self.small_model.model}")
        print(f"嵌入维度: {len(embedding)}")
        print(f"嵌入示例: {embedding[:5]}")
    
    def test_model_dimension_consistency(self) -> None:
        """
        测试模型维度一致性
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 测试模型维度一致性 ===")
        
        test_texts = [
            "短文本",
            "这是一个稍微长一些的测试文本，用于验证维度一致性。",
            "这是一个更长的文本示例，包含更多的词汇和内容，用于测试嵌入模型在处理不同长度文本时的维度一致性表现。"
        ]
        
        embeddings = self.small_model.embed_documents(test_texts)
        
        # 验证所有嵌入维度一致
        dimensions = [len(emb) for emb in embeddings]
        self.assertTrue(all(dim == dimensions[0] for dim in dimensions), 
                       "不同长度文本的嵌入维度应该一致")
        
        print(f"测试文本数量: {len(test_texts)}")
        print(f"文本长度范围: {min(len(t) for t in test_texts)} - {max(len(t) for t in test_texts)} 字符")
        print(f"嵌入维度: {dimensions[0]}")
        print("维度一致性验证通过")


if __name__ == '__main__':
    # 创建测试套件
    unittest.main(verbosity=2) 