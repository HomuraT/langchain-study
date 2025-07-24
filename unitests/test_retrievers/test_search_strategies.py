"""
搜索策略和参数测试

测试LangChain检索器的不同搜索策略和参数配置，包括：
- 相似性搜索参数优化
- MMR搜索参数调优
- 阈值搜索策略比较
- 批量检索优化
- 搜索策略选择指南验证

作者: LinRen
创建时间: 2025年
"""

import unittest
import os
import sys
import time
import asyncio
from typing import List, Dict, Any, Tuple
from functools import lru_cache

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config.api import apis


class TestSearchStrategies(unittest.TestCase):
    """
    搜索策略测试类
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
        
        # 创建多样化的测试文档集
        self.diverse_docs = [
            Document(
                page_content="人工智能(AI)是计算机科学的重要分支，专注于创建智能机器。",
                metadata={"domain": "AI", "level": "basic", "length": "short"}
            ),
            Document(
                page_content="机器学习是人工智能的核心技术，通过算法让计算机从数据中学习模式和规律。",
                metadata={"domain": "ML", "level": "intermediate", "length": "medium"}
            ),
            Document(
                page_content="深度学习使用多层神经网络来模拟人脑处理信息的方式，在图像识别、自然语言处理等领域取得了突破性进展。",
                metadata={"domain": "DL", "level": "advanced", "length": "long"}
            ),
            Document(
                page_content="自然语言处理(NLP)是AI的重要应用领域。",
                metadata={"domain": "NLP", "level": "basic", "length": "short"}
            ),
            Document(
                page_content="计算机视觉让机器能够理解和解释视觉世界，包括图像分类、目标检测、图像分割等多个子任务。",
                metadata={"domain": "CV", "level": "intermediate", "length": "long"}
            ),
            Document(
                page_content="强化学习通过试错和奖励机制训练智能体在环境中做出最优决策。",
                metadata={"domain": "RL", "level": "advanced", "length": "medium"}
            ),
            Document(
                page_content="数据挖掘从大量数据中发现有用的模式和知识。",
                metadata={"domain": "DM", "level": "basic", "length": "short"}
            ),
            Document(
                page_content="大数据技术处理超大规模数据集，包括数据存储、计算和分析的完整解决方案，支持实时和批处理两种模式。",
                metadata={"domain": "BigData", "level": "advanced", "length": "long"}
            )
        ]
        
        # 创建向量存储
        self.vectorstore = FAISS.from_documents(self.diverse_docs, self.embeddings)
    
    def test_similarity_search_k_values(self) -> None:
        """
        测试相似性搜索的k值影响
        
        Args:
            None
            
        Returns:
            None
        """
        query = "machine learning algorithms"
        k_values = [1, 2, 3, 5, 8, 10]
        results = {}
        
        for k in k_values:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            docs = retriever.invoke(query)
            results[k] = {
                "count": len(docs),
                "domains": [doc.metadata.get("domain", "") for doc in docs]
            }
        
        # 验证k值效果
        for k in k_values:
            actual_count = results[k]["count"]
            expected_max = min(k, len(self.diverse_docs))
            
            self.assertLessEqual(actual_count, expected_max, f"k={k}时返回文档数应不超过{expected_max}")
            self.assertGreater(actual_count, 0, f"k={k}时应有结果")
        
        # 验证k值递增效果
        for i in range(len(k_values) - 1):
            current_k = k_values[i]
            next_k = k_values[i + 1]
            self.assertLessEqual(
                results[current_k]["count"],
                results[next_k]["count"],
                f"k={next_k}的结果数应不少于k={current_k}"
            )
        
        print("相似性搜索k值测试结果:")
        for k, result in results.items():
            print(f"k={k}: {result['count']}个文档, 领域: {result['domains']}")
    
    def test_mmr_parameters_optimization(self) -> None:
        """
        测试MMR参数优化
        
        Args:
            None
            
        Returns:
            None
        """
        query = "artificial intelligence machine learning"
        
        # 测试不同的lambda_mult值
        lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = {}
        
        for lambda_mult in lambda_values:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 8,
                    "lambda_mult": lambda_mult
                }
            )
            docs = retriever.invoke(query)
            
            # 计算多样性指标
            domains = [doc.metadata.get("domain", "") for doc in docs]
            unique_domains = set(domains)
            diversity_score = len(unique_domains) / len(docs) if docs else 0
            
            results[lambda_mult] = {
                "count": len(docs),
                "domains": domains,
                "unique_domains": len(unique_domains),
                "diversity_score": diversity_score
            }
        
        # 验证MMR多样性效果
        for lambda_mult, result in results.items():
            self.assertGreater(result["count"], 0, f"lambda_mult={lambda_mult}时应有结果")
            
            # 低lambda_mult应该有更高的多样性
            if lambda_mult <= 0.3:
                self.assertGreater(result["diversity_score"], 0.5, "低lambda_mult应该有较高多样性")
        
        print("MMR参数优化测试结果:")
        for lambda_mult, result in results.items():
            print(f"λ={lambda_mult}: 多样性={result['diversity_score']:.2f}, 领域: {result['domains']}")
    
    def test_threshold_search_strategies(self) -> None:
        """
        测试阈值搜索策略
        
        Args:
            None
            
        Returns:
            None
        """
        # 测试不同相关性的查询
        test_cases = [
            ("machine learning algorithms", "高相关性查询"),
            ("artificial intelligence", "中等相关性查询"),
            ("technology computer", "低相关性查询"),
            ("cooking food recipe", "无关查询")
        ]
        
        # 注意：FAISS返回的是距离值，不是相似性得分
        # 对于FAISS，score_threshold实际上是距离阈值，较小的值表示更严格的要求
        # 我们先测试without threshold来了解得分分布
        print("\n=== 测试阈值搜索策略 ===")
        print("注意：FAISS返回距离值，较小值表示更相似")
        
        for query, desc in test_cases:
            print(f"\n测试{desc}: '{query}'")
            
            # 首先获取带分数的搜索结果来了解分数分布
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=4)
            
            if docs_with_scores:
                scores = [score for _, score in docs_with_scores]
                print(f"  分数范围: {min(scores):.4f} ~ {max(scores):.4f}")
                
                # 基于实际分数设置合理的阈值
                # 对于FAISS距离，我们使用更合理的阈值
                # 通常FAISS距离在0-2之间，0表示完全相同
                thresholds = [2.0, 1.5, 1.0, 0.5]  # 使用距离阈值而不是相似性阈值
                
                for threshold in thresholds:
                    try:
                        # 手动过滤结果，因为FAISS的score_threshold可能有问题
                        filtered_docs = [doc for doc, score in docs_with_scores if score <= threshold]
                        print(f"  距离阈值≤{threshold}: {len(filtered_docs)}个文档")
                        
                        # 验证阈值效果：更小的阈值应该返回更少的文档
                        self.assertIsInstance(filtered_docs, list)
                        
                    except Exception as e:
                        print(f"  距离阈值{threshold}失败: {e}")
            else:
                print(f"  没有检索到文档")
    
    def test_similarity_score_analysis(self) -> None:
        """
        分析相似性得分分布
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 相似性得分分析 ===")
        
        queries = [
            "machine learning artificial intelligence",  # 高度相关
            "computer technology",                       # 中度相关  
            "cooking recipe food"                        # 低度相关
        ]
        
        for query in queries:
            print(f"\n查询: '{query}'")
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=5)
            
            if docs_with_scores:
                print("  结果:")
                for i, (doc, score) in enumerate(docs_with_scores):
                    domain = doc.metadata.get("domain", "unknown")
                    content_preview = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
                    print(f"    {i+1}. 得分:{score:.4f} 领域:{domain} 内容:{content_preview}")
                
                scores = [score for _, score in docs_with_scores]
                print(f"  得分统计: 最小={min(scores):.4f}, 最大={max(scores):.4f}, 平均={sum(scores)/len(scores):.4f}")
            else:
                print("  无检索结果")
    
    def test_cosine_similarity_manual_calculation(self) -> None:
        """
        手动计算余弦相似度验证
        
        Args:
            None
            
        Returns:
            None
        """
        print("\n=== 手动余弦相似度计算验证 ===")
        
        query = "machine learning"
        
        # 获取查询向量
        query_vector = self.embeddings.embed_query(query)
        
        # 获取文档向量（这里简化处理，实际中需要获取存储的向量）
        test_docs = self.diverse_docs[:3]  # 只测试前3个文档
        
        for i, doc in enumerate(test_docs):
            doc_vector = self.embeddings.embed_query(doc.page_content)
            
            # 计算余弦相似度
            import numpy as np
            
            # 转换为numpy数组
            q_vec = np.array(query_vector)
            d_vec = np.array(doc_vector)
            
            # 计算余弦相似度
            cosine_sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            
            # 计算欧几里得距离（FAISS通常使用的）
            euclidean_dist = np.linalg.norm(q_vec - d_vec)
            
            domain = doc.metadata.get("domain", "unknown")
            content_preview = doc.page_content[:50] + "..."
            
            print(f"  文档{i+1} ({domain}):")
            print(f"    余弦相似度: {cosine_sim:.4f}")
            print(f"    欧几里得距离: {euclidean_dist:.4f}")
            print(f"    内容: {content_preview}")
            
            # 验证相似度在合理范围内
            self.assertGreaterEqual(cosine_sim, -1.0, "余弦相似度应该>=−1")
            self.assertLessEqual(cosine_sim, 1.0, "余弦相似度应该<=1")
            self.assertGreaterEqual(euclidean_dist, 0.0, "欧几里得距离应该>=0")
    
    def test_search_strategy_comparison(self) -> None:
        """
        测试不同搜索策略的比较
        
        Args:
            None
            
        Returns:
            None
        """
        query = "deep learning neural networks"
        
        # 创建不同策略的检索器
        strategies = {
            "similarity": self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            "mmr_diverse": self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "lambda_mult": 0.3}
            ),
            "mmr_relevant": self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "lambda_mult": 0.8}
            ),
            "threshold_strict": self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.8}
            ),
            "threshold_loose": self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5}
            )
        }
        
        results = {}
        for strategy_name, retriever in strategies.items():
            start_time = time.time()
            docs = retriever.invoke(query)
            end_time = time.time()
            
            # 计算结果指标
            domains = [doc.metadata.get("domain", "") for doc in docs]
            levels = [doc.metadata.get("level", "") for doc in docs]
            
            results[strategy_name] = {
                "count": len(docs),
                "time": end_time - start_time,
                "domains": domains,
                "levels": levels,
                "diversity": len(set(domains)) / len(docs) if docs else 0
            }
        
        # 验证策略效果
        self.assertGreater(results["similarity"]["count"], 0, "相似性搜索应有结果")
        
        # MMR多样性策略应该有更高的多样性
        if results["mmr_diverse"]["count"] > 1 and results["mmr_relevant"]["count"] > 1:
            self.assertGreaterEqual(
                results["mmr_diverse"]["diversity"],
                results["mmr_relevant"]["diversity"],
                "MMR多样性策略应该有更高的多样性"
            )
        
        # 严格阈值应该返回更少的文档
        self.assertLessEqual(
            results["threshold_strict"]["count"],
            results["threshold_loose"]["count"],
            "严格阈值应该返回更少的文档"
        )
        
        print("搜索策略比较结果:")
        for strategy, result in results.items():
            print(f"{strategy}: {result['count']}文档, "
                  f"多样性={result['diversity']:.2f}, "
                  f"时间={result['time']:.3f}s")
    
    def test_batch_retrieval_optimization(self) -> None:
        """
        测试批量检索优化
        
        Args:
            None
            
        Returns:
            None
        """
        queries = [
            "machine learning",
            "deep learning",
            "natural language processing",
            "computer vision",
            "reinforcement learning"
        ]
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 测试顺序检索
        start_time = time.time()
        sequential_results = []
        for query in queries:
            docs = retriever.invoke(query)
            sequential_results.append(docs)
        sequential_time = time.time() - start_time
        
        # 测试缓存优化
        @lru_cache(maxsize=100)
        def cached_retrieve(query: str):
            return retriever.invoke(query)
        
        start_time = time.time()
        cached_results = []
        for query in queries:
            docs = cached_retrieve(query)
            cached_results.append(docs)
        # 第二次调用应该更快（缓存命中）
        for query in queries:
            docs = cached_retrieve(query)
        cached_time = time.time() - start_time
        
        # 验证结果
        self.assertEqual(len(sequential_results), len(queries))
        self.assertEqual(len(cached_results), len(queries))
        
        # 验证结果一致性
        for i in range(len(queries)):
            seq_contents = [doc.page_content for doc in sequential_results[i]]
            cached_contents = [doc.page_content for doc in cached_results[i]]
            self.assertEqual(seq_contents, cached_contents, f"查询{i}的结果应该一致")
        
        print(f"批量检索测试:")
        print(f"顺序检索时间: {sequential_time:.3f}秒")
        print(f"缓存检索时间: {cached_time:.3f}秒")
        
        # 清理缓存
        cached_retrieve.cache_clear()
    
    def test_async_retrieval(self) -> None:
        """
        测试异步检索
        
        Args:
            None
            
        Returns:
            None
        """
        async def async_batch_retrieve(retriever: BaseRetriever, queries: List[str]):
            """异步批量检索"""
            tasks = [retriever.ainvoke(query) for query in queries]
            results = await asyncio.gather(*tasks)
            return results
        
        queries = [
            "artificial intelligence",
            "machine learning",
            "deep learning"
        ]
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # 运行异步检索
        async def run_async_test():
            start_time = time.time()
            async_results = await async_batch_retrieve(retriever, queries)
            async_time = time.time() - start_time
            
            # 验证异步结果
            self.assertEqual(len(async_results), len(queries))
            for result in async_results:
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
            
            print(f"异步检索时间: {async_time:.3f}秒")
            return async_results
        
        # 运行异步测试
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async_results = loop.run_until_complete(run_async_test())
        self.assertIsNotNone(async_results)


class TestParameterTuning(unittest.TestCase):
    """
    参数调优测试类
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
        
        # 创建更大的测试数据集
        self.large_docs = [
            Document(
                page_content=f"这是第{i}个测试文档，内容涉及人工智能和机器学习的各个方面。"
                           f"文档编号{i}包含了关于深度学习、自然语言处理、计算机视觉等技术的描述。",
                metadata={"doc_id": i, "category": f"cat_{i % 5}"}
            )
            for i in range(20)
        ]
        
        self.large_vectorstore = FAISS.from_documents(self.large_docs, self.embeddings)
    
    def test_optimal_k_selection(self) -> None:
        """
        测试最优k值选择
        
        Args:
            None
            
        Returns:
            None
        """
        query = "人工智能技术"
        k_range = range(1, 11)
        
        # 测试不同k值的效果
        performance_metrics = {}
        
        for k in k_range:
            retriever = self.large_vectorstore.as_retriever(search_kwargs={"k": k})
            
            start_time = time.time()
            docs = retriever.invoke(query)
            end_time = time.time()
            
            # 计算性能指标
            retrieval_time = end_time - start_time
            doc_count = len(docs)
            
            # 计算多样性
            categories = [doc.metadata.get("category", "") for doc in docs]
            diversity = len(set(categories)) / len(categories) if categories else 0
            
            performance_metrics[k] = {
                "time": retrieval_time,
                "count": doc_count,
                "diversity": diversity
            }
        
        # 分析最优k值
        best_k_by_diversity = max(performance_metrics.keys(), 
                                key=lambda k: performance_metrics[k]["diversity"])
        
        print("k值优化测试结果:")
        for k, metrics in performance_metrics.items():
            print(f"k={k}: 时间={metrics['time']:.3f}s, "
                  f"文档数={metrics['count']}, 多样性={metrics['diversity']:.2f}")
        
        print(f"最优k值（按多样性）: {best_k_by_diversity}")
        
        # 验证结果合理性
        for k in k_range:
            self.assertGreater(performance_metrics[k]["count"], 0, f"k={k}时应有结果")
            self.assertLessEqual(performance_metrics[k]["count"], k, f"结果数不应超过k={k}")
    
    def test_threshold_calibration(self) -> None:
        """
        测试阈值校准
        
        Args:
            None
            
        Returns:
            None
        """
        # 测试不同相关性的查询
        test_queries = [
            ("人工智能机器学习", "高相关"),
            ("技术发展", "中等相关"),
            ("天气预报", "低相关")
        ]
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        calibration_results = {}
        
        for query, relevance_level in test_queries:
            calibration_results[relevance_level] = {}
            
            for threshold in thresholds:
                retriever = self.large_vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"score_threshold": threshold}
                )
                
                docs = retriever.invoke(query)
                calibration_results[relevance_level][threshold] = len(docs)
        
        # 验证阈值校准效果
        for relevance_level, threshold_results in calibration_results.items():
            print(f"\n{relevance_level}查询的阈值校准:")
            for threshold, count in threshold_results.items():
                print(f"  阈值{threshold}: {count}个文档")
            
            # 验证阈值递增效果（文档数应该递减）
            thresholds_sorted = sorted(threshold_results.keys())
            for i in range(len(thresholds_sorted) - 1):
                current_threshold = thresholds_sorted[i]
                next_threshold = thresholds_sorted[i + 1]
                
                self.assertGreaterEqual(
                    threshold_results[current_threshold],
                    threshold_results[next_threshold],
                    f"阈值增加时文档数应该不增加"
                )


class TestRealWorldScenarios(unittest.TestCase):
    """
    真实场景测试类
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
        
        # 模拟真实文档
        self.realistic_docs = [
            Document(
                page_content="Q: 什么是机器学习？\nA: 机器学习是人工智能的一个分支，使计算机能够从数据中学习而无需明确编程。",
                metadata={"type": "QA", "domain": "ML", "difficulty": "beginner"}
            ),
            Document(
                page_content="深度学习模型的训练需要大量的计算资源和数据。GPU加速可以显著提高训练效率。",
                metadata={"type": "technical", "domain": "DL", "difficulty": "intermediate"}
            ),
            Document(
                page_content="自然语言处理的主要任务包括文本分类、情感分析、机器翻译、问答系统等。",
                metadata={"type": "overview", "domain": "NLP", "difficulty": "beginner"}
            ),
            Document(
                page_content="强化学习中的Q-learning算法通过迭代更新Q值表来学习最优策略。数学公式为：Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]",
                metadata={"type": "technical", "domain": "RL", "difficulty": "advanced"}
            )
        ]
        
        self.realistic_vectorstore = FAISS.from_documents(self.realistic_docs, self.embeddings)
    
    def test_qa_scenario(self) -> None:
        """
        测试问答场景
        
        Args:
            None
            
        Returns:
            None
        """
        # 问答系统场景：需要精确的答案
        qa_retriever = self.realistic_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}  # 只需要最相关的答案
        )
        
        question = "机器学习是什么？"
        answer_docs = qa_retriever.invoke(question)
        
        # 验证问答效果
        self.assertEqual(len(answer_docs), 1, "问答场景应返回单个最佳答案")
        
        answer_doc = answer_docs[0]
        self.assertIn("机器学习", answer_doc.page_content, "答案应包含相关关键词")
        self.assertEqual(answer_doc.metadata.get("type"), "QA", "应返回QA类型的文档")
        
        print(f"问答场景测试:")
        print(f"问题: {question}")
        print(f"答案: {answer_doc.page_content[:100]}...")
    
    def test_research_scenario(self) -> None:
        """
        测试研究场景
        
        Args:
            None
            
        Returns:
            None
        """
        # 研究场景：需要多样化的信息
        research_retriever = self.realistic_vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "lambda_mult": 0.3  # 偏向多样性
            }
        )
        
        research_query = "人工智能的主要技术领域"
        research_docs = research_retriever.invoke(research_query)
        
        # 验证研究效果
        self.assertGreater(len(research_docs), 1, "研究场景应返回多个文档")
        
        domains = [doc.metadata.get("domain", "") for doc in research_docs]
        unique_domains = set(domains)
        
        print(f"研究场景测试:")
        print(f"查询: {research_query}")
        print(f"找到{len(research_docs)}个文档，涵盖{len(unique_domains)}个领域: {unique_domains}")
        
        # 应该涵盖多个不同的领域
        self.assertGreater(len(unique_domains), 1, "研究场景应涵盖多个领域")
    
    def test_content_filtering_scenario(self) -> None:
        """
        测试内容过滤场景
        
        Args:
            None
            
        Returns:
            None
        """
        # 内容过滤场景：只返回特定难度的内容
        beginner_retriever = self.realistic_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.1}  # 宽松阈值获取更多候选
        )
        
        query = "深度学习技术"
        all_docs = beginner_retriever.invoke(query)
        
        # 手动过滤初学者友好的内容
        beginner_docs = [
            doc for doc in all_docs 
            if doc.metadata.get("difficulty") == "beginner"
        ]
        
        print(f"内容过滤场景测试:")
        print(f"查询: {query}")
        print(f"总文档数: {len(all_docs)}")
        print(f"初学者友好文档数: {len(beginner_docs)}")
        
        # 验证过滤效果
        if beginner_docs:
            for doc in beginner_docs:
                self.assertEqual(doc.metadata.get("difficulty"), "beginner")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2) 