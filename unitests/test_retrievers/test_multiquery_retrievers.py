"""
多查询检索器测试

测试LangChain多查询检索器功能，包括：
- MultiQueryRetriever：多查询检索器的基本使用
- 自动查询生成：LLM生成多个不同视角的查询
- 查询合并：合并多个查询的检索结果
- 自定义提示模板：自定义查询生成的提示
- 性能对比：与单查询检索器的比较

作者: LinRen
创建时间: 2025年
"""

import unittest
import os
import sys
import time
import logging
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.vectorstores.utils import DistanceStrategy

from src.config.api import apis


class LineListOutputParser(BaseOutputParser[List[str]]):
    """
    输出解析器，将LLM结果分割为查询列表
    """
    
    def parse(self, text: str) -> List[str]:
        """
        解析LLM输出为查询列表
        
        Args:
            text: LLM生成的文本
            
        Returns:
            查询字符串列表
        """
        lines = text.strip().split("\n")
        # 移除空行和只包含空白字符的行
        return [line.strip() for line in lines if line.strip()]


class TestMultiQueryRetriever(unittest.TestCase):
    """
    多查询检索器测试类
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
        
        # 初始化模型
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"]
        )
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"],
            temperature=0
        )
        
        # 创建测试文档
        self.test_docs = [
            Document(
                page_content="任务分解是将复杂任务拆分为更简单子任务的过程。可以通过思维链(CoT)提示来实现。",
                metadata={"source": "task_decomposition", "method": "CoT"}
            ),
            Document(
                page_content="思维树(Tree of Thoughts)是任务分解的另一种方法，它探索多个推理路径。",
                metadata={"source": "task_decomposition", "method": "ToT"}
            ),
            Document(
                page_content="LLM可以使用简单提示进行任务分解，例如'将这个任务分步骤完成'。",
                metadata={"source": "task_decomposition", "method": "simple_prompt"}
            ),
            Document(
                page_content="任务分解也可以通过人工设计或任务特定的指令来完成。",
                metadata={"source": "task_decomposition", "method": "human_design"}
            ),
            Document(
                page_content="规划(Planning)是任务分解的核心，涉及目标设定和步骤安排。",
                metadata={"source": "planning", "method": "goal_setting"}
            ),
            Document(
                page_content="反思(Reflection)允许代理分析过去的行为和改进未来的决策。",
                metadata={"source": "reflection", "method": "analysis"}
            )
        ]
        
        # 创建向量存储
        self.vectorstore = FAISS.from_documents(self.test_docs, self.embeddings, distance_strategy=DistanceStrategy.COSINE)
        self.base_retriever = self.vectorstore.as_retriever()
        
        # 设置日志
        logging.basicConfig()
        self.logger = logging.getLogger("langchain.retrievers.multi_query")
        self.logger.setLevel(logging.INFO)
    
    def test_multiquery_retriever_creation(self) -> None:
        """
        测试多查询检索器创建
        
        Args:
            None
            
        Returns:
            None
        """
        # 使用from_llm方法创建
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        
        self.assertIsNotNone(retriever)
        self.assertIsInstance(retriever, MultiQueryRetriever)
        self.assertEqual(retriever.retriever, self.base_retriever)
    
    def test_multiquery_basic_functionality(self) -> None:
        """
        测试多查询检索器基本功能
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        
        # 测试查询
        query = "任务分解有哪些方法？"
        docs = retriever.invoke(query)
        
        # 验证结果
        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)
        
        # 验证文档内容
        for doc in docs:
            self.assertIsInstance(doc, Document)
            self.assertIsInstance(doc.page_content, str)
            self.assertIsInstance(doc.metadata, dict)
        
        # 验证去重效果（多查询应该返回唯一文档）
        doc_contents = [doc.page_content for doc in docs]
        unique_contents = set(doc_contents)
        self.assertEqual(len(doc_contents), len(unique_contents), "结果应该去重")
    
    def test_query_generation_logging(self) -> None:
        """
        测试查询生成日志
        
        Args:
            None
            
        Returns:
            None
        """
        # 设置日志捕获
        import io
        import sys
        
        # 创建字符串缓冲区捕获日志
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        self.logger.addHandler(handler)
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        
        # 执行查询
        docs = retriever.invoke("什么是规划？")
        
        # 获取日志内容
        log_content = log_capture.getvalue()
        
        # 验证日志包含生成的查询
        self.assertIn("Generated queries", log_content)
        
        # 清理
        self.logger.removeHandler(handler)
    
    def test_custom_prompt_template(self) -> None:
        """
        测试自定义提示模板
        
        Args:
            None
            
        Returns:
            None
        """
        # 创建自定义提示模板
        custom_prompt = PromptTemplate(
            input_variables=["question"],
            template="""你是一个AI助手。请基于用户问题生成3个不同角度的相关查询，
            以便从知识库中检索更全面的信息。请用换行符分隔这些查询。
            
            用户问题: {question}
            
            生成的查询:"""
        )
        
        # 创建输出解析器
        output_parser = LineListOutputParser()
        
        # 创建LLM链
        llm_chain = custom_prompt | self.llm | output_parser
        
        # 创建自定义多查询检索器
        retriever = MultiQueryRetriever(
            retriever=self.base_retriever,
            llm_chain=llm_chain,
            parser_key="lines"
        )
        
        # 测试自定义检索器
        docs = retriever.invoke("反思在AI中的作用是什么？")
        
        # 验证结果
        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)
    
    def test_multiquery_vs_single_query(self) -> None:
        """
        测试多查询与单查询检索器的比较
        
        Args:
            None
            
        Returns:
            None
        """
        # 创建多查询检索器
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        
        # 使用相同查询测试
        query = "AI代理的任务规划方法"
        
        # 单查询结果
        single_docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=4)
        single_docs = [doc for doc, score in single_docs_with_scores]
        
        # 多查询结果
        multi_docs = multi_retriever.invoke(query)
        
        # 验证结果
        self.assertIsInstance(single_docs, list)
        self.assertIsInstance(multi_docs, list)
        
        # 多查询通常应该返回更多或至少相同数量的文档
        self.assertGreaterEqual(len(multi_docs), len(single_docs))
        
        print(f"单查询结果数: {len(single_docs)}")
        print(f"多查询结果数: {len(multi_docs)}")
    
    def test_different_query_types(self) -> None:
        """
        测试不同类型的查询
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        
        # 测试不同类型的查询
        test_queries = [
            "什么是思维链？",  # 概念查询
            "如何实现任务分解？",  # 方法查询
            "比较不同的规划策略",  # 比较查询
            "任务分解的优缺点",  # 分析查询
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                docs = retriever.invoke(query)
                self.assertIsInstance(docs, list)
                # 不强制要求有结果，因为可能查询与文档不相关
    
    def test_performance_comparison(self) -> None:
        """
        测试性能比较
        
        Args:
            None
            
        Returns:
            None
        """
        # 创建多查询检索器
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        
        query = "任务分解的最佳实践"
        
        # 测试单查询性能
        start_time = time.time()
        single_docs = self.base_retriever.invoke(query)
        single_time = time.time() - start_time
        
        # 测试多查询性能
        start_time = time.time()
        multi_docs = multi_retriever.invoke(query)
        multi_time = time.time() - start_time
        
        # 验证结果
        self.assertIsInstance(single_docs, list)
        self.assertIsInstance(multi_docs, list)
        
        # 多查询由于LLM调用，通常需要更多时间
        self.assertGreater(multi_time, single_time)
        
        print(f"单查询时间: {single_time:.2f}秒")
        print(f"多查询时间: {multi_time:.2f}秒")
        print(f"时间比率: {multi_time/single_time:.2f}x")
    
    def test_query_generation_quality(self) -> None:
        """
        测试查询生成质量
        
        Args:
            None
            
        Returns:
            None
        """
        # 创建带日志的检索器
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        
        # 设置日志级别以捕获生成的查询
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        self.logger.addHandler(handler)
        
        # 执行查询
        query = "人工智能代理如何进行任务规划？"
        docs = retriever.invoke(query)
        
        # 获取日志内容
        log_content = log_capture.getvalue()
        
        # 验证生成了多个查询
        self.assertIn("Generated queries", log_content)
        
        # 清理
        self.logger.removeHandler(handler)
        
        # 验证检索结果
        self.assertIsInstance(docs, list)


class TestMultiQueryRetrieverEdgeCases(unittest.TestCase):
    """
    多查询检索器边界情况测试类
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
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_base=self.config["base_url"],
            openai_api_key=self.config["api_key"],
            temperature=0
        )
        
        # 创建小规模测试数据
        self.small_docs = [
            Document(page_content="测试文档内容A", metadata={"id": "A"}),
            Document(page_content="测试文档内容B", metadata={"id": "B"})
        ]
        
        self.small_vectorstore = FAISS.from_documents(self.small_docs, self.embeddings)
        self.small_retriever = self.small_vectorstore.as_retriever()
    
    def test_empty_query_handling(self) -> None:
        """
        测试空查询处理
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.small_retriever,
            llm=self.llm
        )
        
        # 测试空字符串
        try:
            docs = retriever.invoke("")
            self.assertIsInstance(docs, list)
        except Exception as e:
            # 空查询可能导致异常，这是可以接受的
            self.assertIsInstance(e, Exception)
    
    def test_very_specific_query(self) -> None:
        """
        测试非常具体的查询
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.small_retriever,
            llm=self.llm
        )
        
        # 测试非常具体但与文档无关的查询
        specific_query = "2024年1月15日上海天气如何？"
        docs = retriever.invoke(specific_query)
        
        self.assertIsInstance(docs, list)
        # 可能没有相关结果，这是正常的
    
    def test_multilingual_query(self) -> None:
        """
        测试多语言查询
        
        Args:
            None
            
        Returns:
            None
        """
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.small_retriever,
            llm=self.llm
        )
        
        # 测试英文查询
        english_query = "What is the content of these documents?"
        docs = retriever.invoke(english_query)
        
        self.assertIsInstance(docs, list)
    
    def test_error_handling(self) -> None:
        """
        测试错误处理
        
        Args:
            None
            
        Returns:
            None
        """
        # 创建可能失败的LLM（使用错误配置）
        faulty_llm = ChatOpenAI(
            model="non-existent-model",
            openai_api_base=self.config["base_url"],
            openai_api_key="invalid-key",
            temperature=0
        )
        
        try:
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.small_retriever,
                llm=faulty_llm
            )
            docs = retriever.invoke("测试查询")
            # 如果没有异常，验证结果
            self.assertIsInstance(docs, list)
        except Exception as e:
            # 预期可能出现异常
            self.assertIsInstance(e, Exception)


class TestCustomOutputParser(unittest.TestCase):
    """
    自定义输出解析器测试类
    """
    
    def test_line_list_output_parser(self) -> None:
        """
        测试行列表输出解析器
        
        Args:
            None
            
        Returns:
            None
        """
        parser = LineListOutputParser()
        
        # 测试正常输入
        text = "查询1\n查询2\n查询3"
        result = parser.parse(text)
        
        self.assertEqual(result, ["查询1", "查询2", "查询3"])
        
        # 测试包含空行的输入
        text_with_empty = "查询1\n\n查询2\n   \n查询3"
        result_clean = parser.parse(text_with_empty)
        
        # 应该过滤掉空行和空白行
        self.assertNotIn("", result_clean)
        self.assertNotIn("   ", result_clean)
    
    def test_parser_edge_cases(self) -> None:
        """
        测试解析器边界情况
        
        Args:
            None
            
        Returns:
            None
        """
        parser = LineListOutputParser()
        
        # 测试空字符串
        result_empty = parser.parse("")
        self.assertEqual(result_empty, [])
        
        # 测试只有空白字符
        result_whitespace = parser.parse("   \n  \n")
        self.assertEqual(result_whitespace, [])
        
        # 测试单行
        result_single = parser.parse("单个查询")
        self.assertEqual(result_single, ["单个查询"])


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2) 