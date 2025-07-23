"""
LangChain Expression Language (LCEL) 并行执行测试

测试LCEL的并行执行和批处理功能：
- 批处理操作 (batch)
- 异步批处理 (abatch)
- 并行性能优化
- 大批量处理

作者: AI Assistant
创建时间: 2025年
"""

import unittest
import asyncio
import time
from typing import List
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.config.api import apis


class TestLCELParallelExecution(unittest.TestCase):
    """LCEL并行执行测试类"""
    
    @classmethod
    def setUpClass(cls) -> None:
        """
        设置测试类的初始配置
        
        输入: 无
        输出: 无
        """
        cls.config = apis["local"]
        cls.model = ChatOpenAI(
            base_url=cls.config["base_url"],
            api_key=cls.config["api_key"],
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=150,
            timeout=30
        )
    
    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        self.simple_prompt = ChatPromptTemplate.from_template("简短回答: {question}")
        self.processing_chain = self.simple_prompt | self.model | StrOutputParser()
        
        # 简单处理函数
        self.add_prefix = lambda x: f"[处理] {x}"
        self.count_words = lambda x: len(str(x).split())
    
    def test_basic_batch_processing(self) -> None:
        """
        测试基本批处理功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试基本批处理功能 ===")
        
        # 创建处理链
        batch_chain = RunnableLambda(self.add_prefix)
        
        # 批处理输入
        batch_inputs = ["输入1", "输入2", "输入3", "输入4"]
        
        # 执行批处理
        start_time = time.time()
        results = batch_chain.batch(batch_inputs)
        batch_time = time.time() - start_time
        
        # 验证结果
        self.assertEqual(len(results), len(batch_inputs))
        for i, result in enumerate(results):
            expected = f"[处理] {batch_inputs[i]}"
            self.assertEqual(result, expected)
        
        print(f"批处理输入: {batch_inputs}")
        print(f"批处理结果: {results}")
        print(f"批处理时间: {batch_time:.4f}秒")
        print("✅ 基本批处理测试通过")
    
    def test_parallel_vs_sequential_performance(self) -> None:
        """
        测试并行与顺序执行的性能对比
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试并行与顺序执行性能对比 ===")
        
        # 创建模拟延迟的处理函数
        def slow_processing(x: str) -> str:
            time.sleep(0.1)  # 模拟处理时间
            return f"慢速处理: {x}"
        
        slow_chain = RunnableLambda(slow_processing)
        test_inputs = [f"测试{i}" for i in range(5)]
        
        # 顺序执行
        start_time = time.time()
        sequential_results = []
        for input_text in test_inputs:
            result = slow_chain.invoke(input_text)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # 并行批处理
        start_time = time.time()
        parallel_results = slow_chain.batch(test_inputs)
        parallel_time = time.time() - start_time
        
        # 验证结果一致性
        self.assertEqual(sequential_results, parallel_results)
        
        print(f"顺序执行时间: {sequential_time:.4f}秒")
        print(f"并行执行时间: {parallel_time:.4f}秒")
        print(f"性能提升: {sequential_time/parallel_time:.2f}x")
        print("✅ 并行与顺序性能对比测试通过")
    
    def test_async_batch_processing(self) -> None:
        """
        测试异步批处理功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试异步批处理功能 ===")
        
        async def run_async_batch_test() -> None:
            # 创建异步处理函数
            async def async_processing(x: str) -> str:
                await asyncio.sleep(0.1)  # 模拟异步操作
                return f"异步处理: {x}"
            
            async_chain = RunnableLambda(async_processing)
            test_inputs = [f"异步{i}" for i in range(4)]
            
            # 异步批处理
            start_time = time.time()
            results = await async_chain.abatch(test_inputs)
            async_batch_time = time.time() - start_time
            
            # 验证结果
            self.assertEqual(len(results), len(test_inputs))
            for i, result in enumerate(results):
                expected = f"异步处理: {test_inputs[i]}"
                self.assertEqual(result, expected)
            
            print(f"异步批处理输入: {test_inputs}")
            print(f"异步批处理结果: {results}")
            print(f"异步批处理时间: {async_batch_time:.4f}秒")
            print("✅ 异步批处理测试通过")
        
        asyncio.run(run_async_batch_test())


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True) 