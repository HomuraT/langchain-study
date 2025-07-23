"""
LangChain Expression Language (LCEL) 异步操作测试

测试LCEL的异步功能：
- 异步调用 (ainvoke)
- 异步流式输出 (astream)
- 异步批处理 (abatch)
- 并发执行性能
- 异步错误处理

作者: AI Assistant
创建时间: 2025年
"""

import unittest
import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncIterator
from langchain_core.runnables import (
    RunnableSequence, 
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.config.api import apis


class TestLCELAsyncOperations(unittest.TestCase):
    """LCEL异步操作测试类"""
    
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
            max_tokens=200,
            timeout=30
        )
    
    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        # 同步函数
        self.add_prefix = lambda x: f"[前缀] {x}"
        self.add_suffix = lambda x: f"{x} [后缀]"
        self.count_chars = lambda x: len(str(x))
        
        # 异步函数
        async def async_add_prefix(x: str) -> str:
            await asyncio.sleep(0.1)  # 模拟异步操作
            return f"[异步前缀] {x}"
        
        async def async_add_suffix(x: str) -> str:
            await asyncio.sleep(0.1)  # 模拟异步操作
            return f"{x} [异步后缀]"
        
        async def async_uppercase(x: str) -> str:
            await asyncio.sleep(0.05)  # 模拟异步操作
            return str(x).upper()
        
        self.async_add_prefix = async_add_prefix
        self.async_add_suffix = async_add_suffix
        self.async_uppercase = async_uppercase
        
        # 创建Prompt
        self.simple_prompt = ChatPromptTemplate.from_template("简要回答: {question}")
    
    def test_basic_async_invoke(self) -> None:
        """
        测试基本的异步调用功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试基本异步调用功能 ===")
        
        async def run_async_test() -> None:
            # 创建包含异步和同步函数的链
            async_chain = (RunnableLambda(self.async_add_prefix) |
                          RunnableLambda(self.add_suffix) |
                          RunnableLambda(self.async_uppercase))
            
            test_input = "异步测试"
            result = await async_chain.ainvoke(test_input)
            
            expected = "[异步前缀] 异步测试 [后缀]".upper()
            self.assertEqual(result, expected)
            
            print(f"输入: {test_input}")
            print(f"异步调用结果: {result}")
            print("✅ 基本异步调用测试通过")
        
        # 运行异步测试
        asyncio.run(run_async_test())
    
    def test_async_vs_sync_performance(self) -> None:
        """
        测试异步与同步操作的性能对比
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试异步与同步性能对比 ===")
        
        async def run_performance_test() -> None:
            # 同步链
            sync_chain = (RunnableLambda(self.add_prefix) |
                         RunnableLambda(self.add_suffix))
            
            # 异步链
            async_chain = (RunnableLambda(self.async_add_prefix) |
                          RunnableLambda(self.async_add_suffix))
            
            test_input = "性能测试"
            iterations = 5
            
            # 测试同步性能
            start_time = time.time()
            for _ in range(iterations):
                sync_result = sync_chain.invoke(test_input)
            sync_time = time.time() - start_time
            
            # 测试异步性能
            start_time = time.time()
            for _ in range(iterations):
                async_result = await async_chain.ainvoke(test_input)
            async_time = time.time() - start_time
            
            print(f"同步执行时间: {sync_time:.4f}秒 ({iterations}次)")
            print(f"异步执行时间: {async_time:.4f}秒 ({iterations}次)")
            print(f"同步结果: {sync_result}")
            print(f"异步结果: {async_result}")
            
            # 验证结果正确性
            self.assertTrue(sync_result.startswith("[前缀]"))
            self.assertTrue(async_result.startswith("[异步前缀]"))
            
            print("✅ 异步与同步性能对比测试通过")
        
        asyncio.run(run_performance_test())
    
    def test_async_parallel_execution(self) -> None:
        """
        测试异步并行执行
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试异步并行执行 ===")
        
        async def run_parallel_test() -> None:
            # 创建并行结构，包含异步函数
            parallel_dict = {
                "async_prefix": RunnableLambda(self.async_add_prefix),
                "async_suffix": RunnableLambda(self.async_add_suffix),
                "sync_count": RunnableLambda(self.count_chars),
                "original": RunnablePassthrough()
            }
            
            parallel_chain = parallel_dict | RunnableLambda(
                lambda x: f"并行结果: {x['async_prefix']}, {x['async_suffix']}, 长度: {x['sync_count']}, 原始: {x['original']}"
            )
            
            test_input = "并行异步"
            
            # 测试异步并行执行时间
            start_time = time.time()
            result = await parallel_chain.ainvoke(test_input)
            execution_time = time.time() - start_time
            
            # 验证结果
            self.assertIn("[异步前缀] 并行异步", result)
            self.assertIn("并行异步 [异步后缀]", result)
            self.assertIn("长度: 4", result)
            
            print(f"输入: {test_input}")
            print(f"并行异步结果: {result}")
            print(f"执行时间: {execution_time:.4f}秒")
            print("✅ 异步并行执行测试通过")
        
        asyncio.run(run_parallel_test())


if __name__ == "__main__":
    # 配置unittest的详细输出
    unittest.main(verbosity=2, buffer=True) 