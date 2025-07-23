"""
LangChain Expression Language (LCEL) 错误处理测试

测试LCEL中的各种错误处理场景：
- 链中的错误传播
- 异步操作错误处理
- 批处理错误处理
- 错误恢复机制

作者: AI Assistant
创建时间: 2025年
"""

import unittest
import asyncio
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.config.api import apis


class TestLCELErrorHandling(unittest.TestCase):
    """LCEL错误处理测试类"""
    
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
            max_tokens=100,
            timeout=30
        )
    
    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        # 正常处理函数
        self.safe_function = lambda x: f"安全处理: {x}"
        
        # 可能失败的函数
        def conditional_error_function(x: str) -> str:
            if "error" in x.lower():
                raise ValueError(f"处理失败: {x}")
            return f"条件处理: {x}"
        
        def always_error_function(x: str) -> str:
            raise RuntimeError(f"总是失败: {x}")
        
        self.conditional_error = conditional_error_function
        self.always_error = always_error_function
    
    def test_error_propagation_in_sequence(self) -> None:
        """
        测试序列链中的错误传播
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试序列链中的错误传播 ===")
        
        # 创建包含可能失败函数的序列
        error_chain = (RunnableLambda(self.safe_function) |
                      RunnableLambda(self.conditional_error) |
                      RunnableLambda(self.safe_function))
        
        # 测试正常情况
        normal_result = error_chain.invoke("正常输入")
        expected = "安全处理: 条件处理: 安全处理: 正常输入"
        self.assertEqual(normal_result, expected)
        print(f"正常情况结果: {normal_result}")
        
        # 测试错误情况
        with self.assertRaises(ValueError) as context:
            error_chain.invoke("error输入")
        
        error_msg = str(context.exception)
        self.assertIn("处理失败", error_msg)
        print(f"错误传播测试: {error_msg}")
        print("✅ 序列链错误传播测试通过")
    
    def test_error_in_parallel_execution(self) -> None:
        """
        测试并行执行中的错误处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试并行执行中的错误处理 ===")
        
        # 创建并行结构，包含可能失败的函数
        parallel_dict = {
            "safe": RunnableLambda(self.safe_function),
            "conditional": RunnableLambda(self.conditional_error),
            "another_safe": RunnableLambda(lambda x: f"另一个安全: {x}")
        }
        
        parallel_chain = RunnableParallel(parallel_dict)
        
        # 测试正常情况
        normal_result = parallel_chain.invoke("正常")
        self.assertEqual(len(normal_result), 3)
        self.assertIn("safe", normal_result)
        self.assertIn("conditional", normal_result)
        print(f"并行正常情况: {normal_result}")
        
        # 测试错误情况 - 整个并行操作应该失败
        with self.assertRaises(ValueError):
            parallel_chain.invoke("error")
        
        print("✅ 并行执行错误处理测试通过")
    
    def test_async_error_handling(self) -> None:
        """
        测试异步操作中的错误处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试异步操作错误处理 ===")
        
        async def run_async_error_test() -> None:
            # 异步错误函数
            async def async_error_function(x: str) -> str:
                await asyncio.sleep(0.1)
                if "async_error" in x:
                    raise ValueError(f"异步错误: {x}")
                return f"异步处理: {x}"
            
            async_chain = (RunnableLambda(self.safe_function) |
                          RunnableLambda(async_error_function))
            
            # 测试正常情况
            normal_result = await async_chain.ainvoke("正常异步")
            self.assertIn("异步处理:", normal_result)
            print(f"异步正常情况: {normal_result}")
            
            # 测试异步错误情况
            try:
                await async_chain.ainvoke("async_error")
                self.fail("应该抛出异步错误")
            except ValueError as e:
                self.assertIn("异步错误:", str(e))
                print(f"异步错误处理: {e}")
            
            print("✅ 异步错误处理测试通过")
        
        asyncio.run(run_async_error_test())
    
    def test_batch_error_handling(self) -> None:
        """
        测试批处理中的错误处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试批处理错误处理 ===")
        
        batch_chain = RunnableLambda(self.conditional_error)
        
        # 混合正常和错误输入
        mixed_inputs = ["正常1", "error输入", "正常2"]
        
        # 批处理应该在遇到错误时失败
        with self.assertRaises(ValueError):
            batch_chain.batch(mixed_inputs)
        
        print("✅ 批处理错误处理测试通过")
    
    def test_error_recovery_mechanism(self) -> None:
        """
        测试错误恢复机制
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试错误恢复机制 ===")
        
        def error_recovery_function(x: str) -> str:
            try:
                # 尝试调用可能失败的函数
                return self.conditional_error(x)
            except ValueError as e:
                # 错误恢复
                return f"恢复处理: {x} (原错误: {str(e)})"
        
        recovery_chain = RunnableLambda(error_recovery_function)
        
        # 测试正常情况
        normal_result = recovery_chain.invoke("正常")
        self.assertTrue(normal_result.startswith("条件处理:"))
        print(f"恢复机制正常情况: {normal_result}")
        
        # 测试错误恢复
        error_result = recovery_chain.invoke("error")
        self.assertTrue(error_result.startswith("恢复处理:"))
        self.assertIn("原错误:", error_result)
        print(f"错误恢复结果: {error_result}")
        
        print("✅ 错误恢复机制测试通过")


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True) 