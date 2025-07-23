"""
LangChain Expression Language (LCEL) 基础组合功能测试

测试 RunnableSequence 和 RunnableParallel 的基本功能和使用场景。

作者: AI Assistant
创建时间: 2025年
"""

import unittest
from typing import Dict, Any, List, Optional
import asyncio
from langchain_core.runnables import (
    RunnableSequence, 
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.config.api import apis


class TestLCELBasicComposition(unittest.TestCase):
    """LCEL基础组合功能测试类"""
    
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
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
        cls.test_input = "测试LCEL功能"
    
    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        # 创建基础的可运行组件
        self.simple_prompt = ChatPromptTemplate.from_template("请分析这个问题: {question}")
        self.add_prefix = RunnableLambda(lambda x: f"前缀: {x}")
        self.add_suffix = RunnableLambda(lambda x: f"{x} :后缀")
        self.double_content = RunnableLambda(lambda x: x * 2 if isinstance(x, str) else str(x) * 2)
        self.count_chars = RunnableLambda(lambda x: len(x) if isinstance(x, str) else len(str(x)))
    
    def test_runnable_sequence_basic(self) -> None:
        """
        测试RunnableSequence的基本功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试RunnableSequence基础功能 ===")
        
        # 创建顺序链
        sequence = self.add_prefix | self.add_suffix
        
        test_input = "Hello World"
        result = sequence.invoke(test_input)
        
        expected = "前缀: Hello World :后缀"
        self.assertEqual(result, expected)
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print(f"期望: {expected}")
        print("✅ RunnableSequence基础功能测试通过")
    
    def test_runnable_sequence_with_prompt_and_model(self) -> None:
        """
        测试RunnableSequence与Prompt和模型的结合
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试RunnableSequence与AI模型结合 ===")
        
        # 创建包含Prompt和模型的链
        sequence = self.simple_prompt | self.model
        
        test_input = {"question": "什么是LCEL？"}
        result = sequence.invoke(test_input)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'content'))
        print(f"输入: {test_input}")
        print(f"输出类型: {type(result)}")
        print(f"输出内容预览: {result.content[:100]}...")
        print("✅ RunnableSequence与AI模型结合测试通过")
    
    def test_runnable_parallel_basic(self) -> None:
        """
        测试RunnableParallel的基本功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试RunnableParallel基础功能 ===")
        
        # 创建并行链
        parallel = RunnableParallel({
            "with_prefix": self.add_prefix,
            "with_suffix": self.add_suffix,
            "char_count": self.count_chars,
            "doubled": self.double_content
        })
        
        test_input = "Hello"
        result = parallel.invoke(test_input)
        
        expected_keys = {"with_prefix", "with_suffix", "char_count", "doubled"}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["with_prefix"], "前缀: Hello")
        self.assertEqual(result["with_suffix"], "Hello :后缀")
        self.assertEqual(result["char_count"], 5)
        self.assertEqual(result["doubled"], "HelloHello")
        
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print("✅ RunnableParallel基础功能测试通过")
    
    def test_runnable_parallel_with_different_inputs(self) -> None:
        """
        测试RunnableParallel处理不同类型输入
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试RunnableParallel处理不同输入类型 ===")
        
        # 测试字符串输入
        parallel = RunnableParallel({
            "original": RunnablePassthrough(),
            "length": self.count_chars,
            "prefixed": self.add_prefix
        })
        
        # 测试字符串
        str_result = parallel.invoke("测试")
        self.assertEqual(str_result["original"], "测试")
        self.assertEqual(str_result["length"], 2)
        print(f"字符串输入结果: {str_result}")
        
        # 测试数字（会被转换为字符串）
        num_result = parallel.invoke(123)
        self.assertEqual(num_result["original"], 123)
        self.assertEqual(num_result["length"], 3)  # "123"的长度
        print(f"数字输入结果: {num_result}")
        
        print("✅ RunnableParallel不同输入类型测试通过")
    
    def test_nested_composition(self) -> None:
        """
        测试嵌套的组合结构
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试嵌套组合结构 ===")
        
        # 创建嵌套结构: 先并行处理，然后顺序处理
        parallel_step = RunnableParallel({
            "prefixed": self.add_prefix,
            "char_count": self.count_chars
        })
        
        # 处理并行结果的函数
        def process_parallel_result(result: Dict[str, Any]) -> str:
            return f"处理结果: {result['prefixed']} (长度: {result['char_count']})"
        
        nested_chain = parallel_step | RunnableLambda(process_parallel_result)
        
        test_input = "LCEL"
        result = nested_chain.invoke(test_input)
        
        expected = "处理结果: 前缀: LCEL (长度: 4)"
        self.assertEqual(result, expected)
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print("✅ 嵌套组合结构测试通过")
    
    def test_complex_parallel_with_sequences(self) -> None:
        """
        测试并行中包含序列的复杂组合
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试并行中包含序列的复杂组合 ===")
        
        # 创建两个不同的序列
        sequence1 = self.add_prefix | self.add_suffix
        
        sequence2 = self.double_content | self.count_chars
        
        # 将序列组合到并行结构中
        complex_parallel = RunnableParallel({
            "processed_text": sequence1,
            "doubled_length": sequence2,
            "original": RunnablePassthrough()
        })
        
        test_input = "Test"
        result = complex_parallel.invoke(test_input)
        
        self.assertEqual(result["processed_text"], "前缀: Test :后缀")
        self.assertEqual(result["doubled_length"], 8)  # "TestTest"的长度
        self.assertEqual(result["original"], "Test")
        
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print("✅ 复杂并行组合测试通过")
    
    def test_runnable_sequence_error_propagation(self) -> None:
        """
        测试RunnableSequence中的错误传播
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试RunnableSequence错误传播 ===")
        
        def failing_function(x: str) -> str:
            if x == "error":
                raise ValueError("测试错误")
            return f"处理: {x}"
        
        sequence = RunnableLambda(failing_function) | self.add_suffix
        
        # 测试正常情况
        normal_result = sequence.invoke("正常输入")
        self.assertEqual(normal_result, "处理: 正常输入 :后缀")
        print(f"正常情况结果: {normal_result}")
        
        # 测试错误情况
        with self.assertRaises(ValueError) as context:
            sequence.invoke("error")
        
        self.assertEqual(str(context.exception), "测试错误")
        print(f"错误传播测试: {context.exception}")
        print("✅ RunnableSequence错误传播测试通过")
    
    def test_runnable_parallel_partial_failure(self) -> None:
        """
        测试RunnableParallel中部分失败的处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试RunnableParallel部分失败处理 ===")
        
        def conditional_fail(x: str) -> str:
            if x == "fail":
                raise RuntimeError("并行处理失败")
            return f"成功处理: {x}"
        
        parallel = RunnableParallel({
            "success": self.add_prefix,
            "might_fail": RunnableLambda(conditional_fail),
            "always_success": RunnablePassthrough()
        })
        
        # 测试正常情况
        normal_result = parallel.invoke("正常")
        self.assertEqual(len(normal_result), 3)
        print(f"正常情况结果: {normal_result}")
        
        # 测试失败情况 - 整个并行操作应该失败
        with self.assertRaises(RuntimeError):
            parallel.invoke("fail")
        
        print("✅ RunnableParallel部分失败处理测试通过")
    
    def test_empty_parallel_and_sequence(self) -> None:
        """
        测试空的并行和序列结构
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试空的并行和序列结构 ===")
        
        # 测试空序列（用RunnablePassthrough模拟）
        empty_sequence = RunnablePassthrough()
        result = empty_sequence.invoke("test")
        self.assertEqual(result, "test")  # 应该直接返回输入
        print(f"空序列结果: {result}")
        
        # 测试空并行
        empty_parallel = RunnableParallel({})
        result = empty_parallel.invoke("test")
        self.assertEqual(result, {})  # 应该返回空字典
        print(f"空并行结果: {result}")
        
        print("✅ 空结构测试通过")
    
    def test_single_element_structures(self) -> None:
        """
        测试单元素的并行和序列结构
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试单元素结构 ===")
        
        # 单元素序列
        single_sequence = self.add_prefix
        result = single_sequence.invoke("test")
        self.assertEqual(result, "前缀: test")
        print(f"单元素序列结果: {result}")
        
        # 单元素并行
        single_parallel = RunnableParallel({"only": self.add_prefix})
        result = single_parallel.invoke("test")
        self.assertEqual(result, {"only": "前缀: test"})
        print(f"单元素并行结果: {result}")
        
        print("✅ 单元素结构测试通过")


if __name__ == "__main__":
    # 配置unittest的详细输出
    unittest.main(verbosity=2, buffer=True) 