"""
LangChain Expression Language (LCEL) 语法操作符测试

测试 | 操作符和 .pipe 方法的各种使用场景和功能。

作者: AI Assistant
创建时间: 2025年
"""

import unittest
from typing import Dict, Any, List, Optional
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


class TestLCELSyntaxOperators(unittest.TestCase):
    """LCEL语法操作符测试类"""
    
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
            max_tokens=500,
            timeout=30
        )
    
    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        # 创建基础组件
        self.add_prefix = RunnableLambda(lambda x: f"[前缀] {x}")
        self.add_suffix = RunnableLambda(lambda x: f"{x} [后缀]")
        self.uppercase = RunnableLambda(lambda x: x.upper() if isinstance(x, str) else str(x).upper())
        self.count_chars = RunnableLambda(lambda x: len(str(x)))
        self.reverse_string = RunnableLambda(lambda x: str(x)[::-1])
        
        # 创建Prompt
        self.simple_prompt = ChatPromptTemplate.from_template("请用一句话回答: {question}")
        self.analysis_prompt = ChatPromptTemplate.from_template("请分析以下内容: {content}")
        
        # 输出解析器
        self.str_parser = StrOutputParser()
    
    def test_pipe_operator_basic(self) -> None:
        """
        测试|操作符的基本功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试|操作符基础功能 ===")
        
        # 使用|操作符创建链
        chain = self.add_prefix | self.add_suffix | self.uppercase
        
        test_input = "Hello World"
        result = chain.invoke(test_input)
        
        expected = "[前缀] HELLO WORLD [后缀]"
        self.assertEqual(result, expected)
        
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print(f"期望: {expected}")
        print("✅ |操作符基础功能测试通过")
    
    def test_pipe_method_basic(self) -> None:
        """
        测试.pipe方法的基本功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试.pipe方法基础功能 ===")
        
        # 使用.pipe方法创建链
        chain = (self.add_prefix
                .pipe(self.add_suffix)
                .pipe(self.uppercase))
        
        test_input = "Hello World"
        result = chain.invoke(test_input)
        
        expected = "[前缀] HELLO WORLD [后缀]"
        self.assertEqual(result, expected)
        
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print(f"期望: {expected}")
        print("✅ .pipe方法基础功能测试通过")
    
    def test_pipe_operator_equivalence(self) -> None:
        """
        测试|操作符与RunnableSequence的等价性
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试|操作符与RunnableSequence等价性 ===")
        
        # 使用|操作符
        pipe_chain = self.add_prefix | self.uppercase | self.add_suffix
        
        # 使用.pipe方法（等价的链式语法）
        sequence_chain = (self.add_prefix
                         .pipe(self.uppercase)
                         .pipe(self.add_suffix))
        
        test_input = "Test Equivalence"
        pipe_result = pipe_chain.invoke(test_input)
        sequence_result = sequence_chain.invoke(test_input)
        
        self.assertEqual(pipe_result, sequence_result)
        
        print(f"输入: {test_input}")
        print(f"|操作符结果: {pipe_result}")
        print(f"RunnableSequence结果: {sequence_result}")
        print("✅ 等价性测试通过")
    
    def test_mixed_syntax_styles(self) -> None:
        """
        测试混合使用不同语法风格
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试混合语法风格 ===")
        
        # 混合使用|操作符和.pipe方法
        chain = (self.add_prefix 
                | self.uppercase
                .pipe(self.add_suffix)
                .pipe(self.reverse_string))
        
        test_input = "Mixed"
        result = chain.invoke(test_input)
        
        # 预期结果: "[前缀] MIXED [后缀]" 反转
        expected = "]缀后[ DEXIM ]缀前["
        self.assertEqual(result, expected)
        
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print(f"期望: {expected}")
        print("✅ 混合语法风格测试通过")
    
    def test_pipe_with_parallel_structures(self) -> None:
        """
        测试管道操作与并行结构的结合
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试管道与并行结构结合 ===")
        
        # 创建并行结构
        parallel = RunnableParallel({
            "prefixed": self.add_prefix,
            "suffixed": self.add_suffix,
            "count": self.count_chars
        })
        
        # 处理并行结果的函数
        def combine_parallel_results(results: Dict[str, Any]) -> str:
            return f"组合结果: {results['prefixed']} | {results['suffixed']} | 长度: {results['count']}"
        
        # 使用|操作符连接并行和后处理
        chain = parallel | RunnableLambda(combine_parallel_results)
        
        test_input = "Test"
        result = chain.invoke(test_input)
        
        expected = "组合结果: [前缀] Test | Test [后缀] | 长度: 4"
        self.assertEqual(result, expected)
        
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print("✅ 管道与并行结构结合测试通过")
    
    def test_pipe_with_prompt_and_model(self) -> None:
        """
        测试管道操作与Prompt和模型的结合
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试管道与AI模型结合 ===")
        
        # 使用|操作符创建完整的AI链
        chain = self.simple_prompt | self.model | self.str_parser
        
        test_input = {"question": "什么是LangChain?"}
        result = chain.invoke(test_input)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        print(f"输入: {test_input}")
        print(f"输出类型: {type(result)}")
        print(f"输出长度: {len(result)}")
        print(f"输出预览: {result[:100]}...")
        print("✅ 管道与AI模型结合测试通过")
    
    def test_complex_chaining_with_preprocessing(self) -> None:
        """
        测试复杂的链式操作与预处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试复杂链式操作与预处理 ===")
        
        # 预处理函数
        def preprocess_input(x: str) -> Dict[str, str]:
            return {"content": x.strip().lower()}
        
        def postprocess_output(x: str) -> str:
            return f"最终结果: {x.strip()}"
        
        # 创建复杂链
        chain = (RunnableLambda(preprocess_input) 
                | self.analysis_prompt 
                | self.model 
                | self.str_parser
                | RunnableLambda(postprocess_output))
        
        test_input = "  LCEL是什么技术？  "
        result = chain.invoke(test_input)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("最终结果:"))
        
        print(f"输入: '{test_input}'")
        print(f"输出: {result}")
        print("✅ 复杂链式操作测试通过")
    
    def test_pipe_operator_with_different_types(self) -> None:
        """
        测试|操作符处理不同数据类型
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试|操作符处理不同数据类型 ===")
        
        # 处理不同类型的函数
        def process_input(x: Any) -> str:
            if isinstance(x, dict):
                return f"字典: {x}"
            elif isinstance(x, list):
                return f"列表: {x}"
            elif isinstance(x, int):
                return f"数字: {x}"
            else:
                return f"字符串: {x}"
        
        chain = RunnableLambda(process_input) | self.add_prefix | self.add_suffix
        
        # 测试不同类型
        test_cases = [
            "文本",
            123,
            {"key": "value"},
            [1, 2, 3]
        ]
        
        for test_input in test_cases:
            result = chain.invoke(test_input)
            self.assertTrue(result.startswith("[前缀]"))
            self.assertTrue(result.endswith("[后缀]"))
            print(f"输入 {type(test_input).__name__}: {test_input} -> {result}")
        
        print("✅ 不同数据类型处理测试通过")
    
    def test_pipe_operator_performance_comparison(self) -> None:
        """
        测试|操作符与显式RunnableSequence的性能对比
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试|操作符性能对比 ===")
        
        import time
        
        # 创建相同功能的链
        pipe_chain = self.add_prefix | self.uppercase | self.add_suffix | self.reverse_string
        sequence_chain = (self.add_prefix
                         .pipe(self.uppercase)
                         .pipe(self.add_suffix)
                         .pipe(self.reverse_string))
        
        test_input = "Performance Test"
        iterations = 100
        
        # 测试|操作符性能
        start_time = time.time()
        for _ in range(iterations):
            pipe_chain.invoke(test_input)
        pipe_time = time.time() - start_time
        
        # 测试RunnableSequence性能
        start_time = time.time()
        for _ in range(iterations):
            sequence_chain.invoke(test_input)
        sequence_time = time.time() - start_time
        
        print(f"|操作符执行时间: {pipe_time:.4f}秒 ({iterations}次)")
        print(f".pipe方法执行时间: {sequence_time:.4f}秒 ({iterations}次)")
        print(f"性能差异: {abs(pipe_time - sequence_time):.4f}秒")
        
        # 确保结果一致
        pipe_result = pipe_chain.invoke(test_input)
        sequence_result = sequence_chain.invoke(test_input)
        self.assertEqual(pipe_result, sequence_result)
        
        print("✅ 性能对比测试通过")
    
    def test_nested_pipe_operations(self) -> None:
        """
        测试嵌套的管道操作
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试嵌套管道操作 ===")
        
        # 创建子链
        preprocessing_chain = self.add_prefix | self.uppercase
        postprocessing_chain = self.add_suffix | self.reverse_string
        
        # 组合成更大的链
        full_chain = preprocessing_chain | postprocessing_chain
        
        test_input = "Nested"
        result = full_chain.invoke(test_input)
        
        # 预期: "[前缀] NESTED" -> "[前缀] NESTED [后缀]" -> 反转
        expected = "]缀后[ DETSEN ]缀前["
        self.assertEqual(result, expected)
        
        print(f"输入: {test_input}")
        print(f"输出: {result}")
        print(f"期望: {expected}")
        print("✅ 嵌套管道操作测试通过")
    
    def test_pipe_operator_error_handling(self) -> None:
        """
        测试|操作符中的错误处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试|操作符错误处理 ===")
        
        def failing_step(x: str) -> str:
            if "error" in x.lower():
                raise ValueError(f"处理失败: {x}")
            return f"成功处理: {x}"
        
        chain = self.add_prefix | RunnableLambda(failing_step) | self.add_suffix
        
        # 测试正常情况
        normal_result = chain.invoke("正常输入")
        expected = "成功处理: [前缀] 正常输入 [后缀]"
        self.assertEqual(normal_result, expected)
        print(f"正常情况: {normal_result}")
        
        # 测试错误情况
        with self.assertRaises(ValueError) as context:
            chain.invoke("error输入")
        
        error_msg = str(context.exception)
        self.assertIn("处理失败", error_msg)
        print(f"错误处理: {error_msg}")
        print("✅ |操作符错误处理测试通过")
    
    def test_pipe_method_vs_operator_readability(self) -> None:
        """
        测试.pipe方法与|操作符的可读性对比
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试.pipe方法与|操作符可读性对比 ===")
        
        # 使用|操作符的版本
        operator_chain = (
            self.add_prefix | 
            self.uppercase | 
            self.add_suffix | 
            self.reverse_string
        )
        
        # 使用.pipe方法的版本
        pipe_method_chain = (
            self.add_prefix
            .pipe(self.uppercase)
            .pipe(self.add_suffix)
            .pipe(self.reverse_string)
        )
        
        test_input = "Readability"
        operator_result = operator_chain.invoke(test_input)
        pipe_result = pipe_method_chain.invoke(test_input)
        
        # 确保结果完全一致
        self.assertEqual(operator_result, pipe_result)
        
        print(f"输入: {test_input}")
        print(f"|操作符结果: {operator_result}")
        print(f".pipe方法结果: {pipe_result}")
        print("✅ 可读性对比测试通过 - 两种方法功能完全一致")


if __name__ == "__main__":
    # 配置unittest的详细输出
    unittest.main(verbosity=2, buffer=True) 