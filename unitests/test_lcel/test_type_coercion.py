"""
LangChain Expression Language (LCEL) 类型转换测试

测试LCEL中的自动类型转换功能：
- 字典到RunnableParallel的转换
- 函数到RunnableLambda的转换
- 各种边界情况和复合场景

作者: AI Assistant
创建时间: 2025年
"""

import unittest
from typing import Dict, Any, List, Optional, Callable
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


class TestLCELTypeCoercion(unittest.TestCase):
    """LCEL类型转换测试类"""
    
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
            max_tokens=300,
            timeout=30
        )
    
    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        # 基础函数定义
        self.add_prefix = lambda x: f"[前缀] {x}"
        self.add_suffix = lambda x: f"{x} [后缀]"
        self.multiply_by_two = lambda x: x * 2 if isinstance(x, (int, float)) else len(str(x)) * 2
        self.to_upper = lambda x: str(x).upper()
        self.reverse_str = lambda x: str(x)[::-1]
        
        # 明确的RunnableLambda版本（用于对比）
        self.explicit_add_prefix = RunnableLambda(self.add_prefix)
        self.explicit_add_suffix = RunnableLambda(self.add_suffix)
    
    def test_dictionary_to_runnable_parallel_basic(self) -> None:
        """
        测试字典到RunnableParallel的基本转换
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试字典到RunnableParallel基本转换 ===")
        
        # 使用字典（会自动转换为RunnableParallel）
        dict_mapping = {
            "prefixed": self.add_prefix,
            "suffixed": self.add_suffix,
            "uppercase": self.to_upper,
            "original": RunnablePassthrough()
        }
        
        # 使用显式RunnableParallel作为对比
        explicit_parallel = RunnableParallel({
            "prefixed": RunnableLambda(self.add_prefix),
            "suffixed": RunnableLambda(self.add_suffix),
            "uppercase": RunnableLambda(self.to_upper),
            "original": RunnablePassthrough()
        })
        
        test_input = "测试文本"
        
        # 通过管道操作测试字典的自动转换
        dict_chain = dict_mapping | RunnableLambda(lambda x: f"组合结果: {x}")
        explicit_chain = explicit_parallel | RunnableLambda(lambda x: f"组合结果: {x}")
        
        dict_result = dict_chain.invoke(test_input)
        explicit_result = explicit_chain.invoke(test_input)
        
        # 结果应该一致
        self.assertEqual(dict_result, explicit_result)
        
        print(f"输入: {test_input}")
        print(f"字典转换结果: {dict_result}")
        print(f"显式RunnableParallel结果: {explicit_result}")
        print("✅ 字典到RunnableParallel基本转换测试通过")
    
    def test_function_to_runnable_lambda_basic(self) -> None:
        """
        测试函数到RunnableLambda的基本转换
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试函数到RunnableLambda基本转换 ===")
        
        # 直接使用函数（会自动转换为RunnableLambda）
        function_chain = RunnableLambda(self.add_prefix) | RunnableLambda(self.to_upper) | RunnableLambda(self.add_suffix)
        
        # 使用显式RunnableLambda作为对比
        explicit_chain = (RunnableLambda(self.add_prefix) | 
                         RunnableLambda(self.to_upper) | 
                         RunnableLambda(self.add_suffix))
        
        test_input = "测试"
        function_result = function_chain.invoke(test_input)
        explicit_result = explicit_chain.invoke(test_input)
        
        self.assertEqual(function_result, explicit_result)
        
        print(f"输入: {test_input}")
        print(f"函数自动转换结果: {function_result}")
        print(f"显式RunnableLambda结果: {explicit_result}")
        print("✅ 函数到RunnableLambda基本转换测试通过")
    
    def test_nested_dictionary_coercion(self) -> None:
        """
        测试嵌套字典的类型转换
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试嵌套字典类型转换 ===")
        
        # 创建嵌套的字典结构
        inner_dict = {
            "processed": self.add_prefix,
            "length": lambda x: len(str(x))
        }
        
        outer_dict = {
            "inner_results": inner_dict,
            "original": RunnablePassthrough(),
            "reversed": self.reverse_str
        }
        
        # 通过管道测试嵌套转换
        def process_nested_result(result: Dict[str, Any]) -> str:
            inner = result["inner_results"]
            return f"嵌套结果 - 处理: {inner['processed']}, 长度: {inner['length']}, 原始: {result['original']}, 反转: {result['reversed']}"
        
        nested_chain = outer_dict | RunnableLambda(process_nested_result)
        
        test_input = "嵌套"
        result = nested_chain.invoke(test_input)
        
        expected_parts = ["[前缀] 嵌套", "长度: 2", "原始: 嵌套", "反转: 套嵌"]
        for part in expected_parts:
            self.assertIn(part, result)
        
        print(f"输入: {test_input}")
        print(f"嵌套转换结果: {result}")
        print("✅ 嵌套字典类型转换测试通过")
    
    def test_mixed_types_in_dictionary(self) -> None:
        """
        测试字典中混合不同类型的转换
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试字典中混合类型转换 ===")
        
        # 字典包含不同类型的可运行对象
        mixed_dict = {
            "lambda_func": self.add_prefix,  # 普通函数
            "explicit_runnable": RunnableLambda(self.add_suffix),  # 显式RunnableLambda
            "passthrough": RunnablePassthrough(),  # RunnablePassthrough
            "nested_dict": {  # 嵌套字典
                "upper": self.to_upper,
                "length": lambda x: len(str(x))
            }
        }
        
        def analyze_mixed_results(results: Dict[str, Any]) -> str:
            return (f"混合结果 - "
                   f"函数: {results['lambda_func']}, "
                   f"显式: {results['explicit_runnable']}, "
                   f"原始: {results['passthrough']}, "
                   f"嵌套大写: {results['nested_dict']['upper']}, "
                   f"嵌套长度: {results['nested_dict']['length']}")
        
        mixed_chain = mixed_dict | RunnableLambda(analyze_mixed_results)
        
        test_input = "混合"
        result = mixed_chain.invoke(test_input)
        
        # 验证各部分都存在
        expected_parts = ["[前缀] 混合", "混合 [后缀]", "原始: 混合", "嵌套大写: 混合", "嵌套长度: 2"]
        for part in expected_parts:
            self.assertIn(part, result)
        
        print(f"输入: {test_input}")
        print(f"混合类型结果: {result}")
        print("✅ 字典中混合类型转换测试通过")
    
    def test_function_with_different_signatures(self) -> None:
        """
        测试不同函数签名的转换
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试不同函数签名转换 ===")
        
        # 不同参数类型的函数
        def single_param_func(x: str) -> str:
            return f"单参数: {x}"
        
        def typed_func(text: str) -> str:
            return f"类型化: {text.capitalize()}"
        
        def generic_func(item) -> str:
            return f"通用: {str(item).lower()}"
        
        # 创建包含不同签名函数的链
        signature_chain = RunnableLambda(single_param_func) | RunnableLambda(typed_func) | RunnableLambda(generic_func)
        
        test_input = "签名测试"
        result = signature_chain.invoke(test_input)
        
        self.assertIsNotNone(result)
        self.assertIn("通用:", result)
        
        print(f"输入: {test_input}")
        print(f"不同签名结果: {result}")
        print("✅ 不同函数签名转换测试通过")
    
    def test_lambda_vs_def_function_coercion(self) -> None:
        """
        测试lambda函数与def函数的转换对比
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试lambda与def函数转换对比 ===")
        
        # lambda函数
        lambda_func = lambda x: f"Lambda: {x}"
        
        # def函数
        def def_func(x: str) -> str:
            return f"Def: {x}"
        
        # 内嵌函数
        def create_inner_func():
            def inner_func(x: str) -> str:
                return f"Inner: {x}"
            return inner_func
        
        inner_func = create_inner_func()
        
        # 测试所有类型的函数转换
        lambda_chain = RunnableLambda(lambda_func) | RunnableLambda(self.add_suffix)
        def_chain = RunnableLambda(def_func) | RunnableLambda(self.add_suffix)
        inner_chain = RunnableLambda(inner_func) | RunnableLambda(self.add_suffix)
        
        test_input = "函数类型"
        
        lambda_result = lambda_chain.invoke(test_input)
        def_result = def_chain.invoke(test_input)
        inner_result = inner_chain.invoke(test_input)
        
        # 验证所有类型都能正确转换
        self.assertTrue(lambda_result.startswith("Lambda:"))
        self.assertTrue(def_result.startswith("Def:"))
        self.assertTrue(inner_result.startswith("Inner:"))
        
        print(f"输入: {test_input}")
        print(f"Lambda函数结果: {lambda_result}")
        print(f"Def函数结果: {def_result}")
        print(f"内嵌函数结果: {inner_result}")
        print("✅ 不同函数类型转换测试通过")
    
    def test_coercion_with_prompt_and_model(self) -> None:
        """
        测试类型转换与Prompt和模型的结合
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试类型转换与AI模型结合 ===")
        
        # 预处理和后处理函数
        def preprocess(question: str) -> Dict[str, str]:
            return {"question": question.strip()}
        
        def postprocess(response: str) -> str:
            return f"AI回答: {response.strip()}"
        
        # 创建Prompt
        prompt = ChatPromptTemplate.from_template("简要回答: {question}")
        
        # 混合使用函数、字典和模型
        preprocessing_dict = {
            "processed_input": preprocess,
            "original": RunnablePassthrough()
        }
        
        # 提取处理后的输入
        extract_processed = lambda x: x["processed_input"]
        
        # 完整链：字典转换 -> 提取 -> Prompt -> 模型 -> 后处理
        full_chain = (RunnableParallel(preprocessing_dict) | 
                     RunnableLambda(extract_processed) | 
                     prompt | 
                     self.model | 
                     StrOutputParser() |
                     RunnableLambda(postprocess))
        
        test_input = "  什么是类型转换？  "
        result = full_chain.invoke(test_input)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("AI回答:"))
        
        print(f"输入: '{test_input}'")
        print(f"AI处理结果: {result}")
        print("✅ 类型转换与AI模型结合测试通过")
    
    def test_type_coercion_error_handling(self) -> None:
        """
        测试类型转换中的错误处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试类型转换错误处理 ===")
        
        # 可能失败的函数
        def risky_function(x: str) -> str:
            if x == "error":
                raise ValueError("转换处理错误")
            return f"安全处理: {x}"
        
        # 包含可能失败函数的字典
        risky_dict = {
            "safe": self.add_prefix,
            "risky": risky_function,
            "backup": lambda x: f"备用: {x}"
        }
        
        def handle_results(results: Dict[str, Any]) -> str:
            return f"结果 - 安全: {results['safe']}, 风险: {results['risky']}, 备用: {results['backup']}"
        
        error_chain = risky_dict | RunnableLambda(handle_results)
        
        # 测试正常情况
        normal_result = error_chain.invoke("正常")
        self.assertIn("安全处理:", normal_result)
        print(f"正常情况: {normal_result}")
        
        # 测试错误情况
        with self.assertRaises(ValueError) as context:
            error_chain.invoke("error")
        
        self.assertEqual(str(context.exception), "转换处理错误")
        print(f"错误处理: {context.exception}")
        print("✅ 类型转换错误处理测试通过")
    
    def test_complex_coercion_performance(self) -> None:
        """
        测试复杂类型转换的性能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试复杂类型转换性能 ===")
        
        import time
        
        # 创建复杂的混合结构
        complex_dict = {
            f"func_{i}": lambda x, i=i: f"处理{i}: {x}" for i in range(10)
        }
        complex_dict["passthrough"] = RunnablePassthrough()
        
        def combine_complex_results(results: Dict[str, Any]) -> str:
            processed = [v for k, v in results.items() if k != "passthrough"]
            return f"组合了{len(processed)}个结果，原始: {results['passthrough']}"
        
        complex_chain = complex_dict | RunnableLambda(combine_complex_results)
        
        # 性能测试
        test_input = "性能测试"
        iterations = 50
        
        start_time = time.time()
        for _ in range(iterations):
            result = complex_chain.invoke(test_input)
        end_time = time.time()
        
        execution_time = end_time - start_time
        avg_time = execution_time / iterations
        
        print(f"复杂转换执行时间: {execution_time:.4f}秒 ({iterations}次)")
        print(f"平均每次: {avg_time:.4f}秒")
        print(f"最终结果: {result}")
        
        # 验证结果正确性
        self.assertIn("组合了10个结果", result)
        self.assertIn("原始: 性能测试", result)
        
        print("✅ 复杂类型转换性能测试通过")
    
    def test_coercion_type_preservation(self) -> None:
        """
        测试类型转换中的类型保持
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试类型转换中的类型保持 ===")
        
        # 测试不同输入输出类型的保持
        def str_to_int(x: str) -> int:
            return len(x)
        
        def int_to_float(x: int) -> float:
            return float(x * 2.5)
        
        def float_to_str(x: float) -> str:
            return f"浮点数: {x}"
        
        # 类型转换链
        type_chain = RunnableLambda(str_to_int) | RunnableLambda(int_to_float) | RunnableLambda(float_to_str)
        
        test_input = "类型"
        result = type_chain.invoke(test_input)
        
        # 验证最终结果
        expected = "浮点数: 5.0"  # len("类型") = 2, 2 * 2.5 = 5.0
        self.assertEqual(result, expected)
        
        print(f"输入: {test_input} (类型: {type(test_input).__name__})")
        print(f"输出: {result} (类型: {type(result).__name__})")
        print("✅ 类型转换中的类型保持测试通过")
    
    def test_coercion_with_custom_classes(self) -> None:
        """
        测试自定义类的转换
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试自定义类转换 ===")
        
        class TextProcessor:
            def __init__(self, prefix: str):
                self.prefix = prefix
            
            def __call__(self, text: str) -> str:
                return f"{self.prefix}: {text}"
        
        class TextAnalyzer:
            @staticmethod
            def analyze(text: str) -> Dict[str, Any]:
                return {
                    "length": len(text),
                    "words": len(text.split()),
                    "uppercase_count": sum(1 for c in text if c.isupper())
                }
        
        # 使用自定义类实例
        processor = TextProcessor("自定义")
        analyzer = TextAnalyzer()
        
        # 创建包含自定义类的字典
        custom_dict = {
            "processed": processor,  # 可调用对象
            "analyzed": analyzer.analyze,  # 方法
            "simple": lambda x: f"简单: {x}"
        }
        
        def combine_custom_results(results: Dict[str, Any]) -> str:
            analysis = results["analyzed"]
            return (f"自定义处理结果 - "
                   f"处理: {results['processed']}, "
                   f"分析: 长度{analysis['length']}/词数{analysis['words']}, "
                   f"简单: {results['simple']}")
        
        custom_chain = custom_dict | RunnableLambda(combine_custom_results)
        
        test_input = "Custom Class Test"
        result = custom_chain.invoke(test_input)
        
        # 验证各部分
        self.assertIn("自定义: Custom Class Test", result)
        self.assertIn("长度17", result)  # len("Custom Class Test") = 17
        self.assertIn("词数3", result)   # 3个单词
        
        print(f"输入: {test_input}")
        print(f"自定义类结果: {result}")
        print("✅ 自定义类转换测试通过")


if __name__ == "__main__":
    # 配置unittest的详细输出
    unittest.main(verbosity=2, buffer=True) 