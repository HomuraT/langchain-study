"""
LangChain Expression Language (LCEL) 测试套件

这个包包含了全面的LCEL功能测试，涵盖：
- 基础组合功能（RunnableSequence, RunnableParallel）
- 组合语法（| 操作符，.pipe 方法）
- 类型转换（自动类型强制转换）
- 异步操作和并行执行
- 流式传输支持
- 批处理操作
- 错误处理和异常管理

测试模块：
- test_basic_composition: 基础组合功能测试
- test_syntax_operators: 语法操作符测试
- test_type_coercion: 类型转换测试
- test_async_operations: 异步操作测试
- test_streaming: 流式传输测试
- test_parallel_execution: 并行执行测试
- test_error_handling: 错误处理测试

作者: AI Assistant
创建时间: 2025年
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .test_basic_composition import TestLCELBasicComposition
from .test_syntax_operators import TestLCELSyntaxOperators
from .test_type_coercion import TestLCELTypeCoercion
from .test_async_operations import TestLCELAsyncOperations
from .test_streaming import TestLCELStreaming
from .test_parallel_execution import TestLCELParallelExecution
from .test_error_handling import TestLCELErrorHandling

__all__ = [
    "TestLCELBasicComposition",
    "TestLCELSyntaxOperators", 
    "TestLCELTypeCoercion",
    "TestLCELAsyncOperations",
    "TestLCELStreaming",
    "TestLCELParallelExecution",
    "TestLCELErrorHandling"
] 