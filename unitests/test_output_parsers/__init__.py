"""
输出解析器测试包

提供LangChain输出解析器的全面测试套件，涵盖：
- StrOutputParser：字符串输出解析器功能测试
- JsonOutputParser：JSON输出解析器功能测试
- PydanticOutputParser：Pydantic输出解析器功能测试
- XMLOutputParser：XML输出解析器功能测试
- YAMLOutputParser：YAML输出解析器功能测试
- 自定义输出解析器：自定义解析器创建和使用
- 错误处理和重试机制测试
- 与ChatOpenAI模型的集成应用测试

测试类别：
- Unit Tests: 单元测试，测试各种解析器功能
- Integration Tests: 集成测试，与AI模型结合测试
- Error Handling Tests: 错误处理和重试机制测试
- Custom Parser Tests: 自定义解析器测试

运行方式：
- 运行所有测试: python unitests/test_output_parsers/run_all_tests.py
- 运行基础解析器测试: python -m unittest unitests.test_output_parsers.test_basic_parsers -v
- 运行Pydantic解析器测试: python -m unittest unitests.test_output_parsers.test_pydantic_parsers -v
- 运行自定义解析器测试: python -m unittest unitests.test_output_parsers.test_custom_parsers -v
- 运行错误处理测试: python -m unittest unitests.test_output_parsers.test_error_handling -v

作者: AI Assistant
创建时间: 2025年
基于LangChain官方文档: https://python.langchain.com/docs/how_to/#output-parsers
"""

__version__ = "1.0.0"
__author__ = "LangChain Study Project"

# 导入主要的测试类
from .test_basic_parsers import TestBasicOutputParsers
from .test_pydantic_parsers import TestPydanticOutputParsers
from .test_error_handling import TestOutputParserErrorHandling

__all__ = [
    "TestBasicOutputParsers",
    "TestPydanticOutputParsers", 
    "TestOutputParserErrorHandling"
] 