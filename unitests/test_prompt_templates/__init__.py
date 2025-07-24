"""
提示模板测试包

提供LangChain提示模板的全面测试套件，涵盖：
- PromptTemplate：字符串模板功能测试
- ChatPromptTemplate：消息模板功能测试
- MessagesPlaceholder：消息占位符功能测试
- 与ChatOpenAI模型的集成应用测试

测试类别：
- Unit Tests: 单元测试，测试各种模板功能
- Integration Tests: 集成测试，与AI模型结合测试

运行方式：
- 运行所有测试: python unitests/test_prompt_templates/run_all_tests.py
- 运行单元测试: python -m unittest unitests.test_prompt_templates.test_prompt_templates -v
- 运行特定测试: python -m unittest unitests.test_prompt_templates.test_prompt_templates.TestPromptTemplates.test_prompt_template_creation -v

作者: AI Assistant
创建时间: 2025年
"""

__version__ = "1.0.0"
__author__ = "LangChain Study Project"

# 导入主要的测试类
from .test_prompt_templates import TestPromptTemplates

__all__ = [
    "TestPromptTemplates"
] 