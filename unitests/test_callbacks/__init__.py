"""
LangChain Callbacks 测试套件

这个包包含了全面的LangChain回调系统测试，涵盖：
- 基础回调处理器（BaseCallbackHandler, AsyncCallbackHandler）
- 运行时回调传递（request time callbacks）
- 构造函数回调（constructor callbacks）
- 异步回调操作
- 流式输出回调
- 回调事件类型（LLM, Chain, Tool, Agent等）
- 错误处理和异常管理
- 性能监控和日志记录

测试模块：
- test_callback_handlers: 基础回调处理器测试
- test_runtime_callbacks: 运行时回调传递测试
- test_constructor_callbacks: 构造函数回调测试
- test_async_callbacks: 异步回调测试
- test_streaming_callbacks: 流式输出回调测试
- test_advanced_callbacks: 高级回调特性测试

测试分类：
- Unit Tests: 单元测试，使用mock响应
- Integration Tests: 集成测试，需要真实API
- Async Tests: 异步回调测试
- Performance Tests: 性能相关测试

运行方式：
- 运行所有测试: python unitests/test_callbacks/run_all_tests.py
- 运行单个测试: python -m unittest unitests.test_callbacks.test_callback_handlers -v
- 运行特定类别: pytest unitests/test_callbacks/ -m "unit"

作者: LangChain Study Project
创建时间: 2025年
"""

__version__ = "1.0.0"
__author__ = "LangChain Study Project"
__description__ = "Comprehensive LangChain Callbacks testing suite"

# 导入主要的测试类
from .test_callback_handlers import TestCallbackHandlers
from .test_callback_inheritance import TestCallbackInheritance

__all__ = [
    "TestCallbackHandlers",
    "TestCallbackInheritance"
]

# 回调事件类型常量
CALLBACK_EVENTS = {
    "chat_model_start": "on_chat_model_start",
    "llm_start": "on_llm_start", 
    "llm_new_token": "on_llm_new_token",
    "llm_end": "on_llm_end",
    "llm_error": "on_llm_error",
    "chain_start": "on_chain_start",
    "chain_end": "on_chain_end",
    "chain_error": "on_chain_error",
    "tool_start": "on_tool_start",
    "tool_end": "on_tool_end",
    "tool_error": "on_tool_error",
    "agent_action": "on_agent_action",
    "agent_finish": "on_agent_finish",
    "retriever_start": "on_retriever_start",
    "retriever_end": "on_retriever_end",
    "retriever_error": "on_retriever_error",
    "text": "on_text",
    "retry": "on_retry"
}

# 测试配置
TEST_CONFIG = {
    "model_name": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30,
    "max_retries": 3
} 