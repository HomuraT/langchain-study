"""
ChatModels测试包

包含ChatOpenAI模型的全面测试套件，涵盖：
- 基础功能测试
- 流式输出测试
- 异步操作测试
- 错误处理测试
- 高级功能测试
- 集成测试

测试类别：
- Unit Tests: 单元测试，使用mock响应
- Integration Tests: 集成测试，需要真实API
- Slow Tests: 耗时测试，可以选择性跳过
- Async Tests: 异步测试

运行方式：
- 运行所有测试: pytest unitests/test_chatmodels/
- 运行单元测试: pytest unitests/test_chatmodels/ -m "unit"
- 运行集成测试: pytest unitests/test_chatmodels/ -m "integration"
- 跳过慢速测试: pytest unitests/test_chatmodels/ -m "not slow"
- 运行异步测试: pytest unitests/test_chatmodels/ -m "async_test"
"""

__version__ = "1.0.0"
__author__ = "LangChain Study Project"

# 导入主要的测试类
from .test_basic_chat import TestBasicChat
from .test_streaming import TestStreamingChat
from .test_async_operations import TestAsyncChat
from .test_error_handling import TestErrorHandling
from .test_advanced_features import TestAdvancedFeatures

# 条件导入集成测试（如果可用）
try:
    from .test_integration import TestIntegration, TestRealWorldScenarios
    INTEGRATION_TESTS_AVAILABLE = True
except ImportError:
    INTEGRATION_TESTS_AVAILABLE = False

__all__ = [
    "TestBasicChat",
    "TestStreamingChat", 
    "TestAsyncChat",
    "TestErrorHandling",
    "TestAdvancedFeatures"
]

if INTEGRATION_TESTS_AVAILABLE:
    __all__.extend(["TestIntegration", "TestRealWorldScenarios"]) 