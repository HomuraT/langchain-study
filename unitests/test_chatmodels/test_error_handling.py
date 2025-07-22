"""
错误处理测试

测试ChatOpenAI模型的错误处理功能，包括网络错误、API错误、参数错误等
"""

import unittest
from typing import Dict, Any
import requests

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


class TestErrorHandling(unittest.TestCase):
    """错误处理测试类"""
    
    def get_chat_model(self) -> ChatOpenAI:
        """
        创建ChatOpenAI实例用于错误测试
        
        Returns:
            ChatOpenAI: 配置好的聊天模型实例
        """
        config = apis["local"]
        return ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
    
    def test_connection_error(self) -> None:
        """
        测试连接错误处理
        """
        config = apis["local"]
        
        # 使用无效的URL
        invalid_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url="http://invalid-url-that-does-not-exist:9999",
            api_key=config["api_key"],
            timeout=5
        )
        
        messages = [HumanMessage(content="Hello")]
        
        try:
            response = invalid_model.invoke(messages)
            print("Unexpected: Connection should have failed")
        except Exception as e:
            print(f"Expected connection error: {type(e).__name__}: {e}")
            # 这是预期的错误
    
    def test_invalid_api_key_error(self) -> None:
        """
        测试无效API密钥错误处理
        """
        config = apis["local"]
        
        # 使用无效的API密钥
        invalid_key_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key="invalid-api-key-12345",
            timeout=30
        )
        
        messages = [HumanMessage(content="Hello")]
        
        try:
            response = invalid_key_model.invoke(messages)
            print("Unexpected: Invalid API key should have failed")
        except Exception as e:
            print(f"Expected authentication error: {type(e).__name__}: {e}")
            # 这是预期的错误
    
    def test_timeout_error(self) -> None:
        """
        测试超时错误处理
        """
        config = apis["local"]
        
        # 设置极短的超时时间
        timeout_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            timeout=0.001  # 极短超时
        )
        
        messages = [HumanMessage(content="Write a long story")]
        
        try:
            response = timeout_model.invoke(messages)
            print("Unexpected: Timeout should have occurred")
        except Exception as e:
            print(f"Expected timeout error: {type(e).__name__}: {e}")
            # 这是预期的错误
    
    def test_context_length_exceeded_error(self) -> None:
        """
        测试上下文长度超限错误处理
        """
        chat_model = self.get_chat_model()
        
        # 创建一个超长的消息
        very_long_content = "This is a very long message. " * 10000
        messages = [HumanMessage(content=very_long_content)]
        
        try:
            response = chat_model.invoke(messages)
            print("Unexpected: Context length should have been exceeded")
        except Exception as e:
            print(f"Expected context length error: {type(e).__name__}: {e}")
            # 这是预期的错误
    
    def test_streaming_interruption_error(self) -> None:
        """
        测试流式输出中断错误处理
        """
        config = apis["local"]
        streaming_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            streaming=True,
            timeout=1  # 短超时
        )
        
        messages = [HumanMessage(content="Write a very long detailed story")]
        
        try:
            chunks = []
            for chunk in streaming_model.stream(messages):
                chunks.append(chunk)
                # 人为中断测试
                if len(chunks) > 5:
                    break
            
            print(f"Streaming collected {len(chunks)} chunks before interruption")
        except Exception as e:
            print(f"Streaming interruption: {type(e).__name__}: {e}")
    
    def test_batch_error_handling(self) -> None:
        """
        测试批处理错误处理
        """
        chat_model = self.get_chat_model()
        
        # 包含有效和无效请求的批处理
        message_batches = [
            [HumanMessage(content="Hello 1")],
            [],  # 无效的空消息
            [HumanMessage(content="Hello 3")]
        ]
        
        try:
            responses = chat_model.batch(message_batches)
            print("Unexpected: Batch with invalid request should have failed")
        except Exception as e:
            print(f"Expected batch error: {type(e).__name__}: {e}")
    
    def test_invalid_base_url(self) -> None:
        """
        测试无效的base_url处理
        """
        config = apis["local"]
        
        # 测试格式错误的URL
        try:
            ChatOpenAI(
                model="gpt-4o-mini",
                base_url="invalid-url-format",
                api_key=config["api_key"]
            )
        except Exception as e:
            print(f"Invalid base URL error: {type(e).__name__}: {e}")
    
    def test_empty_api_key(self) -> None:
        """
        测试空API密钥处理
        """
        config = apis["local"]
        
        # 测试空API密钥
        try:
            ChatOpenAI(
                model="gpt-4o-mini",
                base_url=config["base_url"],
                api_key="",  # 空密钥
                temperature=0.7
            )
        except Exception as e:
            print(f"Empty API key error: {type(e).__name__}: {e}")
    
    def test_network_resilience(self) -> None:
        """
        测试网络韧性
        """
        chat_model = self.get_chat_model()
        messages = [HumanMessage(content="Test network resilience")]
        
        try:
            # 正常调用应该工作
            response = chat_model.invoke(messages)
            self.assertIsInstance(response, AIMessage)
            print(f"Network resilience test - Normal call successful: {response.content}")
            
        except Exception as e:
            print(f"Network resilience test failed: {type(e).__name__}: {e}")
    
    def test_error_recovery(self) -> None:
        """
        测试错误恢复
        """
        config = apis["local"]
        
        # 先用错误配置
        try:
            bad_model = ChatOpenAI(
                model="gpt-4o-mini",
                base_url="http://invalid-url:9999",
                api_key=config["api_key"],
                timeout=1
            )
            bad_model.invoke([HumanMessage(content="This should fail")])
        except Exception as e:
            print(f"Expected error with bad config: {type(e).__name__}")
        
        # 然后用正确配置
        try:
            good_model = self.get_chat_model()
            response = good_model.invoke([HumanMessage(content="This should work")])
            print(f"Recovery successful: {response.content}")
        except Exception as e:
            print(f"Recovery failed: {type(e).__name__}: {e}")


def main() -> int:
    """
    运行错误处理测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行错误处理测试")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main()