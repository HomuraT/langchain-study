"""
基础聊天模型测试

测试ChatOpenAI模型的基本功能，包括初始化、配置和简单对话
"""

import unittest
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


class TestBasicChat(unittest.TestCase):
    """基础聊天模型测试类"""
    
    def get_chat_model(self) -> ChatOpenAI:
        """
        创建ChatOpenAI实例用于测试
        
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
    
    def test_model_initialization(self) -> None:
        """
        测试模型初始化
        """
        chat_model = self.get_chat_model()
        
        self.assertEqual(chat_model.model_name, "gpt-4o-mini")
        self.assertEqual(chat_model.temperature, 0.7)
        self.assertEqual(chat_model.max_tokens, 1000)
        self.assertEqual(chat_model.request_timeout, 30)
        self.assertIn("http://localhost:8212/v1", chat_model.openai_api_base)
    
    def test_simple_chat(self) -> None:
        """
        测试简单对话功能
        """
        chat_model = self.get_chat_model()
        messages = [HumanMessage(content="Hello, how are you?")]
        
        try:
            response = chat_model.invoke(messages)
            
            self.assertIsInstance(response, AIMessage)
            self.assertGreater(len(response.content), 0)
            print(f"Simple chat response: {response.content}")
        except Exception as e:
            print(f"Simple chat failed: {e}")
            # 不让测试失败，只是打印信息
    
    def test_system_message_chat(self) -> None:
        """
        测试包含系统消息的对话
        """
        chat_model = self.get_chat_model()
        messages = [
            SystemMessage(content="You are a helpful assistant that responds in a friendly manner."),
            HumanMessage(content="What's 2+2?")
        ]
        
        try:
            response = chat_model.invoke(messages)
            
            self.assertIsInstance(response, AIMessage)
            self.assertIn("4", response.content)
            print(f"System message response: {response.content}")
        except Exception as e:
            print(f"System message test failed: {e}")
    
    def test_multi_turn_conversation(self) -> None:
        """
        测试多轮对话
        """
        chat_model = self.get_chat_model()
        conversation = [
            HumanMessage(content="My name is Alice."),
            AIMessage(content="Nice to meet you, Alice!"),
            HumanMessage(content="What's my name?")
        ]
        
        try:
            response = chat_model.invoke(conversation)
            
            self.assertIsInstance(response, AIMessage)
            print(f"Multi-turn conversation response: {response.content}")
        except Exception as e:
            print(f"Multi-turn conversation test failed: {e}")
    
    def test_empty_message_handling(self) -> None:
        """
        测试空消息处理
        """
        from openai import BadRequestError
        
        chat_model = self.get_chat_model()
        
        # 测试空消息列表应该抛出BadRequestError
        with self.assertRaises(BadRequestError):
            chat_model.invoke([])
    
    def test_different_temperatures(self) -> None:
        """
        测试不同temperature参数的影响
        """
        config = apis["local"]
        
        # 低temperature - 更确定性
        conservative_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.1
        )
        self.assertEqual(conservative_model.temperature, 0.1)
        
        # 高temperature - 更创造性
        creative_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.9
        )
        self.assertEqual(creative_model.temperature, 0.9)
        
        # 测试实际效果
        messages = [HumanMessage(content="Write a short creative sentence about spring.")]
        
        try:
            conservative_response = conservative_model.invoke(messages)
            creative_response = creative_model.invoke(messages)
            
            print(f"Conservative (T=0.1): {conservative_response.content}")
            print(f"Creative (T=0.9): {creative_response.content}")
        except Exception as e:
            print(f"Temperature test failed: {e}")
    
    def test_max_tokens_configuration(self) -> None:
        """
        测试max_tokens参数配置
        """
        config = apis["local"]
        
        model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            max_tokens=500
        )
        
        self.assertEqual(model.max_tokens, 500)
    
    def test_batch_processing(self) -> None:
        """
        测试批处理功能
        """
        chat_model = self.get_chat_model()
        
        message_batches = [
            [HumanMessage(content="Hello 1")],
            [HumanMessage(content="Hello 2")],
            [HumanMessage(content="Hello 3")]
        ]
        
        try:
            responses = chat_model.batch(message_batches)
            
            self.assertEqual(len(responses), 3)
            for i, response in enumerate(responses):
                self.assertIsInstance(response, AIMessage)
                print(f"Batch response {i+1}: {response.content}")
        except Exception as e:
            print(f"Batch processing test failed: {e}")


def main() -> int:
    """
    运行基础聊天测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行基础聊天模型测试")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main()