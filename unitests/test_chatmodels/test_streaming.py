"""
流式输出测试

测试ChatOpenAI模型的流式输出功能
"""

import unittest
from typing import List, Iterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.callbacks import StreamingStdOutCallbackHandler

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


class TestStreamingChat(unittest.TestCase):
    """流式聊天测试类"""
    
    def get_streaming_chat_model(self) -> ChatOpenAI:
        """
        创建支持流式输出的ChatOpenAI实例
        
        Returns:
            ChatOpenAI: 配置好的流式聊天模型实例
        """
        config = apis["local"]
        return ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.7,
            max_tokens=1000,
            streaming=True,  # 启用流式输出
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    def test_basic_streaming(self) -> None:
        """
        测试基本流式输出功能
        """
        streaming_model = self.get_streaming_chat_model()
        messages = [HumanMessage(content="Count from 1 to 3")]
        
        try:
            # 收集所有chunks
            response_chunks = list(streaming_model.stream(messages))
            
            self.assertGreater(len(response_chunks), 0)
            
            # 验证内容
            full_response = "".join([chunk.content for chunk in response_chunks])
            print(f"Basic streaming response: {full_response}")
            
            # 打印每个chunk
            for i, chunk in enumerate(response_chunks):
                print(f"Chunk {i}: {chunk.content}")
                
        except Exception as e:
            print(f"Basic streaming test failed: {e}")
    
    def test_streaming_with_system_message(self) -> None:
        """
        测试包含系统消息的流式输出
        """
        streaming_model = self.get_streaming_chat_model()
        messages = [
            SystemMessage(content="You are a helpful assistant that counts slowly."),
            HumanMessage(content="Count from 1 to 2")
        ]
        
        try:
            response_chunks = list(streaming_model.stream(messages))
            
            self.assertGreater(len(response_chunks), 1)
            full_response = "".join([chunk.content for chunk in response_chunks])
            print(f"System message streaming response: {full_response}")
        except Exception as e:
            print(f"System message streaming test failed: {e}")
    
    def test_streaming_callback_handler(self) -> None:
        """
        测试流式输出回调处理器
        """
        # 创建自定义回调处理器
        callback_outputs = []
        
        class TestCallbackHandler(StreamingStdOutCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                callback_outputs.append(token)
                print(f"Callback received token: {token}")
        
        config = apis["local"]
        streaming_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            streaming=True,
            callbacks=[TestCallbackHandler()]
        )
        
        # 验证模型配置了回调
        self.assertIsNotNone(streaming_model.callbacks)
        self.assertEqual(len(streaming_model.callbacks), 1)
        self.assertIsInstance(streaming_model.callbacks[0], TestCallbackHandler)
        
        messages = [HumanMessage(content="Say hello")]
        
        try:
            response_chunks = list(streaming_model.stream(messages))
            print(f"Callback handler collected {len(callback_outputs)} tokens")
        except Exception as e:
            print(f"Callback handler test failed: {e}")
    
    def test_streaming_chunk_format(self) -> None:
        """
        测试流式输出chunk的格式
        """
        streaming_model = self.get_streaming_chat_model()
        messages = [HumanMessage(content="Say hello")]
        
        try:
            for chunk in streaming_model.stream(messages):
                # 验证chunk格式
                self.assertIsInstance(chunk, ChatGenerationChunk)
                self.assertTrue(hasattr(chunk, 'content'))
                self.assertIsInstance(chunk.content, str)
                print(f"Chunk format - Type: {type(chunk)}, Content: '{chunk.content}'")
        except Exception as e:
            print(f"Chunk format test failed: {e}")
    
    def test_streaming_long_response(self) -> None:
        """
        测试长响应的流式处理
        """
        streaming_model = self.get_streaming_chat_model()
        messages = [HumanMessage(content="Write a short story about a cat (about 100 words)")]
        
        try:
            response_chunks = list(streaming_model.stream(messages))
            
            self.assertGreater(len(response_chunks), 10)  # 应该有多个chunks
            full_response = "".join([chunk.content for chunk in response_chunks])
            print(f"Long response ({len(response_chunks)} chunks): {full_response}")
            print(f"Total response length: {len(full_response)} characters")
        except Exception as e:
            print(f"Long response streaming test failed: {e}")
    
    def test_streaming_model_configuration(self) -> None:
        """
        测试流式模型配置
        """
        config = apis["local"]
        
        # 测试启用流式输出
        streaming_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            streaming=True
        )
        self.assertTrue(streaming_model.streaming)
        
        # 测试禁用流式输出
        non_streaming_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            streaming=False
        )
        self.assertFalse(non_streaming_model.streaming)
        
        print("Streaming configuration test passed")
    
    def test_streaming_vs_normal(self) -> None:
        """
        比较流式输出和普通输出
        """
        config = apis["local"]
        
        # 流式模型
        streaming_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            streaming=True
        )
        
        # 普通模型
        normal_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            streaming=False
        )
        
        messages = [HumanMessage(content="What is 2+2?")]
        
        try:
            # 流式输出
            streaming_chunks = list(streaming_model.stream(messages))
            streaming_response = "".join([chunk.content for chunk in streaming_chunks])
            
            # 普通输出
            normal_response = normal_model.invoke(messages)
            
            print(f"Streaming response ({len(streaming_chunks)} chunks): {streaming_response}")
            print(f"Normal response: {normal_response.content}")
            
        except Exception as e:
            print(f"Streaming vs normal test failed: {e}")


def main() -> int:
    """
    运行流式输出测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行流式输出测试")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main()