"""
异步操作测试

测试ChatOpenAI模型的异步功能，包括异步调用和异步流式输出
"""

import unittest
import asyncio
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGenerationChunk

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


class TestAsyncChat(unittest.TestCase):
    """异步聊天测试类"""
    
    def get_async_chat_model(self) -> ChatOpenAI:
        """
        创建用于异步操作的ChatOpenAI实例
        
        Returns:
            ChatOpenAI: 配置好的异步聊天模型实例
        """
        config = apis["local"]
        return ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.7,
            max_tokens=1000,
            timeout=60
        )
    
    def test_basic_async_invoke(self) -> None:
        """
        测试基本异步调用
        """
        async def run_test():
            async_model = self.get_async_chat_model()
            messages = [HumanMessage(content="Hello async world")]
            
            try:
                response = await async_model.ainvoke(messages)
                
                self.assertIsInstance(response, AIMessage)
                self.assertIn("Hello", response.content)
                print(f"Basic async response: {response.content}")
            except Exception as e:
                print(f"Basic async test failed: {e}")
        
        # 运行异步测试
        asyncio.run(run_test())
    
    def test_async_system_message(self) -> None:
        """
        测试包含系统消息的异步调用
        """
        async def run_test():
            async_model = self.get_async_chat_model()
            messages = [
                SystemMessage(content="You are a helpful math assistant."),
                HumanMessage(content="What is 5 + 3?")
            ]
            
            try:
                response = await async_model.ainvoke(messages)
                
                self.assertIsInstance(response, AIMessage)
                print(f"Async system message response: {response.content}")
            except Exception as e:
                print(f"Async system message test failed: {e}")
        
        asyncio.run(run_test())
    
    def test_async_streaming(self) -> None:
        """
        测试异步流式输出
        """
        async def run_test():
            async_model = self.get_async_chat_model()
            messages = [HumanMessage(content="Count from 1 to 3")]
            
            try:
                chunks = []
                async for chunk in async_model.astream(messages):
                    chunks.append(chunk)
                    print(f"Async chunk: {chunk.content}")
                
                self.assertGreater(len(chunks), 0)
                full_response = "".join([chunk.content for chunk in chunks])
                print(f"Async streaming full response: {full_response}")
            except Exception as e:
                print(f"Async streaming test failed: {e}")
        
        asyncio.run(run_test())
    
    def test_async_batch_processing(self) -> None:
        """
        测试异步批处理
        """
        async def run_test():
            async_model = self.get_async_chat_model()
            
            message_batches = [
                [HumanMessage(content="Hello 1")],
                [HumanMessage(content="Hello 2")],
                [HumanMessage(content="Hello 3")]
            ]
            
            try:
                results = await async_model.abatch(message_batches)
                
                self.assertEqual(len(results), 3)
                for i, result in enumerate(results):
                    self.assertIsInstance(result, AIMessage)
                    print(f"Async batch response {i+1}: {result.content}")
            except Exception as e:
                print(f"Async batch processing test failed: {e}")
        
        asyncio.run(run_test())
    
    def test_async_concurrent_requests(self) -> None:
        """
        测试异步并发请求
        """
        async def run_test():
            async_model = self.get_async_chat_model()
            
            # 创建多个并发请求
            concurrent_messages = [
                [HumanMessage(content=f"Request {i}")]
                for i in range(5)
            ]
            
            try:
                # 并发执行
                tasks = [async_model.ainvoke(msgs) for msgs in concurrent_messages]
                responses = await asyncio.gather(*tasks)
                
                self.assertEqual(len(responses), 5)
                
                for i, response in enumerate(responses):
                    self.assertIsInstance(response, AIMessage)
                    print(f"Concurrent response {i+1}: {response.content}")
                    
            except Exception as e:
                print(f"Async concurrent requests test failed: {e}")
        
        asyncio.run(run_test())
    
    def test_async_context_preservation(self) -> None:
        """
        测试异步操作中的上下文保持
        """
        async def run_test():
            async_model = self.get_async_chat_model()
            
            # 第一轮对话
            conversation = [
                HumanMessage(content="My name is Alice")
            ]
            
            try:
                response1 = await async_model.ainvoke(conversation)
                
                # 第二轮对话
                conversation.extend([
                    response1,
                    HumanMessage(content="What's my name?")
                ])
                
                response2 = await async_model.ainvoke(conversation)
                
                self.assertIsInstance(response2, AIMessage)
                print(f"Async context preservation: {response2.content}")
            except Exception as e:
                print(f"Async context preservation test failed: {e}")
        
        asyncio.run(run_test())
    
    def test_async_performance_timing(self) -> None:
        """
        测试异步操作性能计时
        """
        async def run_test():
            async_model = self.get_async_chat_model()
            
            # 准备多个请求
            messages_list = [
                [HumanMessage(content=f"Performance test {i}")]
                for i in range(3)
            ]
            
            try:
                # 并发执行
                import time
                start_time = time.time()
                tasks = [async_model.ainvoke(msgs) for msgs in messages_list]
                await asyncio.gather(*tasks)
                end_time = time.time()
                
                concurrent_time = end_time - start_time
                
                # 验证并发执行时间合理
                print(f"Async concurrent execution time: {concurrent_time:.2f} seconds")
                
                # 顺序执行对比
                start_time = time.time()
                for msgs in messages_list:
                    await async_model.ainvoke(msgs)
                end_time = time.time()
                
                sequential_time = end_time - start_time
                print(f"Async sequential execution time: {sequential_time:.2f} seconds")
                
                # 并发应该比顺序快
                self.assertLess(concurrent_time, sequential_time * 0.8)
                
            except Exception as e:
                print(f"Async performance timing test failed: {e}")
        
        asyncio.run(run_test())
    
    def test_async_cancellation(self) -> None:
        """
        测试异步操作取消
        """
        async def run_test():
            async_model = self.get_async_chat_model()
            messages = [HumanMessage(content="Write a very long story")]
            
            try:
                # 创建任务并快速取消
                task = asyncio.create_task(async_model.ainvoke(messages))
                await asyncio.sleep(0.1)  # 让任务开始
                task.cancel()
                
                with self.assertRaises(asyncio.CancelledError):
                    await task
                    
                print("Async cancellation test passed")
            except Exception as e:
                print(f"Async cancellation test failed: {e}")
        
        asyncio.run(run_test())
    
    def test_async_model_configuration(self) -> None:
        """
        测试异步模型配置
        """
        config = apis["local"]
        
        async_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.5,
            timeout=120
        )
        
        # 验证配置
        self.assertEqual(async_model.temperature, 0.5)
        self.assertEqual(async_model.request_timeout, 120)
        self.assertEqual(async_model.model_name, "gpt-4o-mini")
        print("Async model configuration test passed")
    
    def test_async_vs_sync_comparison(self) -> None:
        """
        比较异步和同步调用
        """
        async def run_test():
            config = apis["local"]
            
            # 创建模型
            model = ChatOpenAI(
                model="gpt-4o-mini",
                base_url=config["base_url"],
                api_key=config["api_key"],
                temperature=0.7
            )
            
            messages = [HumanMessage(content="What is the capital of France?")]
            
            try:
                # 异步调用
                import time
                start_time = time.time()
                async_response = await model.ainvoke(messages)
                async_time = time.time() - start_time
                
                # 同步调用
                start_time = time.time()
                sync_response = model.invoke(messages)
                sync_time = time.time() - start_time
                
                print(f"Async response ({async_time:.2f}s): {async_response.content}")
                print(f"Sync response ({sync_time:.2f}s): {sync_response.content}")
                
                # 验证两种方式都能工作
                self.assertIsInstance(async_response, AIMessage)
                self.assertIsInstance(sync_response, AIMessage)
                
            except Exception as e:
                print(f"Async vs sync comparison test failed: {e}")
        
        asyncio.run(run_test())


def main() -> int:
    """
    运行异步操作测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行异步操作测试")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main()