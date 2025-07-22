"""
异步操作 vs 同步操作实际应用示例

演示如何在真实场景中使用异步操作提高性能
"""

import asyncio
import time
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.config.api import apis


def get_chat_model() -> ChatOpenAI:
    """
    创建普通聊天模型实例
    
    Returns:
        ChatOpenAI: 聊天模型实例
    """
    config = apis["local"]
    return ChatOpenAI(
        model="gpt-4o-mini",
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=0.7,
        max_tokens=200
    )


class CustomerServiceBot:
    """客服机器人示例 - 展示异步操作的实际应用"""
    
    def __init__(self):
        """初始化客服机器人"""
        self.model = get_chat_model()
        self.system_message = SystemMessage(
            content="你是一个专业的客服助手，简洁地回答用户问题。"
        )
    
    def handle_single_request_sync(self, user_question: str) -> Dict[str, str]:
        """
        同步处理单个用户请求
        
        Args:
            user_question: 用户问题
            
        Returns:
            Dict[str, str]: 处理结果
        """
        messages = [self.system_message, HumanMessage(content=user_question)]
        
        start_time = time.time()
        response = self.model.invoke(messages)
        end_time = time.time()
        
        return {
            "question": user_question,
            "answer": response.content,
            "duration": f"{end_time - start_time:.2f}s"
        }
    
    async def handle_single_request_async(self, user_question: str) -> Dict[str, str]:
        """
        异步处理单个用户请求
        
        Args:
            user_question: 用户问题
            
        Returns:
            Dict[str, str]: 处理结果
        """
        messages = [self.system_message, HumanMessage(content=user_question)]
        
        start_time = time.time()
        response = await self.model.ainvoke(messages)
        end_time = time.time()
        
        return {
            "question": user_question,
            "answer": response.content,
            "duration": f"{end_time - start_time:.2f}s"
        }
    
    def handle_multiple_requests_sync(self, questions: List[str]) -> List[Dict[str, str]]:
        """
        同步处理多个用户请求（顺序执行）
        
        Args:
            questions: 用户问题列表
            
        Returns:
            List[Dict[str, str]]: 处理结果列表
        """
        print("🔄 同步处理多个请求（顺序执行）...")
        start_time = time.time()
        
        results = []
        for question in questions:
            result = self.handle_single_request_sync(question)
            results.append(result)
            print(f"✅ 完成: {question[:30]}... ({result['duration']})")
        
        total_time = time.time() - start_time
        print(f"⏱️ 同步总耗时: {total_time:.2f}s")
        return results
    
    async def handle_multiple_requests_async(self, questions: List[str]) -> List[Dict[str, str]]:
        """
        异步处理多个用户请求（并发执行）
        
        Args:
            questions: 用户问题列表
            
        Returns:
            List[Dict[str, str]]: 处理结果列表
        """
        print("⚡ 异步处理多个请求（并发执行）...")
        start_time = time.time()
        
        # 创建并发任务
        tasks = [self.handle_single_request_async(question) for question in questions]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        print(f"⚡ 异步总耗时: {total_time:.2f}s")
        
        for result in results:
            print(f"✅ 完成: {result['question'][:30]}... ({result['duration']})")
        
        return results


class StreamingChatExample:
    """流式聊天示例 - 展示实时响应"""
    
    def __init__(self):
        """初始化流式聊天"""
        config = apis["local"]
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.7,
            streaming=True
        )
    
    async def streaming_chat(self, user_input: str) -> None:
        """
        异步流式聊天
        
        Args:
            user_input: 用户输入
        """
        messages = [HumanMessage(content=user_input)]
        
        print(f"👤 用户: {user_input}")
        print("🤖 AI: ", end="", flush=True)
        
        # 异步流式输出
        async for chunk in self.model.astream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\n" + "="*50)


async def real_world_example():
    """真实世界应用示例"""
    print("🌟 真实世界异步应用示例")
    print("="*60)
    
    # 模拟客服场景
    customer_service = CustomerServiceBot()
    
    # 模拟多个用户同时提问
    user_questions = [
        "我的订单什么时候发货？",
        "如何退换货？",
        "有什么优惠活动吗？",
        "产品质量怎么样？",
        "支付方式有哪些？"
    ]
    
    print("📋 处理5个用户问题...")
    print()
    
    # 1. 同步处理（顺序执行）
    sync_results = customer_service.handle_multiple_requests_sync(user_questions)
    print()
    
    # 2. 异步处理（并发执行）
    async_results = await customer_service.handle_multiple_requests_async(user_questions)
    print()
    
    # 3. 流式聊天演示
    print("💬 流式聊天演示:")
    streaming_chat = StreamingChatExample()
    await streaming_chat.streaming_chat("请简单介绍一下人工智能的发展历史")


async def concurrent_model_usage_example():
    """同一个模型处理并发请求的示例"""
    print("🔀 同一个模型处理并发请求示例")
    print("="*60)
    
    # 创建一个模型实例
    model = get_chat_model()
    
    # 模拟5个不同用户的问题
    user_requests = [
        ("用户A", "什么是Python？"),
        ("用户B", "如何学习编程？"),
        ("用户C", "人工智能有什么应用？"),
        ("用户D", "数据科学需要什么技能？"),
        ("用户E", "云计算的优势是什么？")
    ]
    
    async def process_user_request(user_id: str, question: str) -> Dict[str, str]:
        """
        处理单个用户请求
        
        Args:
            user_id: 用户ID
            question: 用户问题
            
        Returns:
            Dict[str, str]: 处理结果
        """
        print(f"📨 {user_id} 发起请求: {question}")
        messages = [HumanMessage(content=question)]
        
        start_time = time.time()
        # 注意：这里使用同一个模型实例处理不同用户的请求
        response = await model.ainvoke(messages)
        duration = time.time() - start_time
        
        result = {
            "user": user_id,
            "question": question,
            "answer": response.content[:100] + "..." if len(response.content) > 100 else response.content,
            "duration": f"{duration:.2f}s"
        }
        
        print(f"✅ {user_id} 请求完成 ({result['duration']})")
        return result
    
    # 并发处理所有用户请求
    print("⚡ 开始并发处理...")
    start_time = time.time()
    
    tasks = [process_user_request(user_id, question) for user_id, question in user_requests]
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    print(f"\n🎉 所有请求处理完成！总耗时: {total_time:.2f}s")
    
    print("\n📊 处理结果:")
    for result in results:
        print(f"  {result['user']}: {result['answer']}")


async def main():
    """主函数"""
    print("🚀 异步操作实际应用演示")
    print("="*80)
    
    # 1. 真实世界应用示例
    await real_world_example()
    print("\n")
    
    # 2. 同一个模型并发使用示例
    await concurrent_model_usage_example()
    
    print("\n📝 总结:")
    print("1. 异步编程使用单线程 + 协程，不是多线程")
    print("2. 同一个模型实例可以安全地处理并发请求")
    print("3. 异步操作在IO密集型任务中显著提升性能")
    print("4. 适用场景：Web服务、聊天机器人、批量处理等")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main()) 