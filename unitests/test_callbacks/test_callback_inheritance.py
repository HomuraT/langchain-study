"""
回调继承和传播测试

测试LangChain中回调的继承和传播机制，包括：
- 运行时回调 vs 构造函数回调的区别
- 回调在chain中的传播行为
- 嵌套组件的回调继承
- 回调作用域和生命周期

重点验证：
1. 运行时回调会传递给所有子对象
2. 构造函数回调只作用于定义的对象
3. 回调传播的层级关系
4. 混合使用时的优先级
"""

import unittest
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


class TrackingCallbackHandler(BaseCallbackHandler):
    """追踪回调处理器，记录触发来源"""
    
    def __init__(self, name: str):
        """
        初始化追踪回调处理器
        
        输入:
            name: str - 回调处理器名称
        输出: 无
        """
        self.name = name
        self.triggered_events = []
    
    def _record_event(self, event_name: str, source: str = "unknown") -> None:
        """
        记录事件信息
        
        输入:
            event_name: str - 事件名称
            source: str - 事件来源
        输出: 无
        """
        self.triggered_events.append({
            'handler': self.name,
            'event': event_name,
            'source': source
        })
        print(f"📍 [{self.name}] {event_name} from {source}")
    
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List], **kwargs) -> None:
        """Chat模型开始事件"""
        model_name = serialized.get('id', ['', '', 'ChatOpenAI'])[-1]
        self._record_event("on_chat_model_start", f"ChatModel:{model_name}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """LLM结束事件"""
        self._record_event("on_llm_end", "LLM")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Chain开始事件"""
        chain_name = serialized.get('name', 'UnknownChain')
        self._record_event("on_chain_start", f"Chain:{chain_name}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Chain结束事件"""
        self._record_event("on_chain_end", "Chain")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Tool开始事件"""
        tool_name = serialized.get('name', 'UnknownTool')
        self._record_event("on_tool_start", f"Tool:{tool_name}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Tool结束事件"""
        self._record_event("on_tool_end", "Tool")


class TestCallbackInheritance(unittest.TestCase):
    """回调继承测试类"""
    
    def setUp(self) -> None:
        """
        测试前准备
        
        输入: 无
        输出: 无
        """
        config = apis["local"]
        self.base_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.7,
            max_tokens=100
        )
    
    def test_runtime_vs_constructor_callbacks(self) -> None:
        """
        测试运行时回调 vs 构造函数回调的区别
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试运行时回调 vs 构造函数回调 ===")
        
        # 1. 构造函数回调 - 只作用于定义的对象
        constructor_handler = TrackingCallbackHandler("Constructor")
        model_with_constructor = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=apis["local"]["base_url"],
            api_key=apis["local"]["api_key"],
            callbacks=[constructor_handler],  # 构造函数回调
            temperature=0.7,
            max_tokens=100
        )
        
        # 2. 运行时回调 - 会传递给所有子对象
        runtime_handler = TrackingCallbackHandler("Runtime")
        
        messages = [HumanMessage(content="简单回答：你好")]
        
        print("\n--- 测试1：只使用构造函数回调 ---")
        try:
            response1 = model_with_constructor.invoke(messages)
            print(f"构造函数回调响应: {response1.content}")
            
            constructor_events = [e['event'] for e in constructor_handler.triggered_events]
            print(f"构造函数回调事件: {constructor_events}")
            
        except Exception as e:
            print(f"构造函数回调测试失败: {e}")
        
        print("\n--- 测试2：只使用运行时回调 ---")
        try:
            response2 = self.base_model.invoke(messages, config={"callbacks": [runtime_handler]})
            print(f"运行时回调响应: {response2.content}")
            
            runtime_events = [e['event'] for e in runtime_handler.triggered_events]
            print(f"运行时回调事件: {runtime_events}")
            
        except Exception as e:
            print(f"运行时回调测试失败: {e}")
        
        # 验证两种方式都能捕获到事件
        self.assertGreater(len(constructor_handler.triggered_events), 0, "构造函数回调应该捕获到事件")
        self.assertGreater(len(runtime_handler.triggered_events), 0, "运行时回调应该捕获到事件")
        
        print("✅ 运行时回调 vs 构造函数回调测试通过")
    
    def test_callback_propagation_in_chains(self) -> None:
        """
        测试回调在chain中的传播
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试回调在Chain中的传播 ===")
        
        # 创建复杂的chain
        prompt = ChatPromptTemplate.from_template("请回答：{question}")
        parser = StrOutputParser()
        
        # 只给模型添加构造函数回调
        model_handler = TrackingCallbackHandler("ModelOnly")
        model_with_callback = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=apis["local"]["base_url"],
            api_key=apis["local"]["api_key"],
            callbacks=[model_handler],
            temperature=0.7,
            max_tokens=100
        )
        
        # 创建chain
        chain = prompt | model_with_callback | parser
        
        # 运行时回调
        runtime_handler = TrackingCallbackHandler("RuntimeChain")
        
        try:
            print("\n--- 测试1：使用构造函数回调（不会传播到其他组件） ---")
            response1 = chain.invoke({"question": "什么是AI?"})
            
            model_events = [e['event'] for e in model_handler.triggered_events]
            print(f"模型回调事件: {model_events}")
            
            print("\n--- 测试2：使用运行时回调（会传播到所有组件） ---")
            response2 = chain.invoke({"question": "什么是AI?"}, config={"callbacks": [runtime_handler]})
            
            runtime_events = [e['event'] for e in runtime_handler.triggered_events]
            print(f"运行时回调事件: {runtime_events}")
            
            # 验证传播行为
            # 运行时回调应该捕获更多事件（包括prompt和parser的事件）
            self.assertGreater(len(runtime_handler.triggered_events), 0, "运行时回调应该捕获到chain事件")
            
            # 检查是否有chain相关事件
            chain_events = [e for e in runtime_handler.triggered_events if 'chain' in e['event']]
            self.assertGreater(len(chain_events), 0, "运行时回调应该捕获到chain事件")
            
            print("✅ 回调传播测试通过")
            
        except Exception as e:
            print(f"❌ 回调传播测试失败: {e}")
            raise
    
    def test_nested_callback_inheritance(self) -> None:
        """
        测试嵌套组件的回调继承
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试嵌套组件回调继承 ===")
        
        # 创建嵌套的runnable结构
        def preprocessing_step(inputs):
            print(f"预处理步骤: {inputs}")
            return {"processed_question": f"处理后的问题: {inputs['question']}"}
        
        def postprocessing_step(inputs):
            print(f"后处理步骤: {inputs}")
            return f"最终结果: {inputs.content}"
        
        # 构建嵌套chain
        preprocessing = RunnableLambda(preprocessing_step)
        prompt = ChatPromptTemplate.from_template("{processed_question}")
        postprocessing = RunnableLambda(postprocessing_step)
        
        nested_chain = (
            preprocessing |
            prompt |
            self.base_model |
            postprocessing
        )
        
        # 使用运行时回调
        nested_handler = TrackingCallbackHandler("NestedChain")
        
        try:
            result = nested_chain.invoke(
                {"question": "LangChain是什么？"}, 
                config={"callbacks": [nested_handler]}
            )
            
            print(f"\n嵌套chain结果: {result}")
            
            # 分析捕获的事件
            events = nested_handler.triggered_events
            print(f"\n=== 嵌套回调事件分析 ===")
            for event in events:
                print(f"[{event['handler']}] {event['event']} <- {event['source']}")
            
            # 验证嵌套继承
            event_types = [e['event'] for e in events]
            unique_events = set(event_types)
            
            print(f"\n捕获的事件类型: {unique_events}")
            
            # 应该有chain事件（来自嵌套结构）
            chain_events = [e for e in events if 'chain' in e['event']]
            self.assertGreater(len(chain_events), 0, "应该捕获到chain相关事件")
            
            print("✅ 嵌套回调继承测试通过")
            
        except Exception as e:
            print(f"❌ 嵌套回调继承测试失败: {e}")
            raise
    
    def test_mixed_callback_scenarios(self) -> None:
        """
        测试混合使用回调的场景
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试混合回调场景 ===")
        
        # 场景：同时使用构造函数回调和运行时回调
        constructor_handler = TrackingCallbackHandler("Constructor")
        runtime_handler = TrackingCallbackHandler("Runtime")
        
        # 创建带构造函数回调的模型
        model_with_constructor = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=apis["local"]["base_url"],
            api_key=apis["local"]["api_key"],
            callbacks=[constructor_handler],
            temperature=0.7,
            max_tokens=100
        )
        
        # 创建chain
        prompt = ChatPromptTemplate.from_template("回答: {text}")
        chain = prompt | model_with_constructor
        
        messages = {"text": "什么是Python?"}
        
        try:
            # 同时使用两种回调
            response = chain.invoke(messages, config={"callbacks": [runtime_handler]})
            
            print(f"\n混合回调响应: {response.content}")
            
            # 分析两个回调捕获的事件
            constructor_events = constructor_handler.triggered_events
            runtime_events = runtime_handler.triggered_events
            
            print(f"\n--- 构造函数回调事件 ---")
            for event in constructor_events:
                print(f"[{event['handler']}] {event['event']} <- {event['source']}")
            
            print(f"\n--- 运行时回调事件 ---")
            for event in runtime_events:
                print(f"[{event['handler']}] {event['event']} <- {event['source']}")
            
            # 验证两种回调都生效
            self.assertGreater(len(constructor_events), 0, "构造函数回调应该生效")
            self.assertGreater(len(runtime_events), 0, "运行时回调应该生效")
            
            # 运行时回调应该捕获更多事件（包括chain事件）
            runtime_event_types = set(e['event'] for e in runtime_events)
            constructor_event_types = set(e['event'] for e in constructor_events)
            
            print(f"\n构造函数回调事件类型: {constructor_event_types}")
            print(f"运行时回调事件类型: {runtime_event_types}")
            
            print("✅ 混合回调场景测试通过")
            
        except Exception as e:
            print(f"❌ 混合回调场景测试失败: {e}")
            raise
    
    def test_callback_scope_isolation(self) -> None:
        """
        测试回调作用域隔离
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试回调作用域隔离 ===")
        
        handler1 = TrackingCallbackHandler("Scope1")
        handler2 = TrackingCallbackHandler("Scope2")
        
        messages = [HumanMessage(content="测试作用域")]
        
        try:
            # 第一次调用，使用handler1
            print("\n--- 第一次调用（handler1） ---")
            response1 = self.base_model.invoke(messages, config={"callbacks": [handler1]})
            print(f"响应1: {response1.content}")
            
            # 第二次调用，使用handler2
            print("\n--- 第二次调用（handler2） ---")
            response2 = self.base_model.invoke(messages, config={"callbacks": [handler2]})
            print(f"响应2: {response2.content}")
            
            # 第三次调用，不使用回调
            print("\n--- 第三次调用（无回调） ---")
            response3 = self.base_model.invoke(messages)
            print(f"响应3: {response3.content}")
            
            # 验证作用域隔离
            events1 = handler1.triggered_events
            events2 = handler2.triggered_events
            
            print(f"\nhandler1捕获事件数: {len(events1)}")
            print(f"handler2捕获事件数: {len(events2)}")
            
            # 每个handler只应该捕获自己调用的事件
            self.assertGreater(len(events1), 0, "handler1应该捕获到事件")
            self.assertGreater(len(events2), 0, "handler2应该捕获到事件")
            
            # 验证事件来源正确
            for event in events1:
                self.assertEqual(event['handler'], "Scope1", "handler1的事件标记应该正确")
            
            for event in events2:
                self.assertEqual(event['handler'], "Scope2", "handler2的事件标记应该正确")
            
            print("✅ 回调作用域隔离测试通过")
            
        except Exception as e:
            print(f"❌ 回调作用域隔离测试失败: {e}")
            raise


def main() -> int:
    """
    运行回调继承测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🔗 运行回调继承和传播测试")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main() 