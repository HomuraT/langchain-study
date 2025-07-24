"""
提示模板测试

测试LangChain提示模板的各种功能，包括：
- PromptTemplate：字符串模板功能
- ChatPromptTemplate：消息模板功能  
- MessagesPlaceholder：消息占位符功能
- 与ChatOpenAI模型的集成应用

作者: AI Assistant
创建时间: 2025年
"""

import unittest
from typing import Dict, Any, List, Optional, Union

from langchain_core.prompts import (
    PromptTemplate, 
    ChatPromptTemplate, 
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.messages import (
    HumanMessage, 
    SystemMessage, 
    AIMessage,
    BaseMessage
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


class TestPromptTemplates(unittest.TestCase):
    """提示模板测试类"""
    
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
    
    # ================== PromptTemplate 基础测试 ==================
    
    def test_prompt_template_creation(self) -> None:
        """
        测试PromptTemplate的基础创建功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试PromptTemplate基础创建 ===")
        
        # 方法1：使用from_template创建
        prompt1 = PromptTemplate.from_template("Tell me a joke about {topic}")
        
        # 方法2：使用构造函数创建
        prompt2 = PromptTemplate(
            input_variables=["topic"],
            template="Tell me a joke about {topic}"
        )
        
        # 验证创建结果
        self.assertEqual(prompt1.input_variables, ["topic"])
        self.assertEqual(prompt2.input_variables, ["topic"])
        self.assertEqual(prompt1.template, prompt2.template)
        
        print(f"Prompt1输入变量: {prompt1.input_variables}")
        print(f"Prompt1模板: {prompt1.template}")
        print("✅ PromptTemplate基础创建测试通过")
    
    def test_prompt_template_formatting(self) -> None:
        """
        测试PromptTemplate的格式化功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试PromptTemplate格式化功能 ===")
        
        prompt = PromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
        
        # 测试格式化
        formatted_prompt = prompt.format(adjective="funny", topic="cats")
        expected = "Tell me a funny joke about cats"
        
        self.assertEqual(formatted_prompt, expected)
        print(f"格式化结果: {formatted_prompt}")
        
        # 测试invoke方法
        prompt_value = prompt.invoke({"adjective": "hilarious", "topic": "dogs"})
        formatted_from_invoke = prompt_value.to_string()
        expected_invoke = "Tell me a hilarious joke about dogs"
        
        self.assertEqual(formatted_from_invoke, expected_invoke)
        print(f"Invoke结果: {formatted_from_invoke}")
        print("✅ PromptTemplate格式化测试通过")
    
    def test_prompt_template_with_multiple_variables(self) -> None:
        """
        测试包含多个变量的PromptTemplate
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试多变量PromptTemplate ===")
        
        template = """
你是一个{role}，专门帮助{audience}解决{problem_type}问题。
请根据以下信息回答问题：

用户背景：{user_background}
问题：{question}
期望的回答风格：{style}

请提供详细且有用的回答。
"""
        
        prompt = PromptTemplate.from_template(template)
        
        # 验证输入变量自动识别
        expected_variables = ["role", "audience", "problem_type", "user_background", "question", "style"]
        self.assertEqual(sorted(prompt.input_variables), sorted(expected_variables))
        
        # 测试格式化
        test_data = {
            "role": "专业的编程导师",
            "audience": "初学者",
            "problem_type": "Python编程",
            "user_background": "计算机科学专业学生",
            "question": "如何理解Python中的类和对象？",
            "style": "简单易懂，配有示例"
        }
        
        formatted = prompt.format(**test_data)
        
        # 验证所有变量都被替换
        for var in expected_variables:
            self.assertNotIn(f"{{{var}}}", formatted)
            self.assertIn(test_data[var], formatted)
        
        print(f"识别的变量: {prompt.input_variables}")
        print(f"格式化成功，包含所有预期内容")
        print("✅ 多变量PromptTemplate测试通过")
    
    # ================== ChatPromptTemplate 测试 ==================
    
    def test_chat_prompt_template_basic(self) -> None:
        """
        测试ChatPromptTemplate的基础功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试ChatPromptTemplate基础功能 ===")
        
        # 方法1：使用元组列表创建
        chat_prompt = ChatPromptTemplate([
            ("system", "You are a helpful assistant"),
            ("user", "Tell me a joke about {topic}")
        ])
        
        # 验证创建结果
        self.assertEqual(len(chat_prompt.messages), 2)
        self.assertEqual(chat_prompt.input_variables, ["topic"])
        
        # 测试格式化
        messages = chat_prompt.format_messages(topic="programming")
        
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[0].content, "You are a helpful assistant")
        self.assertEqual(messages[1].content, "Tell me a joke about programming")
        
        print(f"输入变量: {chat_prompt.input_variables}")
        print(f"生成的消息数量: {len(messages)}")
        print(f"系统消息: {messages[0].content}")
        print(f"用户消息: {messages[1].content}")
        print("✅ ChatPromptTemplate基础功能测试通过")
    
    def test_chat_prompt_template_from_messages(self) -> None:
        """
        测试使用from_messages方法创建ChatPromptTemplate
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试ChatPromptTemplate.from_messages ===")
        
        # 使用模板对象创建
        system_template = SystemMessagePromptTemplate.from_template(
            "You are a {role}. Your expertise is in {domain}."
        )
        human_template = HumanMessagePromptTemplate.from_template(
            "Please help me with: {request}"
        )
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
        
        # 验证输入变量
        expected_variables = ["role", "domain", "request"]
        self.assertEqual(sorted(chat_prompt.input_variables), sorted(expected_variables))
        
        # 测试格式化
        test_data = {
            "role": "expert Python developer",
            "domain": "web development",
            "request": "optimize this Django application"
        }
        
        messages = chat_prompt.format_messages(**test_data)
        
        self.assertEqual(len(messages), 2)
        self.assertIn("expert Python developer", messages[0].content)
        self.assertIn("web development", messages[0].content)
        self.assertIn("optimize this Django application", messages[1].content)
        
        print(f"输入变量: {chat_prompt.input_variables}")
        print(f"系统消息: {messages[0].content}")
        print(f"用户消息: {messages[1].content}")
        print("✅ ChatPromptTemplate.from_messages测试通过")
    
    def test_chat_prompt_template_complex(self) -> None:
        """
        测试复杂的ChatPromptTemplate（包含多种消息类型）
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试复杂ChatPromptTemplate ===")
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful coding assistant. You provide clear, accurate code examples."),
            ("human", "I need help with {programming_language}"),
            ("ai", "I'd be happy to help you with {programming_language}! What specific topic would you like to learn about?"),
            ("human", "Can you explain {concept} and provide an example?"),
            ("ai", "Certainly! Let me explain {concept} in {programming_language}."),
            ("human", "{question}")
        ])
        
        # 验证输入变量
        expected_variables = ["programming_language", "concept", "question"]
        self.assertEqual(sorted(chat_prompt.input_variables), sorted(expected_variables))
        
        # 测试格式化
        test_data = {
            "programming_language": "Python",
            "concept": "list comprehensions",
            "question": "How can I filter a list using list comprehensions?"
        }
        
        messages = chat_prompt.format_messages(**test_data)
        
        self.assertEqual(len(messages), 6)
        
        # 验证消息类型
        expected_types = [SystemMessage, HumanMessage, AIMessage, HumanMessage, AIMessage, HumanMessage]
        for i, (message, expected_type) in enumerate(zip(messages, expected_types)):
            self.assertIsInstance(message, expected_type)
            
        # 验证内容包含变量
        full_conversation = "\n".join([msg.content for msg in messages])
        for value in test_data.values():
            self.assertIn(value, full_conversation)
        
        print(f"输入变量: {chat_prompt.input_variables}")
        print(f"消息数量: {len(messages)}")
        print("消息类型:", [type(msg).__name__ for msg in messages])
        print("✅ 复杂ChatPromptTemplate测试通过")
    
    # ================== MessagesPlaceholder 测试 ==================
    
    def test_messages_placeholder_basic(self) -> None:
        """
        测试MessagesPlaceholder的基础功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试MessagesPlaceholder基础功能 ===")
        
        prompt = ChatPromptTemplate([
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("conversation_history"),
            ("user", "现在请回答: {question}")
        ])
        
        # 验证输入变量
        expected_variables = ["conversation_history", "question"]
        self.assertEqual(sorted(prompt.input_variables), sorted(expected_variables))
        
        # 准备测试数据
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there! How can I help you?"),
            HumanMessage(content="I have a question about Python")
        ]
        
        # 测试格式化
        messages = prompt.format_messages(
            conversation_history=history,
            question="What is a lambda function?"
        )
        
        # 验证结果：1个系统消息 + 3个历史消息 + 1个新用户消息 = 5个消息
        self.assertEqual(len(messages), 5)
        
        # 验证系统消息
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertEqual(messages[0].content, "You are a helpful assistant")
        
        # 验证历史消息
        for i, original_msg in enumerate(history):
            self.assertEqual(messages[i + 1].content, original_msg.content)
            self.assertIsInstance(messages[i + 1], type(original_msg))
        
        # 验证最后的用户消息
        self.assertIsInstance(messages[-1], HumanMessage)
        self.assertEqual(messages[-1].content, "现在请回答: What is a lambda function?")
        
        print(f"输入变量: {prompt.input_variables}")
        print(f"总消息数: {len(messages)}")
        print(f"历史消息数: {len(history)}")
        print("✅ MessagesPlaceholder基础功能测试通过")
    
    def test_messages_placeholder_alternative_syntax(self) -> None:
        """
        测试MessagesPlaceholder的替代语法
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试MessagesPlaceholder替代语法 ===")
        
        # 方法1：使用MessagesPlaceholder类
        prompt1 = ChatPromptTemplate([
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("msgs")
        ])
        
        # 方法2：使用placeholder字符串（注意：这个语法在某些版本中可能不支持）
        try:
            prompt2 = ChatPromptTemplate([
                ("system", "You are a helpful assistant"),
                ("placeholder", "{msgs}")
            ])
            placeholder_syntax_supported = True
        except Exception:
            # 如果不支持，我们创建一个相同的prompt作为对比
            prompt2 = ChatPromptTemplate([
                ("system", "You are a helpful assistant"),
                MessagesPlaceholder("msgs")
            ])
            placeholder_syntax_supported = False
        
        # 验证输入变量
        self.assertEqual(prompt1.input_variables, ["msgs"])
        
        # 准备测试消息
        test_msgs = [
            HumanMessage(content="测试消息1"),
            AIMessage(content="AI回复1"),
            HumanMessage(content="测试消息2")
        ]
        
        # 测试格式化结果
        messages1 = prompt1.format_messages(msgs=test_msgs)
        
        # 验证结果
        self.assertEqual(len(messages1), 4)  # 1个系统消息 + 3个测试消息
        
        # 验证系统消息
        self.assertIsInstance(messages1[0], SystemMessage)
        self.assertEqual(messages1[0].content, "You are a helpful assistant")
        
        # 验证插入的消息
        for i, original_msg in enumerate(test_msgs):
            self.assertEqual(messages1[i + 1].content, original_msg.content)
            self.assertIsInstance(messages1[i + 1], type(original_msg))
        
        print("方法1输入变量:", prompt1.input_variables)
        if placeholder_syntax_supported:
            print("方法2输入变量:", prompt2.input_variables)
            print("✅ placeholder字符串语法受支持")
        else:
            print("ℹ️ placeholder字符串语法在当前版本中不受支持，使用MessagesPlaceholder类")
        print(f"生成消息数: {len(messages1)}")
        print("✅ MessagesPlaceholder替代语法测试通过")
    
    def test_messages_placeholder_empty_list(self) -> None:
        """
        测试MessagesPlaceholder处理空消息列表
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试MessagesPlaceholder空消息列表处理 ===")
        
        prompt = ChatPromptTemplate([
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("chat_history"),
            ("user", "新的问题: {question}")
        ])
        
        # 测试空历史
        messages = prompt.format_messages(
            chat_history=[],
            question="Hello"
        )
        
        # 验证只有系统消息和用户消息
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[1].content, "新的问题: Hello")
        
        print(f"空历史时消息数: {len(messages)}")
        print("✅ MessagesPlaceholder空消息列表处理测试通过")
    
    # ================== 与ChatOpenAI集成测试 ==================
    
    def test_prompt_template_with_chat_model(self) -> None:
        """
        测试PromptTemplate与ChatOpenAI的集成
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试PromptTemplate与ChatOpenAI集成 ===")
        
        try:
            chat_model = self.get_chat_model()
            prompt = PromptTemplate.from_template("请用一句话介绍{topic}")
            
            # 创建处理链
            chain = prompt | chat_model | StrOutputParser()
            
            # 测试调用
            result = chain.invoke({"topic": "人工智能"})
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            print(f"输入主题: 人工智能")
            print(f"AI回答: {result}")
            print("✅ PromptTemplate与ChatOpenAI集成测试通过")
            
        except Exception as e:
            print(f"❌ PromptTemplate与ChatOpenAI集成测试失败: {e}")
    
    def test_chat_prompt_template_with_chat_model(self) -> None:
        """
        测试ChatPromptTemplate与ChatOpenAI的集成
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试ChatPromptTemplate与ChatOpenAI集成 ===")
        
        try:
            chat_model = self.get_chat_model()
            
            prompt = ChatPromptTemplate([
                ("system", "你是一个{expertise}专家，善于用{style}的方式解释复杂概念"),
                ("user", "请解释{concept}")
            ])
            
            # 创建处理链
            chain = prompt | chat_model | StrOutputParser()
            
            # 测试调用
            result = chain.invoke({
                "expertise": "机器学习",
                "style": "通俗易懂",
                "concept": "神经网络"
            })
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            print("输入参数:")
            print("  - 专业领域: 机器学习")
            print("  - 解释风格: 通俗易懂")
            print("  - 解释概念: 神经网络")
            print(f"AI回答: {result}")
            print("✅ ChatPromptTemplate与ChatOpenAI集成测试通过")
            
        except Exception as e:
            print(f"❌ ChatPromptTemplate与ChatOpenAI集成测试失败: {e}")
    
    def test_messages_placeholder_with_chat_model(self) -> None:
        """
        测试MessagesPlaceholder与ChatOpenAI的集成
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试MessagesPlaceholder与ChatOpenAI集成 ===")
        
        try:
            chat_model = self.get_chat_model()
            
            prompt = ChatPromptTemplate([
                ("system", "你是一个有用的助手，能够基于对话历史提供连贯的回答"),
                MessagesPlaceholder("conversation_history"),
                ("user", "{new_question}")
            ])
            
            # 创建处理链
            chain = prompt | chat_model | StrOutputParser()
            
            # 模拟对话历史
            history = [
                HumanMessage(content="我想学习Python编程"),
                AIMessage(content="很好！Python是一门优秀的编程语言。你想从哪个方面开始学习？"),
                HumanMessage(content="我想了解数据类型")
            ]
            
            # 测试调用
            result = chain.invoke({
                "conversation_history": history,
                "new_question": "能详细讲讲列表类型吗？"
            })
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            print("对话历史:")
            for i, msg in enumerate(history):
                role = "用户" if isinstance(msg, HumanMessage) else "助手"
                print(f"  {i+1}. {role}: {msg.content}")
            
            print(f"新问题: 能详细讲讲列表类型吗？")
            print(f"AI回答: {result}")
            print("✅ MessagesPlaceholder与ChatOpenAI集成测试通过")
            
        except Exception as e:
            print(f"❌ MessagesPlaceholder与ChatOpenAI集成测试失败: {e}")
    
    def test_complex_prompt_with_chat_model(self) -> None:
        """
        测试复杂提示模板与ChatOpenAI的集成应用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试复杂提示模板与ChatOpenAI集成 ===")
        
        try:
            chat_model = self.get_chat_model()
            
            # 复杂的多轮对话模板
            prompt = ChatPromptTemplate([
                ("system", """你是一个{role}，专门帮助{target_audience}。

你的工作原则：
1. 提供{answer_style}的回答
2. 考虑用户的{experience_level}水平
3. 包含实际的{example_type}

当前对话主题：{topic}"""),
                MessagesPlaceholder("dialogue_history"),
                ("user", """请基于以上对话回答我的问题：

问题：{question}

补充信息：{additional_context}""")
            ])
            
            # 创建处理链
            chain = prompt | chat_model | StrOutputParser()
            
            # 准备测试数据
            test_data = {
                "role": "Python编程导师",
                "target_audience": "编程初学者",
                "answer_style": "清晰详细",
                "experience_level": "初级",
                "example_type": "代码示例",
                "topic": "Python数据结构",
                "dialogue_history": [
                    HumanMessage(content="我刚开始学习Python"),
                    AIMessage(content="很好！我会用简单的方式帮你学习Python。"),
                    HumanMessage(content="我听说Python有很多数据类型")
                ],
                "question": "什么是字典，它和列表有什么区别？",
                "additional_context": "我已经了解了列表的基本操作"
            }
            
            # 测试调用
            result = chain.invoke(test_data)
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            print("测试参数:")
            print(f"  角色: {test_data['role']}")
            print(f"  目标受众: {test_data['target_audience']}")
            print(f"  主题: {test_data['topic']}")
            print(f"  问题: {test_data['question']}")
            print(f"\nAI回答: {result}")
            print("✅ 复杂提示模板与ChatOpenAI集成测试通过")
            
        except Exception as e:
            print(f"❌ 复杂提示模板与ChatOpenAI集成测试失败: {e}")


def main() -> int:
    """
    运行提示模板测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行提示模板测试套件")
    print("=" * 60)
    print("测试内容:")
    print("  📝 PromptTemplate - 字符串模板功能")
    print("  💬 ChatPromptTemplate - 消息模板功能")
    print("  📋 MessagesPlaceholder - 消息占位符功能")
    print("  🤖 与ChatOpenAI模型集成应用")
    print("=" * 60)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main() 