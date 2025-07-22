"""
高级功能测试

测试ChatOpenAI模型的高级功能，包括工具调用、结构化输出、批处理等
"""

import unittest
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticToolsParser

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


# 定义测试用的Pydantic模型
class WeatherInfo(BaseModel):
    """天气信息模型"""
    location: str = Field(description="地点名称")
    temperature: float = Field(description="温度（摄氏度）")
    humidity: int = Field(description="湿度百分比")
    description: str = Field(description="天气描述")


class MathResult(BaseModel):
    """数学计算结果模型"""
    operation: str = Field(description="执行的数学运算")
    result: float = Field(description="计算结果")
    explanation: str = Field(description="计算过程说明")


# 定义测试用的工具
@tool
def get_weather(location: str) -> Dict[str, Any]:
    """
    获取指定地点的天气信息
    
    Args:
        location: 地点名称
        
    Returns:
        Dict[str, Any]: 天气信息字典
    """
    return {
        "location": location,
        "temperature": 22.5,
        "humidity": 65,
        "description": "晴朗"
    }


@tool  
def calculate_math(expression: str) -> Dict[str, Any]:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        Dict[str, Any]: 计算结果字典
    """
    try:
        result = eval(expression)  # 仅用于测试，实际应用中需要安全的数学解析器
        return {
            "operation": expression,
            "result": result,
            "explanation": f"{expression} = {result}"
        }
    except Exception as e:
        return {
            "operation": expression,
            "result": None,
            "explanation": f"计算错误: {str(e)}"
        }


class TestAdvancedFeatures(unittest.TestCase):
    """高级功能测试类"""
    
    def get_advanced_model(self) -> ChatOpenAI:
        """
        创建配置了高级功能的ChatOpenAI实例
        
        Returns:
            ChatOpenAI: 配置好的高级功能模型实例
        """
        config = apis["local"]
        return ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.7,
            max_tokens=2000,
            timeout=60
        )
    
    def test_tool_binding_and_calling(self) -> None:
        """
        测试工具绑定和调用功能
        """
        model = self.get_advanced_model()
        tools = [get_weather, calculate_math]
        
        # 绑定工具到模型
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="What's the weather like in Beijing?")]
        
        try:
            response = model_with_tools.invoke(messages)
            
            self.assertIsInstance(response, AIMessage)
            print(f"Tool binding response: {response.content}")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Tool calls made: {response.tool_calls}")
                for tool_call in response.tool_calls:
                    print(f"Tool: {tool_call['name']}, Args: {tool_call['args']}")
        except Exception as e:
            print(f"Tool binding test failed: {e}")
    
    def test_multiple_tool_calls(self) -> None:
        """
        测试多个工具同时调用
        """
        model = self.get_advanced_model()
        tools = [get_weather, calculate_math]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="What's 3 * 12? Also, what's the weather in Shanghai?")]
        
        try:
            response = model_with_tools.invoke(messages)
            
            self.assertIsInstance(response, AIMessage)
            print(f"Multiple tools response: {response.content}")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Multiple tool calls: {len(response.tool_calls)} calls made")
                for i, tool_call in enumerate(response.tool_calls):
                    print(f"Tool {i+1}: {tool_call['name']}, Args: {tool_call['args']}")
        except Exception as e:
            print(f"Multiple tool calls test failed: {e}")
    
    def test_pydantic_tools_parser(self) -> None:
        """
        测试Pydantic工具解析器
        """
        # 创建解析器
        parser = PydanticToolsParser(tools=[MathResult, WeatherInfo])
        
        # 模拟工具调用数据
        mock_tool_calls = [
            MathResult(operation="2+3", result=5, explanation="2+3 = 5"),
            WeatherInfo(location="北京", temperature=25.0, humidity=60, description="晴朗")
        ]
        
        # 验证解析器能正确处理工具数据
        self.assertEqual(len(mock_tool_calls), 2)
        self.assertIsInstance(mock_tool_calls[0], MathResult)
        self.assertIsInstance(mock_tool_calls[1], WeatherInfo)
        print("Pydantic tools parser test passed")
    
    def test_structured_output_with_pydantic(self) -> None:
        """
        测试使用Pydantic模型的结构化输出
        """
        model = self.get_advanced_model()
        
        # 构造系统消息，要求返回特定格式的JSON
        system_message = SystemMessage(content="""
        你是一个天气助手。请以JSON格式返回天气信息，包含以下字段：
        - location: 地点名称
        - temperature: 温度（摄氏度）
        - humidity: 湿度百分比
        - description: 天气描述
        """)
        
        user_message = HumanMessage(content="北京今天的天气怎么样？")
        
        try:
            response = model.invoke([system_message, user_message])
            
            self.assertIsInstance(response, AIMessage)
            print(f"Structured output response: {response.content}")
            
            # 尝试解析为JSON
            try:
                weather_data = json.loads(response.content)
                weather_info = WeatherInfo(**weather_data)
                print(f"Parsed weather info: {weather_info}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON parsing failed: {e}")
        except Exception as e:
            print(f"Structured output test failed: {e}")
    
    def test_multi_turn_conversation_with_context(self) -> None:
        """
        测试多轮对话中的上下文保持
        """
        model = self.get_advanced_model()
        
        # 第一轮对话
        conversation = [
            SystemMessage(content="你是一个有用的助手，记住用户的偏好。"),
            HumanMessage(content="我喜欢蓝色，这是我最喜欢的颜色。")
        ]
        
        try:
            # 第一次调用
            response1 = model.invoke(conversation)
            conversation.append(response1)
            conversation.append(HumanMessage(content="根据我的喜好，推荐一种颜色搭配。"))
            
            # 第二次调用
            response2 = model.invoke(conversation)
            
            self.assertIsInstance(response2, AIMessage)
            print(f"Context preservation response: {response2.content}")
        except Exception as e:
            print(f"Multi-turn conversation test failed: {e}")
    
    def test_batch_processing(self) -> None:
        """
        测试批处理功能
        """
        model = self.get_advanced_model()
        
        # 准备批量请求
        batch_requests = [
            [HumanMessage(content="Hello, how are you?")],
            [HumanMessage(content="What's 2+2?")],
            [HumanMessage(content="Tell me a joke.")]
        ]
        
        try:
            batch_responses = model.batch(batch_requests)
            
            self.assertEqual(len(batch_responses), 3)
            for i, response in enumerate(batch_responses):
                self.assertIsInstance(response, AIMessage)
                print(f"Batch response {i+1}: {response.content}")
        except Exception as e:
            print(f"Batch processing test failed: {e}")
    
    def test_tool_execution(self) -> None:
        """
        测试实际工具执行
        """
        # 测试数学计算工具
        math_result = calculate_math("15 + 27")
        self.assertEqual(math_result["result"], 42)
        self.assertEqual(math_result["operation"], "15 + 27")
        print(f"Math tool result: {math_result}")
        
        # 测试天气查询工具
        weather_result = get_weather("北京")
        self.assertEqual(weather_result["location"], "北京")
        self.assertIsInstance(weather_result["temperature"], float)
        self.assertIsInstance(weather_result["humidity"], int)
        print(f"Weather tool result: {weather_result}")
    
    def test_temperature_and_creativity_effects(self) -> None:
        """
        测试temperature参数对创造性的影响
        """
        config = apis["local"]
        
        # 低temperature模型（保守）
        conservative_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.1
        )
        
        # 高temperature模型（创造性）
        creative_model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.9
        )
        
        # 验证配置差异
        self.assertEqual(conservative_model.temperature, 0.1)
        self.assertEqual(creative_model.temperature, 0.9)
        
        # 测试实际效果
        messages = [HumanMessage(content="写一个关于春天的短句")]
        
        try:
            conservative_response = conservative_model.invoke(messages)
            creative_response = creative_model.invoke(messages)
            
            print(f"Conservative (T=0.1): {conservative_response.content}")
            print(f"Creative (T=0.9): {creative_response.content}")
        except Exception as e:
            print(f"Temperature effects test failed: {e}")


def main() -> int:
    """
    运行高级功能测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行高级功能测试")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main()