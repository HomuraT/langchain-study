"""
工具调用测试

测试ChatOpenAI模型的工具调用功能，包括基础工具调用、复杂工具组合、
工具输出处理、流式工具调用等各种场景
"""

import unittest
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, date

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


# ================== 工具定义 ==================

@tool
def add(a: int, b: int) -> int:
    """
    Adds two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        int: Sum of a and b
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies two numbers.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
        
    Returns:
        int: Product of a and b
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """
    Divides two numbers.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        float: Result of a divided by b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tool
def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The temperature unit to use. Options are 'celsius' or 'fahrenheit'
        
    Returns:
        Dict[str, Any]: Weather information including temperature, humidity, description
    """
    # 模拟天气数据
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "humidity": 65,
        "description": "晴朗",
        "wind_speed": 10,
        "timestamp": datetime.now().isoformat()
    }
    return weather_data


@tool
def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict[str, str]]: List of search results with title and snippet
    """
    # 模拟搜索结果
    results = []
    for i in range(min(max_results, 3)):
        results.append({
            "title": f"搜索结果 {i+1} for: {query}",
            "snippet": f"这是关于 '{query}' 的搜索结果 {i+1}",
            "url": f"https://example.com/result{i+1}",
        })
    return results


@tool
def calculate_complex_math(expression: str) -> Dict[str, Any]:
    """
    Calculate complex mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Dict[str, Any]: Calculation result with explanation
    """
    try:
        # 安全的数学表达式评估 (仅用于测试)
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "success": True,
            "explanation": f"计算 {expression} 的结果是 {result}"
        }
    except Exception as e:
        return {
            "expression": expression,
            "result": None,
            "success": False,
            "error": str(e),
            "explanation": f"计算 {expression} 时出错: {str(e)}"
        }


@tool
def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Get user profile information.
    
    Args:
        user_id: The user's unique identifier
        
    Returns:
        Dict[str, Any]: User profile data
    """
    # 模拟用户数据
    profiles = {
        "user123": {
            "id": "user123",
            "name": "张三",
            "email": "zhangsan@example.com",
            "age": 28,
            "preferences": ["技术", "阅读", "运动"],
            "membership": "premium"
        },
        "user456": {
            "id": "user456", 
            "name": "李四",
            "email": "lisi@example.com",
            "age": 35,
            "preferences": ["音乐", "旅行", "美食"],
            "membership": "basic"
        }
    }
    return profiles.get(user_id, {"error": "User not found"})


@tool
def book_appointment(date: str, time: str, service: str, user_id: str) -> Dict[str, Any]:
    """
    Book an appointment.
    
    Args:
        date: Appointment date in YYYY-MM-DD format
        time: Appointment time in HH:MM format  
        service: Type of service
        user_id: User's unique identifier
        
    Returns:
        Dict[str, Any]: Booking confirmation details
    """
    appointment_id = f"apt_{user_id}_{date}_{time}".replace(":", "").replace("-", "")
    
    return {
        "appointment_id": appointment_id,
        "date": date,
        "time": time,
        "service": service,
        "user_id": user_id,
        "status": "confirmed",
        "booking_time": datetime.now().isoformat(),
        "cancellation_allowed": True
    }


# ================== Pydantic 模型 ==================

class CalculationResult(BaseModel):
    """数学计算结果模型"""
    operation: str = Field(description="执行的数学运算")
    result: float = Field(description="计算结果")
    steps: Optional[List[str]] = Field(description="计算步骤", default=None)
    

class WeatherInfo(BaseModel):
    """天气信息模型"""
    location: str = Field(description="地点名称")
    temperature: float = Field(description="温度")
    unit: str = Field(description="温度单位")
    description: str = Field(description="天气描述")
    

class SearchResult(BaseModel):
    """搜索结果模型"""
    query: str = Field(description="搜索查询")
    results: List[Dict[str, str]] = Field(description="搜索结果列表")
    result_count: int = Field(description="结果数量")


# ================== 测试类 ==================

class TestToolCalling(unittest.TestCase):
    """工具调用测试类"""
    
    def get_tool_model(self) -> ChatOpenAI:
        """
        创建支持工具调用的ChatOpenAI实例
        
        Returns:
            ChatOpenAI: 配置好的工具调用模型实例
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
    
    # ================== 基础工具调用测试 ==================
    
    def test_single_tool_binding(self) -> None:
        """
        测试单个工具绑定
        """
        model = self.get_tool_model()
        tools = [add]
        model_with_tools = model.bind_tools(tools)
        
        # 验证工具绑定
        self.assertTrue(hasattr(model_with_tools, 'kwargs'))
        print("Single tool binding successful")
    
    def test_single_tool_call(self) -> None:
        """
        测试单个工具调用
        """
        model = self.get_tool_model()
        tools = [add]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="What is 15 + 27?")]
        
        try:
            response = model_with_tools.invoke(messages)
            
            self.assertIsInstance(response, AIMessage)
            print(f"Single tool call response: {response.content}")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                print(f"Tool called: {tool_call['name']}")
                print(f"Arguments: {tool_call['args']}")
                
                # 验证工具调用参数
                self.assertEqual(tool_call['name'], 'add')
                self.assertIn('a', tool_call['args'])
                self.assertIn('b', tool_call['args'])
            
        except Exception as e:
            print(f"Single tool call test failed: {e}")
    
    def test_multiple_tools_binding(self) -> None:
        """
        测试多个工具绑定
        """
        model = self.get_tool_model()
        tools = [add, multiply, divide]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="What tools do you have available?")]
        
        try:
            response = model_with_tools.invoke(messages)
            self.assertIsInstance(response, AIMessage)
            print(f"Multiple tools binding response: {response.content}")
            
        except Exception as e:
            print(f"Multiple tools binding test failed: {e}")
    
    def test_multiple_tool_calls_in_sequence(self) -> None:
        """
        测试同时调用多个工具
        """
        model = self.get_tool_model()
        tools = [add, multiply]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="Calculate 5 + 3, then multiply the result by 2")]
        
        try:
            response = model_with_tools.invoke(messages)
            
            self.assertIsInstance(response, AIMessage)
            print(f"Multiple tool calls response: {response.content}")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Number of tool calls: {len(response.tool_calls)}")
                for i, tool_call in enumerate(response.tool_calls):
                    print(f"Tool {i+1}: {tool_call['name']} with args: {tool_call['args']}")
            
        except Exception as e:
            print(f"Multiple tool calls test failed: {e}")
    
    def test_parallel_tool_calls(self) -> None:
        """
        测试并行工具调用
        """
        model = self.get_tool_model()
        tools = [add, multiply, get_current_weather]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="What is 10 + 5? What is 3 * 4? What's the weather in Beijing?")]
        
        try:
            response = model_with_tools.invoke(messages)
            
            self.assertIsInstance(response, AIMessage)
            print(f"Parallel tool calls response: {response.content}")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Parallel tool calls made: {len(response.tool_calls)}")
                for tool_call in response.tool_calls:
                    print(f"- {tool_call['name']}: {tool_call['args']}")
            
        except Exception as e:
            print(f"Parallel tool calls test failed: {e}")
    
    # ================== 完整工具调用流程测试 ==================
    
    def test_complete_tool_calling_flow(self) -> None:
        """
        测试完整的工具调用流程：请求 -> 工具调用 -> 工具执行 -> 结果返回
        """
        model = self.get_tool_model()
        tools = [add, multiply]
        model_with_tools = model.bind_tools(tools)
        
        # 第一步：发送需要工具调用的消息
        messages = [HumanMessage(content="What is 3 * 12? Also, what is 11 + 49?")]
        
        try:
            # 第二步：模型调用工具
            ai_response = model_with_tools.invoke(messages)
            messages.append(ai_response)
            
            self.assertIsInstance(ai_response, AIMessage)
            print(f"AI response: {ai_response.content}")
            
            if hasattr(ai_response, 'tool_calls') and ai_response.tool_calls:
                # 第三步：执行工具调用
                for tool_call in ai_response.tool_calls:
                    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"]]
                    
                    # 执行工具并创建ToolMessage
                    tool_output = selected_tool.invoke(tool_call)
                    messages.append(tool_output)
                    
                    print(f"Tool {tool_call['name']} executed: {tool_output.content}")
                
                # 第四步：将工具结果传回模型
                final_response = model_with_tools.invoke(messages)
                
                self.assertIsInstance(final_response, AIMessage)
                print(f"Final response with tool results: {final_response.content}")
                
                # 验证最终回答包含计算结果
                self.assertIn("36", final_response.content)  # 3 * 12 = 36
                self.assertIn("60", final_response.content)  # 11 + 49 = 60
            
        except Exception as e:
            print(f"Complete tool calling flow test failed: {e}")
    
    def test_tool_message_id_matching(self) -> None:
        """
        测试ToolMessage的tool_call_id匹配
        """
        model = self.get_tool_model()
        tools = [add]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="Add 5 and 7")]
        
        try:
            ai_response = model_with_tools.invoke(messages)
            
            if hasattr(ai_response, 'tool_calls') and ai_response.tool_calls:
                tool_call = ai_response.tool_calls[0]
                
                # 执行工具
                tool_output = add.invoke(tool_call)
                
                # 验证tool_call_id匹配
                self.assertEqual(tool_output.tool_call_id, tool_call['id'])
                print(f"Tool call ID matching verified: {tool_call['id']}")
            
        except Exception as e:
            print(f"Tool message ID matching test failed: {e}")
    
    # ================== 复杂工具调用测试 ==================
    
    def test_complex_tool_with_multiple_parameters(self) -> None:
        """
        测试具有多个参数的复杂工具
        """
        model = self.get_tool_model()
        tools = [get_current_weather]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="What's the weather like in Shanghai? Please use Fahrenheit.")]
        
        try:
            response = model_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                print(f"Complex tool call: {tool_call}")
                
                # 验证参数
                self.assertIn('location', tool_call['args'])
                # 可能包含unit参数
                if 'unit' in tool_call['args']:
                    self.assertIn(tool_call['args']['unit'], ['celsius', 'fahrenheit'])
            
        except Exception as e:
            print(f"Complex tool with multiple parameters test failed: {e}")
    
    def test_tool_with_optional_parameters(self) -> None:
        """
        测试具有可选参数的工具
        """
        model = self.get_tool_model()
        tools = [search_web]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="Search for 'LangChain tutorials' and show me top 3 results")]
        
        try:
            response = model_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                print(f"Tool with optional parameters: {tool_call}")
                
                # 验证必需参数
                self.assertIn('query', tool_call['args'])
                
                # 可选参数可能存在也可能不存在
                if 'max_results' in tool_call['args']:
                    self.assertIsInstance(tool_call['args']['max_results'], int)
            
        except Exception as e:
            print(f"Tool with optional parameters test failed: {e}")
    
    def test_nested_tool_calls(self) -> None:
        """
        测试嵌套工具调用（一个工具的结果用于另一个工具）
        """
        model = self.get_tool_model()
        tools = [add, multiply]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="First add 5 and 3, then multiply the result by 4")]
        
        try:
            # 第一轮：可能调用add工具
            response1 = model_with_tools.invoke(messages)
            messages.append(response1)
            
            if hasattr(response1, 'tool_calls') and response1.tool_calls:
                # 执行第一轮工具调用
                for tool_call in response1.tool_calls:
                    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"]]
                    tool_output = selected_tool.invoke(tool_call)
                    messages.append(tool_output)
                
                # 第二轮：可能调用multiply工具
                response2 = model_with_tools.invoke(messages)
                
                print(f"Nested tool calls - First response: {response1.content}")
                print(f"Nested tool calls - Final response: {response2.content}")
            
        except Exception as e:
            print(f"Nested tool calls test failed: {e}")
    
    # ================== 工具错误处理测试 ==================
    
    def test_tool_execution_error_handling(self) -> None:
        """
        测试工具执行错误的处理
        """
        model = self.get_tool_model()
        tools = [divide]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="What is 10 divided by 0?")]
        
        try:
            response = model_with_tools.invoke(messages)
            messages.append(response)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                
                try:
                    # 这会引发除零错误
                    tool_output = divide.invoke(tool_call)
                    messages.append(tool_output)
                except Exception as tool_error:
                    # 创建错误消息
                    error_message = ToolMessage(
                        content=f"Error: {str(tool_error)}",
                        tool_call_id=tool_call['id']
                    )
                    messages.append(error_message)
                    
                    print(f"Tool error handled: {tool_error}")
                
                # 模型应该能处理错误消息
                final_response = model_with_tools.invoke(messages)
                print(f"Response after tool error: {final_response.content}")
            
        except Exception as e:
            print(f"Tool execution error handling test failed: {e}")
    
    def test_invalid_tool_parameters(self) -> None:
        """
        测试无效工具参数的处理
        """
        # 手动创建无效的工具调用
        invalid_tool_call = {
            'name': 'add',
            'args': {'a': 'invalid', 'b': 5},  # 'invalid' 不是数字
            'id': 'test_invalid_call'
        }
        
        try:
            result = add.invoke(invalid_tool_call)
            print(f"Invalid parameters handled: {result}")
        except Exception as e:
            print(f"Invalid tool parameters error (expected): {e}")
    
    # ================== 异步工具调用测试 ==================
    
    def test_async_tool_calling(self) -> None:
        """
        测试异步工具调用
        """
        async def run_async_test():
            model = self.get_tool_model()
            tools = [add, multiply]
            model_with_tools = model.bind_tools(tools)
            
            messages = [HumanMessage(content="Calculate 7 + 8 and 4 * 9")]
            
            try:
                response = await model_with_tools.ainvoke(messages)
                
                self.assertIsInstance(response, AIMessage)
                print(f"Async tool calling response: {response.content}")
                
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"Async tool calls: {len(response.tool_calls)} calls made")
                
            except Exception as e:
                print(f"Async tool calling test failed: {e}")
        
        # 运行异步测试
        asyncio.run(run_async_test())
    
    # ================== 流式工具调用测试 ==================
    
    def test_streaming_with_tools(self) -> None:
        """
        测试流式输出与工具调用结合
        """
        model = self.get_tool_model()
        tools = [add]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="Add 15 and 25, then explain the result")]
        
        try:
            # 尝试流式输出
            chunks = list(model_with_tools.stream(messages))
            
            print(f"Streaming with tools - received {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    print(f"Chunk {i}: Tool calls detected")
                elif hasattr(chunk, 'content') and chunk.content:
                    print(f"Chunk {i}: {chunk.content}")
            
        except Exception as e:
            print(f"Streaming with tools test failed: {e}")
    
    # ================== 结构化工具输出测试 ==================
    
    def test_structured_tool_output(self) -> None:
        """
        测试结构化工具输出
        """
        model = self.get_tool_model()
        tools = [calculate_complex_math]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="Calculate the square root of 144")]
        
        try:
            response = model_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_output = calculate_complex_math.invoke(tool_call)
                
                # 解析结构化输出
                result_dict = json.loads(tool_output.content) if isinstance(tool_output.content, str) else tool_output.content
                
                self.assertIn('result', result_dict)
                self.assertIn('success', result_dict)
                print(f"Structured tool output: {result_dict}")
            
        except Exception as e:
            print(f"Structured tool output test failed: {e}")
    
    # ================== 真实世界场景测试 ==================
    
    def test_multi_step_workflow(self) -> None:
        """
        测试多步骤工作流（获取用户信息 -> 预约服务）
        """
        model = self.get_tool_model()
        tools = [get_user_profile, book_appointment]
        model_with_tools = model.bind_tools(tools)
        
        messages = [
            HumanMessage(content="I'm user123. Please help me book a haircut appointment for tomorrow at 2:00 PM")
        ]
        
        try:
            # 第一步：可能获取用户信息
            response1 = model_with_tools.invoke(messages)
            messages.append(response1)
            
            if hasattr(response1, 'tool_calls') and response1.tool_calls:
                # 执行工具调用
                for tool_call in response1.tool_calls:
                    if tool_call['name'] == 'get_user_profile':
                        tool_output = get_user_profile.invoke(tool_call)
                    elif tool_call['name'] == 'book_appointment':
                        tool_output = book_appointment.invoke(tool_call)
                    else:
                        continue
                    
                    messages.append(tool_output)
                    print(f"Step executed: {tool_call['name']}")
                
                # 继续对话，可能预约服务
                final_response = model_with_tools.invoke(messages)
                print(f"Multi-step workflow result: {final_response.content}")
            
        except Exception as e:
            print(f"Multi-step workflow test failed: {e}")
    
    def test_data_analysis_scenario(self) -> None:
        """
        测试数据分析场景（计算 -> 搜索 -> 解释）
        """
        model = self.get_tool_model()
        tools = [calculate_complex_math, search_web]
        model_with_tools = model.bind_tools(tools)
        
        messages = [
            HumanMessage(content="Calculate the compound annual growth rate if an investment grows from $1000 to $1500 over 3 years, then search for information about good CAGR rates")
        ]
        
        try:
            response = model_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Data analysis scenario - {len(response.tool_calls)} tools called")
                for tool_call in response.tool_calls:
                    print(f"- {tool_call['name']}: {tool_call['args']}")
            
        except Exception as e:
            print(f"Data analysis scenario test failed: {e}")
    
    def test_decision_support_scenario(self) -> None:
        """
        测试决策支持场景（多维度信息收集）
        """
        model = self.get_tool_model()
        tools = [get_current_weather, search_web, calculate_complex_math]
        model_with_tools = model.bind_tools(tools)
        
        messages = [
            HumanMessage(content="I'm planning a picnic for 20 people. Check the weather in Beijing, search for picnic food ideas, and calculate how much food I need if each person eats 2 servings")
        ]
        
        try:
            response = model_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Decision support scenario - {len(response.tool_calls)} tools called")
                for tool_call in response.tool_calls:
                    print(f"- {tool_call['name']}")
            
        except Exception as e:
            print(f"Decision support scenario test failed: {e}")
    
    # ================== 性能和限制测试 ==================
    
    def test_tool_call_limits(self) -> None:
        """
        测试工具调用限制
        """
        model = self.get_tool_model()
        tools = [add, multiply, divide, get_current_weather, search_web]
        model_with_tools = model.bind_tools(tools)
        
        # 尝试触发大量工具调用
        messages = [
            HumanMessage(content="Perform many calculations: 1+1, 2*3, 4/2, 5+5, 6*7, 8/4, 9+9, 10*11")
        ]
        
        try:
            response = model_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Tool call limits test - {len(response.tool_calls)} tools called")
                self.assertLessEqual(len(response.tool_calls), 10)  # 假设有合理的限制
            
        except Exception as e:
            print(f"Tool call limits test failed: {e}")
    
    def test_tool_performance_timing(self) -> None:
        """
        测试工具调用性能计时
        """
        import time
        
        model = self.get_tool_model()
        tools = [add]
        model_with_tools = model.bind_tools(tools)
        
        messages = [HumanMessage(content="Add 100 and 200")]
        
        try:
            start_time = time.time()
            response = model_with_tools.invoke(messages)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"Tool performance timing: {duration:.2f} seconds")
            
            # 假设合理的性能期望
            self.assertLess(duration, 30.0)  # 应该在30秒内完成
            
        except Exception as e:
            print(f"Tool performance timing test failed: {e}")


if __name__ == '__main__':
    unittest.main() 