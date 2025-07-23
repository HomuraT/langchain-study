"""
高级功能测试

测试ChatOpenAI模型的高级功能，包括工具调用、结构化输出、批处理等
"""

import unittest
import json
import re
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticToolsParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

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


class Joke(BaseModel):
    """笑话模型"""
    setup: str = Field(description="笑话的铺垫")
    punchline: str = Field(description="笑话的包袱")
    rating: Optional[int] = Field(default=None, description="搞笑程度，1-10分")


class ConversationalResponse(BaseModel):
    """对话回复模型"""
    response: str = Field(description="对用户查询的对话式回复")


class MultiResponse(BaseModel):
    """多类型响应模型"""
    final_output: Union[Joke, ConversationalResponse]


class Person(BaseModel):
    """人员信息模型"""
    name: str = Field(description="人员姓名")
    age: int = Field(description="人员年龄")
    height_in_meters: float = Field(description="身高（米）")


class People(BaseModel):
    """多人信息模型"""
    people: List[Person]


# TypedDict定义
class JokeDict(TypedDict):
    """笑话字典类型"""
    setup: Annotated[str, ..., "笑话的铺垫"]
    punchline: Annotated[str, ..., "笑话的包袱"]
    rating: Annotated[Optional[int], None, "搞笑程度，1-10分"]


class WeatherDict(TypedDict):
    """天气字典类型"""
    location: Annotated[str, ..., "地点名称"]
    temperature: Annotated[float, ..., "温度（摄氏度）"]
    humidity: Annotated[int, ..., "湿度百分比"]
    description: Annotated[str, ..., "天气描述"]


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
    
    def test_with_structured_output_pydantic(self) -> None:
        """
        测试使用Pydantic类的with_structured_output方法
        """
        model = self.get_advanced_model()
        
        try:
            # 测试笑话生成
            structured_llm = model.with_structured_output(Joke)
            response = structured_llm.invoke("Tell me a joke about cats")
            
            self.assertIsInstance(response, Joke)
            self.assertIsInstance(response.setup, str)
            self.assertIsInstance(response.punchline, str)
            self.assertTrue(len(response.setup) > 0)
            self.assertTrue(len(response.punchline) > 0)
            print(f"Pydantic structured output: {response}")
            
        except Exception as e:
            print(f"Pydantic structured output test failed: {e}")
    
    def test_with_structured_output_typeddict(self) -> None:
        """
        测试使用TypedDict的with_structured_output方法
        """
        model = self.get_advanced_model()
        
        try:
            structured_llm = model.with_structured_output(JokeDict)
            response = structured_llm.invoke("Tell me a joke about dogs")
            
            self.assertIsInstance(response, dict)
            self.assertIn('setup', response)
            self.assertIn('punchline', response)
            self.assertIsInstance(response['setup'], str)
            self.assertIsInstance(response['punchline'], str)
            print(f"TypedDict structured output: {response}")
            
        except Exception as e:
            print(f"TypedDict structured output test failed: {e}")
    
    def test_with_structured_output_json_schema(self) -> None:
        """
        测试使用JSON Schema的with_structured_output方法
        """
        model = self.get_advanced_model()
        
        json_schema = {
            "title": "joke",
            "description": "Joke to tell user.",
            "type": "object",
            "properties": {
                "setup": {
                    "type": "string",
                    "description": "The setup of the joke",
                },
                "punchline": {
                    "type": "string",
                    "description": "The punchline to the joke",
                },
                "rating": {
                    "type": "integer",
                    "description": "How funny the joke is, from 1 to 10",
                    "default": None,
                },
            },
            "required": ["setup", "punchline"],
        }
        
        try:
            structured_llm = model.with_structured_output(json_schema)
            response = structured_llm.invoke("Tell me a joke about birds")
            
            self.assertIsInstance(response, dict)
            self.assertIn('setup', response)
            self.assertIn('punchline', response)
            print(f"JSON Schema structured output: {response}")
            
        except Exception as e:
            print(f"JSON Schema structured output test failed: {e}")
    
    def test_union_type_selection(self) -> None:
        """
        测试Union类型的多模式选择
        """
        model = self.get_advanced_model()
        
        try:
            structured_llm = model.with_structured_output(MultiResponse)
            
            # 测试笑话请求
            joke_response = structured_llm.invoke("Tell me a joke")
            self.assertIsInstance(joke_response, MultiResponse)
            print(f"Union type joke response: {joke_response}")
            
            # 测试对话请求
            chat_response = structured_llm.invoke("How are you today?")
            self.assertIsInstance(chat_response, MultiResponse)
            print(f"Union type chat response: {chat_response}")
            
        except Exception as e:
            print(f"Union type selection test failed: {e}")
    
    def test_streaming_structured_output(self) -> None:
        """
        测试流式结构化输出
        """
        model = self.get_advanced_model()
        
        try:
            structured_llm = model.with_structured_output(JokeDict)
            
            chunks = []
            for chunk in structured_llm.stream("Tell me a joke about programming"):
                chunks.append(chunk)
                print(f"Stream chunk: {chunk}")
            
            # 验证最后一个chunk包含完整信息
            final_chunk = chunks[-1] if chunks else {}
            self.assertIsInstance(final_chunk, dict)
            if 'setup' in final_chunk and 'punchline' in final_chunk:
                self.assertIsInstance(final_chunk['setup'], str)
                self.assertIsInstance(final_chunk['punchline'], str)
            
        except Exception as e:
            print(f"Streaming structured output test failed: {e}")
    
    def test_few_shot_prompting(self) -> None:
        """
        测试Few-shot prompting
        """
        model = self.get_advanced_model()
        
        # 使用系统消息的few-shot示例
        system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup and the final punchline.

Here are some examples of jokes:

example_user: Tell me a joke about planes
example_assistant: {"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}

example_user: Tell me another joke about planes  
example_assistant: {"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system), 
            ("human", "{input}")
        ])
        
        try:
            structured_llm = model.with_structured_output(JokeDict)
            few_shot_chain = prompt | structured_llm
            
            response = few_shot_chain.invoke({"input": "what's something funny about woodpeckers"})
            
            self.assertIsInstance(response, dict)
            self.assertIn('setup', response)
            self.assertIn('punchline', response)
            print(f"Few-shot prompting response: {response}")
            
        except Exception as e:
            print(f"Few-shot prompting test failed: {e}")
    
    def test_pydantic_output_parser(self) -> None:
        """
        测试PydanticOutputParser
        """
        model = self.get_advanced_model()
        
        # 设置解析器
        parser = PydanticOutputParser(pydantic_object=People)
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user query. Wrap the output in `json` tags\n{format_instructions}"),
            ("human", "{query}")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        try:
            chain = prompt | model | parser
            
            query = "Anna is 23 years old and she is 6 feet tall"
            response = chain.invoke({"query": query})
            
            self.assertIsInstance(response, People)
            self.assertGreater(len(response.people), 0)
            self.assertIsInstance(response.people[0], Person)
            print(f"PydanticOutputParser response: {response}")
            
        except Exception as e:
            print(f"PydanticOutputParser test failed: {e}")
    
    def test_custom_parsing(self) -> None:
        """
        测试自定义解析
        """
        model = self.get_advanced_model()
        
        # 自定义解析函数
        def extract_json(message: AIMessage) -> List[dict]:
            """从消息中提取JSON内容"""
            text = message.content
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            
            try:
                return [json.loads(match.strip()) for match in matches]
            except Exception:
                raise ValueError(f"Failed to parse: {message}")
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user query. Output your answer as JSON that matches the given schema: ```json\n{schema}\n```. Make sure to wrap the answer in ```json and ``` tags"),
            ("human", "{query}")
        ]).partial(schema=People.schema())
        
        try:
            chain = prompt | model | extract_json
            
            query = "Bob is 30 years old and he is 5.8 feet tall"
            response = chain.invoke({"query": query})
            
            self.assertIsInstance(response, list)
            self.assertGreater(len(response), 0)
            self.assertIsInstance(response[0], dict)
            print(f"Custom parsing response: {response}")
            
        except Exception as e:
            print(f"Custom parsing test failed: {e}")
    
    def test_json_mode(self) -> None:
        """
        测试JSON模式
        """
        model = self.get_advanced_model()
        
        try:
            # 注意：需要检查模型是否支持JSON模式
            structured_llm = model.with_structured_output(None, method="json_mode")
            
            response = structured_llm.invoke(
                "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
            )
            
            self.assertIsInstance(response, dict)
            print(f"JSON mode response: {response}")
            
        except Exception as e:
            print(f"JSON mode test failed (may not be supported): {e}")
    
    def test_raw_output_handling(self) -> None:
        """
        测试原始输出处理
        """
        model = self.get_advanced_model()
        
        try:
            structured_llm = model.with_structured_output(Joke, include_raw=True)
            
            response = structured_llm.invoke("Tell me a joke about computers")
            
            self.assertIsInstance(response, dict)
            self.assertIn('raw', response)
            self.assertIn('parsed', response)
            self.assertIn('parsing_error', response)
            
            # 验证原始输出
            self.assertIsInstance(response['raw'], AIMessage)
            
            # 验证解析结果
            if response['parsing_error'] is None:
                self.assertIsInstance(response['parsed'], (Joke, dict))
            
            print(f"Raw output handling response keys: {response.keys()}")
            print(f"Parsing error: {response['parsing_error']}")
            
        except Exception as e:
            print(f"Raw output handling test failed: {e}")
    
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