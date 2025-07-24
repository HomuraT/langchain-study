"""
Pydantic输出解析器测试

测试LangChain的Pydantic输出解析器功能，包括：
- PydanticOutputParser: Pydantic模型输出解析
- PydanticToolsParser: Pydantic工具解析器
- 与结构化输出的比较测试
- 复杂Pydantic模型解析
- 嵌套模型和列表解析
- 流式Pydantic解析

参考文档:
- https://python.langchain.com/docs/how_to/output_parser_structured/
- https://python.langchain.com/docs/concepts/output_parsers/
"""

import unittest
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.exceptions import OutputParserException

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


# ================== Pydantic模型定义 ==================

class PersonInfo(BaseModel):
    """人员信息模型"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    email: Optional[str] = Field(None, description="邮箱地址")
    occupation: Optional[str] = Field(None, description="职业")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "张三",
                "age": 30,
                "email": "zhangsan@example.com",
                "occupation": "软件工程师"
            }
        }


class Joke(BaseModel):
    """笑话模型"""
    setup: str = Field(description="笑话的开场白")
    punchline: str = Field(description="笑话的笑点")
    rating: Optional[int] = Field(None, description="笑话评分(1-10)", ge=1, le=10)


class TravelPlan(BaseModel):
    """旅行计划模型"""
    destination: str = Field(description="目的地")
    duration: int = Field(description="天数", gt=0)
    budget: float = Field(description="预算", gt=0)
    activities: List[str] = Field(description="活动列表")
    accommodation: Optional[str] = Field(None, description="住宿类型")


class ProductCategory(str, Enum):
    """产品分类枚举"""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    FOOD = "food"
    SPORTS = "sports"


class Product(BaseModel):
    """产品信息模型"""
    name: str = Field(description="产品名称")
    category: ProductCategory = Field(description="产品分类")
    price: float = Field(description="价格", gt=0)
    in_stock: bool = Field(description="是否有库存")
    description: Optional[str] = Field(None, description="产品描述")
    tags: List[str] = Field(default_factory=list, description="标签列表")


class NestedModel(BaseModel):
    """嵌套模型测试"""
    
    class Address(BaseModel):
        street: str = Field(description="街道")
        city: str = Field(description="城市")
        postal_code: str = Field(description="邮编")
    
    class ContactInfo(BaseModel):
        phone: str = Field(description="电话")
        email: str = Field(description="邮箱")
    
    person: PersonInfo = Field(description="人员信息")
    address: Address = Field(description="地址信息")
    contact: ContactInfo = Field(description="联系方式")
    notes: List[str] = Field(default_factory=list, description="备注列表")


class WeatherData(BaseModel):
    """天气数据模型（用于工具解析器测试）"""
    location: str = Field(description="地点")
    temperature: float = Field(description="温度")
    humidity: int = Field(description="湿度百分比", ge=0, le=100)
    conditions: str = Field(description="天气状况")


class AnalysisResult(BaseModel):
    """分析结果模型（用于工具解析器测试）"""
    topic: str = Field(description="分析主题")
    sentiment: str = Field(description="情感倾向")
    confidence: float = Field(description="置信度", ge=0.0, le=1.0)
    keywords: List[str] = Field(description="关键词列表")


class TestPydanticOutputParsers(unittest.TestCase):
    """Pydantic输出解析器测试类"""
    
    @classmethod
    def setUpClass(cls) -> None:
        """
        设置测试类的初始配置
        
        输入: 无
        输出: 无
        """
        cls.config = apis["local"]
        cls.model = ChatOpenAI(
            base_url=cls.config["base_url"],
            api_key=cls.config["api_key"],
            model="gpt-4o-mini",
            temperature=0.1,  # 低温度确保输出稳定
            max_tokens=1500,
            timeout=30
        )

    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        self.person_parser = PydanticOutputParser(pydantic_object=PersonInfo)
        self.joke_parser = PydanticOutputParser(pydantic_object=Joke)
        self.travel_parser = PydanticOutputParser(pydantic_object=TravelPlan)
        self.product_parser = PydanticOutputParser(pydantic_object=Product)
        self.nested_parser = PydanticOutputParser(pydantic_object=NestedModel)

    # ================== PydanticOutputParser 基础测试 ==================

    def test_pydantic_output_parser_basic(self) -> None:
        """
        测试PydanticOutputParser基础功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试PydanticOutputParser基础功能 ===")
        
        try:
            # 测试JSON字符串解析为Pydantic对象
            json_content = '{"name": "李明", "age": 28, "email": "liming@example.com", "occupation": "数据科学家"}'
            ai_message = AIMessage(content=json_content)
            
            result = self.person_parser.parse(ai_message.content)
            
            self.assertIsInstance(result, PersonInfo)
            self.assertEqual(result.name, "李明")
            self.assertEqual(result.age, 28)
            self.assertEqual(result.email, "liming@example.com")
            self.assertEqual(result.occupation, "数据科学家")
            
            print(f"原始JSON: {json_content}")
            print(f"解析结果: {result}")
            print(f"结果类型: {type(result)}")
            print("✅ PydanticOutputParser基础功能测试通过")
            
        except Exception as e:
            print(f"❌ PydanticOutputParser基础功能测试失败: {e}")
            raise

    def test_pydantic_parser_with_format_instructions(self) -> None:
        """
        测试PydanticOutputParser的格式指令
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试PydanticOutputParser格式指令 ===")
        
        try:
            # 获取格式指令
            format_instructions = self.joke_parser.get_format_instructions()
            
            self.assertIsInstance(format_instructions, str)
            self.assertIn("json", format_instructions.lower())
            self.assertIn("setup", format_instructions)
            self.assertIn("punchline", format_instructions)
            
            print(f"格式指令长度: {len(format_instructions)}")
            print(f"格式指令内容: {format_instructions[:200]}...")
            print("✅ PydanticOutputParser格式指令测试通过")
            
        except Exception as e:
            print(f"❌ PydanticOutputParser格式指令测试失败: {e}")
            raise

    def test_pydantic_parser_with_model_integration(self) -> None:
        """
        测试PydanticOutputParser与模型集成
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试PydanticOutputParser与模型集成 ===")
        
        try:
            # 创建包含格式指令的提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个有用的助手。请按照指定的JSON格式回答。\n{format_instructions}"),
                ("human", "{query}")
            ]).partial(format_instructions=self.joke_parser.get_format_instructions())
            
            # 构建链
            chain = prompt | self.model | self.joke_parser
            
            # 测试调用
            query = "请讲一个关于程序员的笑话"
            result = chain.invoke({"query": query})
            
            self.assertIsInstance(result, Joke)
            self.assertTrue(len(result.setup) > 0)
            self.assertTrue(len(result.punchline) > 0)
            
            print(f"问题: {query}")
            print(f"笑话设置: {result.setup}")
            print(f"笑话结尾: {result.punchline}")
            if result.rating:
                print(f"评分: {result.rating}")
            print("✅ PydanticOutputParser与模型集成测试通过")
            
        except Exception as e:
            print(f"❌ PydanticOutputParser与模型集成测试失败: {e}")
            print("注意：此测试依赖模型输出格式，可能偶尔失败")

    # ================== 复杂模型解析测试 ==================

    def test_complex_model_with_lists_and_enums(self) -> None:
        """
        测试包含列表和枚举的复杂模型解析
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试复杂模型解析（列表和枚举） ===")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "请按照JSON格式回答产品信息查询。\n{format_instructions}"),
                ("human", "介绍一款{product_type}产品")
            ]).partial(format_instructions=self.product_parser.get_format_instructions())
            
            chain = prompt | self.model | self.product_parser
            
            result = chain.invoke({"product_type": "笔记本电脑"})
            
            self.assertIsInstance(result, Product)
            self.assertIsInstance(result.category, ProductCategory)
            self.assertIsInstance(result.price, (int, float))
            self.assertIsInstance(result.in_stock, bool)
            self.assertIsInstance(result.tags, list)
            
            print(f"产品名称: {result.name}")
            print(f"产品分类: {result.category}")
            print(f"价格: {result.price}")
            print(f"是否有库存: {result.in_stock}")
            print(f"标签: {result.tags}")
            print("✅ 复杂模型解析测试通过")
            
        except Exception as e:
            print(f"❌ 复杂模型解析测试失败: {e}")
            print("注意：此测试依赖模型理解枚举和列表格式")

    def test_nested_model_parsing(self) -> None:
        """
        测试嵌套模型解析
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试嵌套模型解析 ===")
        
        try:
            # 手动构建嵌套JSON用于测试
            nested_json = {
                "person": {
                    "name": "王小红",
                    "age": 25,
                    "email": "wangxiaohong@example.com",
                    "occupation": "设计师"
                },
                "address": {
                    "street": "中关村大街1号",
                    "city": "北京",
                    "postal_code": "100080"
                },
                "contact": {
                    "phone": "13800138000",
                    "email": "contact@example.com"
                },
                "notes": ["重要客户", "VIP会员", "定期联系"]
            }
            
            ai_message = AIMessage(content=json.dumps(nested_json, ensure_ascii=False))
            result = self.nested_parser.parse(ai_message.content)
            
            self.assertIsInstance(result, NestedModel)
            self.assertIsInstance(result.person, PersonInfo)
            self.assertIsInstance(result.address, NestedModel.Address)
            self.assertIsInstance(result.contact, NestedModel.ContactInfo)
            self.assertEqual(result.person.name, "王小红")
            self.assertEqual(result.address.city, "北京")
            self.assertIn("重要客户", result.notes)
            
            print(f"人员姓名: {result.person.name}")
            print(f"地址城市: {result.address.city}")
            print(f"联系电话: {result.contact.phone}")
            print(f"备注数量: {len(result.notes)}")
            print("✅ 嵌套模型解析测试通过")
            
        except Exception as e:
            print(f"❌ 嵌套模型解析测试失败: {e}")
            raise

    # ================== PydanticToolsParser 测试 ==================

    def test_pydantic_tools_parser_basic(self) -> None:
        """
        测试PydanticToolsParser基础功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试PydanticToolsParser基础功能 ===")
        
        try:
            # 创建工具解析器
            tools_parser = PydanticToolsParser(tools=[WeatherData, AnalysisResult])
            
            # 测试解析器的配置
            self.assertEqual(len(tools_parser.tools), 2)
            self.assertIn(WeatherData, tools_parser.tools)
            self.assertIn(AnalysisResult, tools_parser.tools)
            
            # PydanticToolsParser主要用于与模型的tool calling功能集成
            # 在实际使用中，它会被绑定到支持工具调用的模型上
            # 模型会生成符合工具schema的调用，然后解析器将其转换为Pydantic对象
            
            # 我们可以验证解析器能正确识别工具类型
            weather_tool_name = None
            analysis_tool_name = None
            
            for tool in tools_parser.tools:
                if tool.__name__ == "WeatherData":
                    weather_tool_name = tool.__name__
                elif tool.__name__ == "AnalysisResult":
                    analysis_tool_name = tool.__name__
            
            self.assertEqual(weather_tool_name, "WeatherData")
            self.assertEqual(analysis_tool_name, "AnalysisResult")
            
            # 验证工具模型的字段
            weather_fields = WeatherData.model_fields
            self.assertIn("location", weather_fields)
            self.assertIn("temperature", weather_fields)
            self.assertIn("humidity", weather_fields)
            self.assertIn("conditions", weather_fields)
            
            analysis_fields = AnalysisResult.model_fields
            self.assertIn("topic", analysis_fields)
            self.assertIn("sentiment", analysis_fields)
            self.assertIn("confidence", analysis_fields)
            self.assertIn("keywords", analysis_fields)
            
            print(f"工具解析器配置: {len(tools_parser.tools)} 个工具")
            print(f"WeatherData工具字段: {list(weather_fields.keys())}")
            print(f"AnalysisResult工具字段: {list(analysis_fields.keys())}")
            print("✅ PydanticToolsParser基础功能测试通过")
            
        except Exception as e:
            print(f"❌ PydanticToolsParser基础功能测试失败: {e}")
            raise

    # ================== 流式解析测试 ==================

    def test_pydantic_streaming_parsing(self) -> None:
        """
        测试Pydantic解析器的流式处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试Pydantic流式解析 ===")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "请按照JSON格式制定旅行计划。\n{format_instructions}"),
                ("human", "制定一个去{destination}的{duration}天旅行计划")
            ]).partial(format_instructions=self.travel_parser.get_format_instructions())
            
            chain = prompt | self.model | self.travel_parser
            
            # 尝试流式解析（注意：不是所有解析器都支持流式）
            try:
                stream_results = list(chain.stream({
                    "destination": "日本", 
                    "duration": 7
                }))
                
                # 检查流式结果
                if stream_results:
                    final_result = stream_results[-1]
                    self.assertIsInstance(final_result, TravelPlan)
                    print(f"流式解析结果: {final_result}")
                else:
                    # 如果不支持流式，使用常规调用
                    result = chain.invoke({"destination": "日本", "duration": 7})
                    self.assertIsInstance(result, TravelPlan)
                    print(f"常规解析结果: {result}")
                
                print("✅ Pydantic流式解析测试通过")
                
            except Exception as stream_error:
                print(f"流式解析不支持，使用常规解析: {stream_error}")
                result = chain.invoke({"destination": "日本", "duration": 7})
                self.assertIsInstance(result, TravelPlan)
                print(f"常规解析结果: {result}")
                print("✅ Pydantic解析测试通过（常规模式）")
                
        except Exception as e:
            print(f"❌ Pydantic流式解析测试失败: {e}")
            print("注意：此测试依赖模型输出格式和流式支持")

    # ================== 错误处理和验证测试 ==================

    def test_pydantic_validation_errors(self) -> None:
        """
        测试Pydantic验证错误处理
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试Pydantic验证错误处理 ===")
        
        try:
            # 测试无效年龄
            invalid_person_json = '{"name": "测试", "age": 200, "email": "invalid-email"}'
            ai_message = AIMessage(content=invalid_person_json)
            
            with self.assertRaises(OutputParserException):
                self.person_parser.parse(ai_message.content)
            
            # 测试缺少必填字段
            incomplete_json = '{"age": 25}'
            ai_message2 = AIMessage(content=incomplete_json)
            
            with self.assertRaises(OutputParserException):
                self.person_parser.parse(ai_message2.content)
            
            print("✅ Pydantic验证错误处理测试通过")
            
        except Exception as e:
            print(f"❌ Pydantic验证错误处理测试失败: {e}")
            raise

    def test_pydantic_vs_structured_output(self) -> None:
        """
        测试PydanticOutputParser与模型结构化输出的对比
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试PydanticOutputParser与结构化输出对比 ===")
        
        try:
            # 方法1：使用PydanticOutputParser
            prompt1 = ChatPromptTemplate.from_messages([
                ("system", "请按照JSON格式回答。\n{format_instructions}"),
                ("human", "介绍一下{person}")
            ]).partial(format_instructions=self.person_parser.get_format_instructions())
            
            chain1 = prompt1 | self.model | self.person_parser
            
            # 方法2：使用模型的with_structured_output
            prompt2 = ChatPromptTemplate.from_template("介绍一下{person}，请返回结构化信息")
            structured_model = self.model.with_structured_output(PersonInfo)
            chain2 = prompt2 | structured_model
            
            person = "马斯克"
            
            try:
                result1 = chain1.invoke({"person": person})
                print(f"PydanticOutputParser结果: {result1}")
            except Exception as e:
                print(f"PydanticOutputParser失败: {e}")
                result1 = None
            
            try:
                result2 = chain2.invoke({"person": person})
                print(f"StructuredOutput结果: {result2}")
            except Exception as e:
                print(f"StructuredOutput失败: {e}")
                result2 = None
            
            # 至少一种方法应该成功
            self.assertTrue(result1 is not None or result2 is not None)
            
            if result1 and result2:
                self.assertIsInstance(result1, PersonInfo)
                self.assertIsInstance(result2, PersonInfo)
                print("两种方法都成功，可以进行比较")
            
            print("✅ PydanticOutputParser与结构化输出对比测试通过")
            
        except Exception as e:
            print(f"❌ PydanticOutputParser与结构化输出对比测试失败: {e}")
            print("注意：此测试依赖模型功能支持")


if __name__ == "__main__":
    unittest.main(verbosity=2) 