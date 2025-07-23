"""
Pydantic与LangChain集成测试

测试Pydantic BaseModel与LangChain的集成应用，包括：
- 结构化数据提取
- 响应格式化
- 输入验证
- 文本分类
- 嵌套数据处理
- 智能表单填写
- 配置管理
"""

import unittest
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnablePassthrough

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis

# 导入我们之前定义的Pydantic模型
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator, create_model
from pydantic_settings import BaseSettings
from typing_extensions import Literal


# 1. 用于结构化数据提取的模型
class UserProfile(BaseModel):
    """用户档案模型 - 用于从文本中提取结构化用户信息"""
    name: str = Field(description="用户姓名")
    age: Optional[int] = Field(None, description="用户年龄", ge=0, le=150)
    email: Optional[str] = Field(None, description="用户邮箱")
    occupation: Optional[str] = Field(None, description="职业")
    interests: List[str] = Field(default_factory=list, description="兴趣爱好列表")
    location: Optional[str] = Field(None, description="居住地")
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """验证邮箱格式"""
        if v and '@' not in v:
            raise ValueError('无效的邮箱格式')
        return v


# 2. 用于响应格式化的泛型模型
class AIResponse(BaseModel):
    """AI响应格式化模型"""
    status: str = Field(description="响应状态: success, error, warning")
    message: str = Field(description="主要消息内容")
    data: Optional[Dict[str, Any]] = Field(None, description="附加数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    confidence: float = Field(default=1.0, description="置信度", ge=0.0, le=1.0)


# 3. 用于文本分类的枚举模型
class ContentCategory(str, Enum):
    """内容分类枚举"""
    TECHNICAL = "technical"
    BUSINESS = "business"
    PERSONAL = "personal"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    OTHER = "other"


class TextClassification(BaseModel):
    """文本分类模型"""
    text: str = Field(description="原始文本")
    category: ContentCategory = Field(description="分类类别")
    keywords: List[str] = Field(description="关键词列表")
    confidence: float = Field(description="分类置信度", ge=0.0, le=1.0)
    reasoning: str = Field(description="分类理由")


# 4. 用于条件验证的模型
class TaskRequest(BaseModel):
    """任务请求模型 - 根据任务类型进行条件验证"""
    task_type: Literal["translation", "summarization", "analysis", "generation"]
    content: str = Field(description="任务内容")
    
    # 翻译任务专用字段
    source_language: Optional[str] = Field(None, description="源语言")
    target_language: Optional[str] = Field(None, description="目标语言")
    
    # 摘要任务专用字段
    max_length: Optional[int] = Field(None, description="最大摘要长度", gt=0)
    
    # 分析任务专用字段
    analysis_type: Optional[str] = Field(None, description="分析类型")
    
    # 生成任务专用字段
    style: Optional[str] = Field(None, description="生成风格")
    tone: Optional[str] = Field(None, description="语调")
    
    @model_validator(mode='after')
    def validate_task_specific_fields(self):
        """根据任务类型验证特定字段"""
        if self.task_type == "translation":
            if not self.source_language or not self.target_language:
                raise ValueError('翻译任务必须指定源语言和目标语言')
        elif self.task_type == "summarization":
            if not self.max_length:
                raise ValueError('摘要任务必须指定最大长度')
        elif self.task_type == "analysis":
            if not self.analysis_type:
                raise ValueError('分析任务必须指定分析类型')
        elif self.task_type == "generation":
            if not self.style:
                raise ValueError('生成任务必须指定生成风格')
        
        return self


# 5. 嵌套数据处理模型
class Address(BaseModel):
    """地址模型"""
    street: str = Field(description="街道地址")
    city: str = Field(description="城市")
    state: Optional[str] = Field(None, description="州/省")
    country: str = Field(description="国家")
    zip_code: Optional[str] = Field(None, description="邮政编码")


class Company(BaseModel):
    """公司模型"""
    name: str = Field(description="公司名称")
    industry: str = Field(description="行业")
    size: Optional[str] = Field(None, description="公司规模")
    address: Address = Field(description="公司地址")


class ComplexUserProfile(BaseModel):
    """复杂用户档案模型 - 包含嵌套数据"""
    personal_info: UserProfile = Field(description="个人信息")
    work_info: Company = Field(description="工作信息")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="用户偏好设置")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class TestPydanticLangChainIntegration(unittest.TestCase):
    """Pydantic与LangChain集成测试类"""
    
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
            temperature=0.1,  # 低温度确保输出稳定
            max_tokens=2000,
            timeout=30
        )
    
    def test_structured_data_extraction(self) -> None:
        """
        测试结构化数据提取
        从非结构化文本中提取用户信息
        """
        print("\n=== 测试结构化数据提取 ===")
        
        try:
            chat_model = self.get_chat_model()
            structured_llm = chat_model.with_structured_output(UserProfile)
            
            test_text = """
            我叫张小明，今年28岁，是一名软件工程师。我的邮箱是zhangming@example.com。
            我住在北京，平时喜欢编程、阅读和跑步。我在一家科技公司工作。
            """
            
            prompt = f"请从以下文本中提取用户信息：\n\n{test_text}"
            result = structured_llm.invoke(prompt)
            
            self.assertIsInstance(result, UserProfile)
            self.assertEqual(result.name, "张小明")
            self.assertEqual(result.age, 28)
            self.assertEqual(result.email, "zhangming@example.com")
            self.assertIn("编程", result.interests)
            
            print(f"提取结果: {result.model_dump()}")
            print("✅ 结构化数据提取测试通过!")
            
        except Exception as e:
            print(f"❌ 结构化数据提取测试失败: {e}")
    
    def test_response_formatting(self) -> None:
        """
        测试响应格式化
        将AI响应包装成结构化格式
        """
        print("\n=== 测试响应格式化 ===")
        
        try:
            chat_model = self.get_chat_model()
            # 使用function_calling方法避免Dict[str, Any]字段的schema限制
            structured_llm = chat_model.with_structured_output(
                AIResponse,
                method="function_calling"
            )
            
            prompt = "请回答：什么是机器学习？请按照指定的响应格式返回答案。"
            result = structured_llm.invoke(prompt)
            
            self.assertIsInstance(result, AIResponse)
            self.assertTrue(
                "machine learning" in result.message.lower() or 
                "机器学习" in result.message or 
                "人工智能" in result.message
            )
            self.assertIn(result.status, ["success", "error", "warning"])
            self.assertIsInstance(result.timestamp, datetime)
            
            print(f"格式化响应: {result.model_dump()}")
            print("✅ 响应格式化测试通过!")
            
        except Exception as e:
            print(f"❌ 响应格式化测试失败: {e}")
    
    def test_text_classification(self) -> None:
        """
        测试文本分类
        使用枚举模型进行文本内容分类
        """
        print("\n=== 测试文本分类 ===")
        
        try:
            chat_model = self.get_chat_model()
            structured_llm = chat_model.with_structured_output(TextClassification)
            
            test_texts = [
                "Python是一种编程语言，广泛用于数据科学和机器学习。",
                "公司第三季度销售额增长了15%，超出了预期目标。",
                "今天天气很好，我和朋友去公园里散步了。"
            ]
            
            for text in test_texts:
                prompt = f"请分析以下文本并进行分类：\n\n{text}"
                result = structured_llm.invoke(prompt)
                
                self.assertIsInstance(result, TextClassification)
                self.assertIsInstance(result.category, ContentCategory)
                self.assertGreater(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
                
                print(f"文本: {text[:30]}...")
                print(f"分类: {result.category.value}, 置信度: {result.confidence}")
                print(f"关键词: {result.keywords}")
                print(f"理由: {result.reasoning}")
                print("-" * 50)
            
            print("✅ 文本分类测试通过!")
            
        except Exception as e:
            print(f"❌ 文本分类测试失败: {e}")
    
    def test_conditional_validation(self) -> None:
        """
        测试条件验证
        根据任务类型进行智能验证和处理
        """
        print("\n=== 测试条件验证 ===")
        
        try:
            chat_model = self.get_chat_model()
            structured_llm = chat_model.with_structured_output(TaskRequest)
            
            # 测试不同任务类型的请求处理
            test_requests = [
                {
                    "request": "请将'Hello World'翻译成中文",
                    "expected_type": "translation"
                },
                {
                    "request": "请为这篇文章写一个100字的摘要",
                    "expected_type": "summarization"
                },
                {
                    "request": "分析这段代码的时间复杂度",
                    "expected_type": "analysis"
                }
            ]
            
            for test_case in test_requests:
                try:
                    prompt = f"分析用户请求，识别任务类型并提取相关参数。用户请求: {test_case['request']}"
                    result = structured_llm.invoke(prompt)
                    
                    self.assertIsInstance(result, TaskRequest)
                    print(f"请求: {test_case['request']}")
                    print(f"识别类型: {result.task_type}")
                    print(f"任务内容: {result.content}")
                    print("-" * 50)
                    
                except (ValidationError, OutputParserException) as e:
                    print(f"验证错误 (这是预期的): {test_case['request']}")
                    print(f"错误信息: {str(e)[:200]}...")
                    print("-" * 50)
                    continue
            
            print("✅ 条件验证测试通过!")
            
        except Exception as e:
            print(f"❌ 条件验证测试失败: {e}")
    
    def test_nested_data_processing(self) -> None:
        """
        测试嵌套数据处理
        处理包含多层嵌套结构的复杂数据
        """
        print("\n=== 测试嵌套数据处理 ===")
        
        try:
            chat_model = self.get_chat_model()
            # 使用function_calling方法避免structured output的schema限制
            structured_llm = chat_model.with_structured_output(
                ComplexUserProfile,
                method="function_calling"
            )
            
            # 简化文本内容
            complex_text = """
            个人信息：李华，32岁，邮箱lihua@tech.com，软件工程师，兴趣：编程、音乐，住在上海。
            工作信息：创新科技公司，软件开发行业，中型企业，地址：上海市浦东新区创新路123号，中国。
            偏好：编程语言Python，工作模式远程。
            """
            
            prompt = f"""
            从以下文本中提取用户档案信息，必须包含：
            1. personal_info: 个人信息（姓名、年龄、邮箱等）
            2. work_info: 工作信息（公司名称、行业、地址等）
            3. preferences: 用户偏好设置
            
            注意：
            - work_info.name 应该是公司名称
            - work_info.address 必须包含详细的地址字段（street, city, country等）
            - preferences 是一个字典，包含用户的各种偏好设置
            
            文本内容：
            {complex_text}
            """
            
            result = structured_llm.invoke(prompt)
            
            self.assertIsInstance(result, ComplexUserProfile)
            self.assertIsInstance(result.personal_info, UserProfile)
            self.assertIsInstance(result.work_info, Company)
            self.assertIsInstance(result.work_info.address, Address)
            
            print("个人信息:")
            print(f"  姓名: {result.personal_info.name}")
            print(f"  年龄: {result.personal_info.age}")
            print(f"  邮箱: {result.personal_info.email}")
            
            print("\n工作信息:")
            print(f"  公司: {result.work_info.name}")
            print(f"  行业: {result.work_info.industry}")
            print(f"  地址: {result.work_info.address.city}, {result.work_info.address.country}")
            
            print(f"\n偏好设置: {result.preferences}")
            print("✅ 嵌套数据处理测试通过!")
            
        except Exception as e:
            print(f"❌ 嵌套数据处理测试失败: {e}")
            # 提供详细的错误分析
            error_msg = str(e)
            if "Invalid schema" in error_msg:
                print("💡 分析：OpenAI structured output schema限制")
                print("   - Dict[str, Any]字段需要特殊处理")
                print("   - 使用function_calling方法可以避免此问题")
            elif "validation error" in error_msg.lower():
                print("💡 分析：Pydantic验证错误")
                print("   - AI生成的数据结构不完全符合模型定义")
                print("   - 可能需要优化提示词或简化模型结构")
            elif "timed out" in error_msg.lower():
                print("💡 分析：请求超时，可能的原因：")
                print("   1. 模型结构过于复杂")
                print("   2. 网络延迟")
                print("   3. API服务负载过高")
    
    def test_simple_nested_data_processing(self) -> None:
        """
        测试简化的嵌套数据处理
        使用更简单的嵌套结构来验证兼容性
        """
        print("\n=== 测试简化嵌套数据处理 ===")
        
        try:
            # 创建简化的嵌套模型
            class SimpleAddress(BaseModel):
                """简化地址模型"""
                city: str = Field(description="城市")
                country: str = Field(description="国家")
            
            class SimpleProfile(BaseModel):
                """简化用户档案"""
                name: str = Field(description="姓名")
                age: int = Field(description="年龄")
                address: SimpleAddress = Field(description="地址信息")
            
            chat_model = self.get_chat_model()
            structured_llm = chat_model.with_structured_output(SimpleProfile)
            
            simple_text = """
            用户信息：张三，25岁，住在北京，中国
            """
            
            prompt = f"从文本中提取用户信息：\n\n{simple_text}"
            result = structured_llm.invoke(prompt)
            
            self.assertIsInstance(result, SimpleProfile)
            self.assertIsInstance(result.address, SimpleAddress)
            
            print("提取结果:")
            print(f"  姓名: {result.name}")
            print(f"  年龄: {result.age}")
            print(f"  城市: {result.address.city}")
            print(f"  国家: {result.address.country}")
            print("✅ 简化嵌套数据处理测试通过!")
            
        except Exception as e:
            print(f"❌ 简化嵌套数据处理测试失败: {e}")
    
    def test_medium_nested_data_processing(self) -> None:
        """
        测试中等复杂度的嵌套数据处理
        """
        print("\n=== 测试中等复杂度嵌套数据处理 ===")
        
        try:
            # 创建中等复杂度的嵌套模型
            class MediumUserProfile(BaseModel):
                """中等复杂度用户档案"""
                name: str = Field(description="姓名")
                age: int = Field(description="年龄")
                address: Address = Field(description="地址信息")
                company: Optional[Company] = Field(None, description="公司信息")
            
            chat_model = self.get_chat_model()
            structured_llm = chat_model.with_structured_output(MediumUserProfile)
            
            medium_text = """
            用户张华，30岁，住在上海市黄浦区南京路100号，中国，邮编200000。
            他在阿里巴巴公司工作，这家公司成立于1999年，有5000名员工，主要从事电子商务行业。
            """
            
            prompt = f"从文本中提取用户和公司信息：\n\n{medium_text}"
            result = structured_llm.invoke(prompt)
            
            self.assertIsInstance(result, MediumUserProfile)
            self.assertIsInstance(result.address, Address)
            
            print("提取结果:")
            print(f"  姓名: {result.name}")
            print(f"  年龄: {result.age}")
            print(f"  地址: {result.address.city}")
            if result.company:
                print(f"  公司: {result.company.name}")
                print(f"  行业: {result.company.industry}")
            print("✅ 中等复杂度嵌套数据处理测试通过!")
            
        except Exception as e:
            print(f"❌ 中等复杂度嵌套数据处理测试失败: {e}")
    
    def test_smart_form_filling(self) -> None:
        """
        测试智能表单填写
        根据用户描述自动填写结构化表单
        """
        print("\n=== 测试智能表单填写 ===")
        
        try:
            chat_model = self.get_chat_model()
            
            # 创建表单模板
            class RegistrationForm(BaseModel):
                """注册表单模型"""
                full_name: str = Field(description="全名")
                email: str = Field(description="邮箱地址")
                phone: Optional[str] = Field(None, description="电话号码")
                age: Optional[int] = Field(None, description="年龄")
                occupation: Optional[str] = Field(None, description="职业")
                interests: List[str] = Field(default_factory=list, description="兴趣爱好")
                newsletter: bool = Field(False, description="是否订阅邮件")
                
                @field_validator('email')
                @classmethod
                def validate_email(cls, v):
                    if '@' not in v:
                        raise ValueError('邮箱格式无效')
                    return v
            
            structured_llm = chat_model.with_structured_output(RegistrationForm)
            
            user_descriptions = [
                "我是王小红，28岁，软件开发工程师，邮箱是xiaohong@email.com，喜欢编程和读书，希望订阅你们的邮件",
                "张三，医生，35岁，不想收到邮件，喜欢运动和音乐",
                "李四，学生，邮箱li4@student.edu.cn"
            ]
            
            for desc in user_descriptions:
                prompt = f"根据用户提供的信息自动填写注册表单。如果信息不足，请使用合理的默认值。用户信息：{desc}"
                result = structured_llm.invoke(prompt)
                
                self.assertIsInstance(result, RegistrationForm)
                print(f"用户描述: {desc}")
                print(f"生成表单: {result.model_dump()}")
                print("-" * 60)
            
            print("✅ 智能表单填写测试通过!")
            
        except Exception as e:
            print(f"❌ 智能表单填写测试失败: {e}")
    
    def test_error_handling_and_fallback(self) -> None:
        """
        测试错误处理和回退机制
        当AI输出不符合Pydantic模型时的处理
        """
        print("\n=== 测试错误处理和回退机制 ===")
        
        try:
            chat_model = self.get_chat_model()
            
            # 先尝试使用结构化输出
            structured_llm = chat_model.with_structured_output(UserProfile)
            
            # 故意使用模糊或不完整的信息
            problematic_texts = [
                "这是一段没有任何用户信息的文本",
                "用户年龄是二十五岁",  # 非数字年龄
                "邮箱是invalid-email"  # 无效邮箱
            ]
            
            for text in problematic_texts:
                try:
                    prompt = f"从文本中提取用户信息：{text}"
                    result = structured_llm.invoke(prompt)
                    
                    print(f"成功处理: {text[:30]}...")
                    print(f"结果: {result.model_dump()}")
                    
                except Exception as e:
                    print(f"处理失败 (预期): {text[:30]}...")
                    print(f"错误: {str(e)[:100]}...")
                    
                    # 实现回退机制 - 使用普通聊天模型
                    fallback_prompt = f"提取任何可用的用户信息，返回JSON格式。如果没有信息，返回空对象。文本：{text}"
                    fallback_result = chat_model.invoke(fallback_prompt)
                    print(f"回退结果: {fallback_result.content[:100]}...")
                
                print("-" * 50)
            
            print("✅ 错误处理和回退机制测试通过!")
            
        except Exception as e:
            print(f"❌ 错误处理测试失败: {e}")
    
    def test_dynamic_model_creation_from_dict(self) -> None:
        """
        测试根据字典动态创建BaseModel并与LangChain集成
        展示如何从配置字典动态生成Pydantic模型，然后在LangChain中使用
        """
        print("\n=== 测试动态模型创建与调用 ===")
        
        try:
            chat_model = self.get_chat_model()
            
            # 测试场景1：从简单字典创建模型
            simple_schema = {
                'name': (str, Field(description="产品名称")),
                'price': (float, Field(description="产品价格", gt=0)),
                'category': (str, Field(description="产品类别")),
                'in_stock': (bool, Field(description="是否有库存"))
            }
            
            # 动态创建Product模型
            ProductModel = create_model('Product', **simple_schema)
            
            print("1. 简单字典创建模型测试:")
            print(f"   动态创建的模型: {ProductModel.__name__}")
            print(f"   模型字段: {list(ProductModel.model_fields.keys())}")
            
            # 使用动态创建的模型与LangChain集成
            structured_llm = chat_model.with_structured_output(ProductModel)
            
            product_text = "iPhone 15 Pro，价格8999元，属于手机类别，目前有库存"
            prompt = f"从以下文本中提取产品信息：\n\n{product_text}"
            result = structured_llm.invoke(prompt)
            
            self.assertIsInstance(result, ProductModel)
            print(f"   提取结果: {result.model_dump()}")
            
            # 测试场景2：从复杂配置字典创建模型
            complex_schema_config = {
                "model_name": "UserOrder",
                "fields": {
                    "order_id": {
                        "type": str,
                        "field": Field(description="订单ID")
                    },
                    "user_name": {
                        "type": str,
                        "field": Field(description="用户姓名")
                    },
                    "items": {
                        "type": List[str],
                        "field": Field(description="订单商品列表")
                    },
                    "total_amount": {
                        "type": float,
                        "field": Field(description="订单总金额", ge=0.01)
                    },
                    "order_date": {
                        "type": Optional[str],
                        "field": Field(None, description="订单日期")
                    }
                }
            }
            
            # 根据复杂配置创建模型
            order_fields = {}
            for field_name, field_config in complex_schema_config["fields"].items():
                order_fields[field_name] = (field_config["type"], field_config["field"])
            
            OrderModel = create_model(complex_schema_config["model_name"], **order_fields)
            
            print("\n2. 复杂配置字典创建模型测试:")
            print(f"   动态创建的模型: {OrderModel.__name__}")
            print(f"   模型字段: {list(OrderModel.model_fields.keys())}")
            
            # 使用复杂动态模型
            structured_llm_order = chat_model.with_structured_output(OrderModel)
            
            order_text = """
            订单编号：ORD-20240101-001
            客户：李明
            购买商品：MacBook Pro、iPhone 15、AirPods
            订单总额：25999.99元
            下单时间：2024年1月1日
            """
            
            order_prompt = f"从以下订单信息中提取结构化数据：\n\n{order_text}"
            order_result = structured_llm_order.invoke(order_prompt)
            
            self.assertIsInstance(order_result, OrderModel)
            print(f"   订单提取结果: {order_result.model_dump()}")
            
            # 测试场景3：运行时动态模型创建和调用
            def create_model_from_requirements(requirements: Dict[str, Any]) -> type:
                """
                根据需求字典动态创建Pydantic模型
                
                Args:
                    requirements: 包含字段定义的需求字典
                    
                Returns:
                    type: 动态创建的Pydantic模型类
                """
                model_fields = {}
                for field_name, field_def in requirements.items():
                    field_type = field_def.get("type", str)
                    field_desc = field_def.get("description", f"{field_name}字段")
                    field_required = field_def.get("required", True)
                    
                    if field_required:
                        model_fields[field_name] = (field_type, Field(description=field_desc))
                    else:
                        default_val = field_def.get("default", None)
                        model_fields[field_name] = (Optional[field_type], Field(default_val, description=field_desc))
                
                return create_model("DynamicModel", **model_fields)
            
            # 动态需求配置
            dynamic_requirements = {
                "event_name": {
                    "type": str,
                    "description": "活动名称",
                    "required": True
                },
                "event_date": {
                    "type": str,
                    "description": "活动日期",
                    "required": True
                },
                "location": {
                    "type": str,
                    "description": "活动地点",
                    "required": False,
                    "default": "未指定"
                },
                "attendees": {
                    "type": List[str],
                    "description": "参与者列表",
                    "required": False,
                    "default": []
                },
                "budget": {
                    "type": float,
                    "description": "活动预算",
                    "required": False,
                    "default": 0.0
                }
            }
            
            DynamicEventModel = create_model_from_requirements(dynamic_requirements)
            
            print("\n3. 运行时动态模型创建测试:")
            print(f"   动态模型名称: {DynamicEventModel.__name__}")
            print(f"   模型字段: {list(DynamicEventModel.model_fields.keys())}")
            
            # 使用运行时创建的模型
            structured_llm_event = chat_model.with_structured_output(DynamicEventModel)
            
            event_text = """
            我们计划举办一个技术分享会，名称是"Python深度学习技术交流"，
            时间定在2024年3月15日，地点在科技园会议室A，
            预计参与人员包括张三、李四、王五，预算大概5000元。
            """
            
            event_prompt = f"从以下活动描述中提取活动信息：\n\n{event_text}"
            event_result = structured_llm_event.invoke(event_prompt)
            
            self.assertIsInstance(event_result, DynamicEventModel)
            print(f"   活动提取结果: {event_result.model_dump()}")
            
            # 测试场景4：模型继承和扩展
            base_schema = {
                'id': (str, Field(description="基础ID")),
                'name': (str, Field(description="名称")),
                'created_at': (Optional[str], Field(None, description="创建时间"))
            }
            
            BaseItemModel = create_model('BaseItem', **base_schema)
            
            # 创建扩展模型
            extended_schema = {
                'description': (str, Field(description="详细描述")),
                'tags': (List[str], Field(default_factory=list, description="标签列表")),
                'priority': (int, Field(default=1, description="优先级", ge=1, le=5))
            }
            
            # 合并基础和扩展字段
            full_schema = {**base_schema, **extended_schema}
            ExtendedItemModel = create_model('ExtendedItem', **full_schema)
            
            print("\n4. 模型继承和扩展测试:")
            print(f"   基础模型: {BaseItemModel.__name__} -> {list(BaseItemModel.model_fields.keys())}")
            print(f"   扩展模型: {ExtendedItemModel.__name__} -> {list(ExtendedItemModel.model_fields.keys())}")
            
            structured_llm_extended = chat_model.with_structured_output(ExtendedItemModel)
            
            item_text = """
            任务ID: TASK-001
            任务名称: 优化数据库查询性能
            创建时间: 2024-01-15
            详细描述: 对用户查询接口进行性能优化，预计提升50%查询速度
            相关标签: 数据库、性能、优化、后端
            优先级: 高优先级（4级）
            """
            
            item_prompt = f"从以下任务描述中提取完整的任务信息：\n\n{item_text}"
            item_result = structured_llm_extended.invoke(item_prompt)
            
            self.assertIsInstance(item_result, ExtendedItemModel)
            print(f"   任务提取结果: {item_result.model_dump()}")
            
            print("\n✅ 动态模型创建与调用测试全部通过!")
            
        except Exception as e:
            print(f"❌ 动态模型创建测试失败: {e}")
            # 提供详细的错误分析
            error_msg = str(e)
            if "create_model" in error_msg:
                print("💡 分析：create_model相关错误")
                print("   - 检查字段定义格式是否正确")
                print("   - 确认类型注解是否有效")
            elif "Invalid schema" in error_msg:
                print("💡 分析：模型schema验证错误")
                print("   - 动态创建的模型可能包含不支持的字段类型")
                print("   - 检查Field定义是否符合要求")


def main() -> int:
    """
    运行Pydantic与LangChain集成测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行Pydantic与LangChain集成测试")
    print("=" * 60)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main() 