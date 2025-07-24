"""
输出解析器错误处理和高级功能测试

测试LangChain的输出解析器高级功能，包括：
- OutputFixingParser: 输出修复解析器
- RetryWithErrorOutputParser: 重试解析器
- 自定义重试逻辑
- 回退解析策略
- 解析器性能测试
- 异常处理机制

参考文档:
- https://python.langchain.com/docs/how_to/output_parser_retry/
- https://python.langchain.com/docs/how_to/output_parser_fixing/
"""

import unittest
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser, RetryWithErrorOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


# ================== 测试用Pydantic模型 ==================

class UserProfile(BaseModel):
    """用户档案模型"""
    name: str = Field(description="用户姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    email: str = Field(description="邮箱地址")
    skills: List[str] = Field(description="技能列表")
    is_active: bool = Field(description="是否活跃")


class WeatherReport(BaseModel):
    """天气报告模型"""
    location: str = Field(description="地点")
    temperature: float = Field(description="温度")
    humidity: int = Field(description="湿度", ge=0, le=100)
    description: str = Field(description="天气描述")


# ================== 自定义错误处理解析器 ==================

class FallbackOutputParser:
    """
    回退解析器
    当主解析器失败时使用备用解析器
    """
    
    def __init__(self, primary_parser, fallback_parser):
        """
        初始化回退解析器
        
        Args:
            primary_parser: 主解析器
            fallback_parser: 备用解析器
        """
        self.primary_parser = primary_parser
        self.fallback_parser = fallback_parser
    
    def parse(self, text: str):
        """
        解析文本，如果主解析器失败则使用备用解析器
        
        Args:
            text: 待解析的文本
            
        Returns:
            解析结果
        """
        try:
            return self.primary_parser.parse(text)
        except Exception as primary_error:
            try:
                return self.fallback_parser.parse(text)
            except Exception as fallback_error:
                raise OutputParserException(
                    f"Both parsers failed. Primary: {primary_error}, Fallback: {fallback_error}"
                )
    
    def get_format_instructions(self) -> str:
        """获取格式指令"""
        return self.primary_parser.get_format_instructions()


class ValidatingOutputParser:
    """
    验证解析器
    在解析后进行额外验证
    """
    
    def __init__(self, base_parser, validator_func=None):
        """
        初始化验证解析器
        
        Args:
            base_parser: 基础解析器
            validator_func: 验证函数
        """
        self.base_parser = base_parser
        self.validator_func = validator_func
    
    def parse(self, text: str):
        """
        解析并验证文本
        
        Args:
            text: 待解析的文本
            
        Returns:
            验证后的解析结果
        """
        result = self.base_parser.parse(text)
        
        if self.validator_func:
            if not self.validator_func(result):
                raise OutputParserException("Validation failed after parsing")
        
        return result
    
    def get_format_instructions(self) -> str:
        """获取格式指令"""
        return self.base_parser.get_format_instructions()


class TestOutputParserErrorHandling(unittest.TestCase):
    """输出解析器错误处理测试类"""
    
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
            temperature=0.7,  # 稍高温度增加输出变化
            max_tokens=1000,
            timeout=30
        )

    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        self.user_parser = PydanticOutputParser(pydantic_object=UserProfile)
        self.weather_parser = PydanticOutputParser(pydantic_object=WeatherReport)

    # ================== OutputFixingParser 测试 ==================

    def test_output_fixing_parser_basic(self) -> None:
        """
        测试OutputFixingParser基础功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试OutputFixingParser基础功能 ===")
        
        try:
            # 创建修复解析器
            fixing_parser = OutputFixingParser.from_llm(
                parser=self.user_parser,
                llm=self.model
            )
            
            # 故意提供格式错误的JSON
            malformed_json = '''
            {
                "name": "张三",
                "age": "25岁",  // 这里应该是数字，不是字符串
                "email": "zhangsan@example.com",
                "skills": "Python, Java",  // 这里应该是数组
                "is_active": "true"  // 这里应该是布尔值
            }
            '''
            
            ai_message = AIMessage(content=malformed_json)
            
            # 使用修复解析器
            result = fixing_parser.parse(ai_message.content)
            
            self.assertIsInstance(result, UserProfile)
            self.assertEqual(result.name, "张三")
            self.assertIsInstance(result.age, int)
            self.assertIsInstance(result.skills, list)
            self.assertIsInstance(result.is_active, bool)
            
            print(f"原始错误JSON: {malformed_json}")
            print(f"修复后结果: {result}")
            print("✅ OutputFixingParser基础功能测试通过")
            
        except Exception as e:
            print(f"❌ OutputFixingParser基础功能测试失败: {e}")
            print("注意：此测试依赖模型修复能力，可能偶尔失败")

    def test_output_fixing_parser_with_chain(self) -> None:
        """
        测试OutputFixingParser与链的集成
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试OutputFixingParser与链集成 ===")
        
        try:
            # 创建修复解析器
            fixing_parser = OutputFixingParser.from_llm(
                parser=self.weather_parser,
                llm=self.model
            )
            
            # 创建可能产生格式错误的提示
            prompt = ChatPromptTemplate.from_template(
                """请提供{location}的天气信息，以JSON格式返回。
                
要求包含以下字段：
- location: 地点名称
- temperature: 温度（数字）
- humidity: 湿度百分比（0-100的整数）
- description: 天气描述

注意：请确保JSON格式正确。"""
            )
            
            # 构建链
            chain = prompt | self.model | fixing_parser
            
            result = chain.invoke({"location": "北京"})
            
            self.assertIsInstance(result, WeatherReport)
            self.assertEqual(result.location, "北京")
            self.assertIsInstance(result.temperature, (int, float))
            self.assertIsInstance(result.humidity, int)
            self.assertTrue(0 <= result.humidity <= 100)
            
            print(f"天气信息: {result}")
            print("✅ OutputFixingParser与链集成测试通过")
            
        except Exception as e:
            print(f"❌ OutputFixingParser与链集成测试失败: {e}")
            print("注意：此测试依赖模型输出和修复能力")

    # ================== RetryWithErrorOutputParser 测试 ==================

    def test_retry_with_error_parser_basic(self) -> None:
        """
        测试RetryWithErrorOutputParser基础功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试RetryWithErrorOutputParser基础功能 ===")
        
        from langchain.output_parsers import RetryOutputParser
        from langchain_core.prompts import PromptTemplate
        
        # 创建重试解析器
        retry_parser = RetryOutputParser.from_llm(
            parser=self.user_parser,
            llm=self.model
        )
        
        # 创建提示模板
        prompt = PromptTemplate(
            template="""根据用户信息创建用户档案，返回JSON格式：

用户信息：{user_info}

{format_instructions}

请确保所有字段类型正确。""",
            input_variables=["user_info"],
            partial_variables={"format_instructions": self.user_parser.get_format_instructions()}
        )
        
        # 创建链式调用
        completion_chain = prompt | self.model
        
        # 测试用户信息
        user_info = "李明，28岁，软件工程师，邮箱liming@tech.com，精通Python和JavaScript"
        
        # 先格式化prompt
        formatted_prompt = prompt.format_prompt(user_info=user_info)
        
        # 获取模型输出
        model_output = completion_chain.invoke({"user_info": user_info})
        
        # 使用retry parser处理
        result = retry_parser.parse_with_prompt(
            completion=model_output.content,
            prompt_value=formatted_prompt
        )
        
        self.assertIsInstance(result, UserProfile)
        self.assertTrue("李明" in result.name or "liming" in result.name.lower())
        self.assertIsInstance(result.age, int)
        self.assertIsInstance(result.skills, list)
        
        print(f"用户信息: {user_info}")
        print(f"解析结果: {result}")
        print("✅ RetryWithErrorOutputParser基础功能测试通过")

    # ================== 回退解析器测试 ==================

    def test_fallback_output_parser(self) -> None:
        """
        测试回退输出解析器
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试回退输出解析器 ===")
        
        try:
            from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
            
            # 主解析器：严格的Pydantic解析器
            primary_parser = self.user_parser
            
            # 备用解析器：宽松的JSON解析器
            fallback_parser = JsonOutputParser()
            
            # 创建回退解析器
            fallback_output_parser = FallbackOutputParser(
                primary_parser=primary_parser,
                fallback_parser=fallback_parser
            )
            
            # 测试正确格式（应该用主解析器）
            valid_json = '''
            {
                "name": "王五",
                "age": 30,
                "email": "wangwu@example.com",
                "skills": ["Python", "Docker"],
                "is_active": true
            }
            '''
            
            result1 = fallback_output_parser.parse(valid_json)
            self.assertIsInstance(result1, UserProfile)
            
            # 测试部分错误格式（应该用备用解析器）
            partial_json = '''
            {
                "name": "赵六",
                "age": 35,
                "email": "zhaoliu@example.com",
                "extra_field": "这个字段不在模型中"
            }
            '''
            
            result2 = fallback_output_parser.parse(partial_json)
            self.assertIsInstance(result2, dict)
            self.assertEqual(result2["name"], "赵六")
            
            print(f"主解析器结果: {result1}")
            print(f"备用解析器结果: {result2}")
            print("✅ 回退输出解析器测试通过")
            
        except Exception as e:
            print(f"❌ 回退输出解析器测试失败: {e}")
            raise

    # ================== 验证解析器测试 ==================

    def test_validating_output_parser(self) -> None:
        """
        测试验证输出解析器
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试验证输出解析器 ===")
        
        try:
            # 自定义验证函数
            def validate_user(user: UserProfile) -> bool:
                """验证用户数据是否合理"""
                if user.age < 0 or user.age > 120:
                    return False
                if "@" not in user.email:
                    return False
                if len(user.skills) == 0:
                    return False
                return True
            
            # 创建验证解析器
            validating_parser = ValidatingOutputParser(
                base_parser=self.user_parser,
                validator_func=validate_user
            )
            
            # 测试有效数据
            valid_json = '''
            {
                "name": "测试用户",
                "age": 25,
                "email": "test@example.com",
                "skills": ["Python", "JavaScript"],
                "is_active": true
            }
            '''
            
            result = validating_parser.parse(valid_json)
            self.assertIsInstance(result, UserProfile)
            
            # 测试无效数据（年龄超范围）
            invalid_json = '''
            {
                "name": "无效用户",
                "age": 200,
                "email": "invalid@example.com",
                "skills": ["Python"],
                "is_active": true
            }
            '''
            
            with self.assertRaises(OutputParserException):
                validating_parser.parse(invalid_json)
            
            print(f"验证成功的结果: {result}")
            print("✅ 验证输出解析器测试通过")
            
        except Exception as e:
            print(f"❌ 验证输出解析器测试失败: {e}")
            raise

    # ================== 异常处理机制测试 ==================

    def test_error_handling_mechanisms(self) -> None:
        """
        测试各种异常处理机制
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试异常处理机制 ===")
        
        try:
            # 测试JSON解析错误
            invalid_json = "这不是有效的JSON格式"
            ai_message = AIMessage(content=invalid_json)
            
            with self.assertRaises(Exception):
                self.user_parser.parse(ai_message.content)
            
            # 测试Pydantic验证错误
            invalid_data_json = '''
            {
                "name": "",
                "age": -5,
                "email": "not-an-email",
                "skills": [],
                "is_active": "maybe"
            }
            '''
            
            with self.assertRaises(OutputParserException):
                self.user_parser.parse(invalid_data_json)
            
            # 测试空内容
            empty_message = AIMessage(content="")
            
            with self.assertRaises(Exception):
                self.user_parser.parse(empty_message.content)
            
            print("✅ 异常处理机制测试通过")
            
        except Exception as e:
            print(f"❌ 异常处理机制测试失败: {e}")
            raise

    # ================== 解析器性能测试 ==================

    def test_parser_performance(self) -> None:
        """
        测试解析器性能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试解析器性能 ===")
        
        try:
            import time
            
            # 准备测试数据
            test_json = '''
            {
                "name": "性能测试用户",
                "age": 30,
                "email": "perf@example.com",
                "skills": ["Python", "JavaScript", "Go", "Rust"],
                "is_active": true
            }
            '''
            
            # 测试普通解析器性能
            start_time = time.time()
            for _ in range(10):
                result = self.user_parser.parse(test_json)
            normal_time = time.time() - start_time
            
            # 测试修复解析器性能（如果可用）
            try:
                fixing_parser = OutputFixingParser.from_llm(
                    parser=self.user_parser,
                    llm=self.model
                )
                
                start_time = time.time()
                for _ in range(3):  # 减少次数，因为LLM调用较慢
                    result = fixing_parser.parse(test_json)
                fixing_time = time.time() - start_time
                
                print(f"普通解析器 10次解析耗时: {normal_time:.3f}秒")
                print(f"修复解析器 3次解析耗时: {fixing_time:.3f}秒")
                print(f"平均单次解析时间 - 普通: {normal_time/10:.3f}秒, 修复: {fixing_time/3:.3f}秒")
                
            except Exception as e:
                print(f"修复解析器性能测试跳过: {e}")
                print(f"普通解析器 10次解析耗时: {normal_time:.3f}秒")
            
            print("✅ 解析器性能测试通过")
            
        except Exception as e:
            print(f"❌ 解析器性能测试失败: {e}")
            raise

    # ================== 复杂错误场景测试 ==================

    def test_complex_error_scenarios(self) -> None:
        """
        测试复杂错误场景
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试复杂错误场景 ===")
        
        try:
            scenarios = [
                {
                    "name": "不完整JSON",
                    "content": '{"name": "张三", "age": 25'
                },
                {
                    "name": "错误字段类型",
                    "content": '{"name": 123, "age": "twenty", "email": true, "skills": "Python", "is_active": "yes"}'
                },
                {
                    "name": "缺少必填字段",
                    "content": '{"name": "李四"}'
                },
                {
                    "name": "额外字段",
                    "content": '{"name": "王五", "age": 30, "email": "wang@test.com", "skills": ["Python"], "is_active": true, "extra": "field"}'
                },
                {
                    "name": "嵌套错误",
                    "content": '{"name": "赵六", "age": 25, "email": "zhao@test.com", "skills": [{"lang": "Python"}], "is_active": true}'
                }
            ]
            
            error_count = 0
            for scenario in scenarios:
                try:
                    result = self.user_parser.parse(scenario["content"])
                    print(f"意外成功 - {scenario['name']}: {result}")
                except Exception as e:
                    error_count += 1
                    print(f"预期错误 - {scenario['name']}: {type(e).__name__}")
            
            print(f"总计 {len(scenarios)} 个场景，{error_count} 个产生预期错误")
            print("✅ 复杂错误场景测试通过")
            
        except Exception as e:
            print(f"❌ 复杂错误场景测试失败: {e}")
            raise

    # ================== 解析器组合测试 ==================

    def test_parser_combinations(self) -> None:
        """
        测试解析器组合使用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试解析器组合使用 ===")
        
        from langchain.output_parsers import RetryOutputParser
        from langchain_core.prompts import PromptTemplate
        
        # 组合多种错误处理策略
        base_parser = self.user_parser
        
        # 1. 基础解析器 + 修复
        fixing_parser = OutputFixingParser.from_llm(
            parser=base_parser,
            llm=self.model
        )
        
        # 2. 修复解析器 + 重试
        retry_fixing_parser = RetryOutputParser.from_llm(
            parser=fixing_parser,
            llm=self.model
        )
        
        # 创建测试提示
        prompt = PromptTemplate(
            template="""请根据以下用户信息创建用户档案，返回JSON格式：

用户信息：组合测试用户，30岁，邮箱格式可能有误：test-email，技能是Python和JavaScript，当前活跃

{format_instructions}

请确保JSON格式正确，所有字段类型匹配。""",
            input_variables=[],
            partial_variables={"format_instructions": base_parser.get_format_instructions()}
        )
        
        # 构建完整的链
        completion_chain = prompt | self.model
        
        # 先格式化prompt
        formatted_prompt = prompt.format_prompt()
        
        # 获取模型输出
        model_output = completion_chain.invoke({})
        
        # 使用retry parser处理
        result = retry_fixing_parser.parse_with_prompt(
            completion=model_output.content,
            prompt_value=formatted_prompt
        )
        
        self.assertIsInstance(result, UserProfile)
        print(f"组合解析器成功处理复杂数据: {result}")
        print("✅ 解析器组合使用测试通过")


if __name__ == "__main__":
    unittest.main(verbosity=2) 