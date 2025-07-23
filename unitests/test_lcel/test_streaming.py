"""
LangChain Expression Language (LCEL) 流式传输测试

测试LCEL的流式传输功能：
- 基本流式输出
- 异步流式传输
- 流式输出处理
- 流式错误处理

作者: AI Assistant
创建时间: 2025年
"""

import unittest
import asyncio
from typing import Iterator, AsyncIterator, List
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.config.api import apis


class TestLCELStreaming(unittest.TestCase):
    """LCEL流式传输测试类"""
    
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
            temperature=0.7,
            max_tokens=300,
            timeout=30
        )
    
    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        self.prompt = ChatPromptTemplate.from_template("请详细回答: {question}")
        self.str_parser = StrOutputParser()
    
    def test_basic_streaming(self) -> None:
        """
        测试基本流式输出功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试基本流式输出功能 ===")
        
        # 创建流式链
        streaming_chain = self.prompt | self.model | self.str_parser
        
        test_input = {"question": "什么是人工智能？"}
        
        # 收集流式输出
        chunks = []
        for chunk in streaming_chain.stream(test_input):
            chunks.append(chunk)
            if len(chunks) <= 5:  # 只打印前几个chunk
                print(f"流式chunk {len(chunks)}: {repr(chunk)}")
        
        # 验证流式输出
        self.assertGreater(len(chunks), 1)  # 应该有多个chunk
        full_response = "".join(chunks)
        self.assertGreater(len(full_response), 0)
        
        print(f"总共收到 {len(chunks)} 个chunks")
        print(f"完整响应长度: {len(full_response)}")
        print(f"完整响应预览: {full_response[:100]}...")
        print("✅ 基本流式输出测试通过")
    
    def test_async_streaming(self) -> None:
        """
        测试异步流式输出功能
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试异步流式输出功能 ===")
        
        async def run_async_streaming_test() -> None:
            streaming_chain = self.prompt | self.model | self.str_parser
            
            test_input = {"question": "解释量子计算的基本原理"}
            
            # 收集异步流式输出
            chunks = []
            async for chunk in streaming_chain.astream(test_input):
                chunks.append(chunk)
                if len(chunks) <= 5:
                    print(f"异步流式chunk {len(chunks)}: {repr(chunk)}")
            
            # 验证结果
            self.assertGreater(len(chunks), 1)
            full_response = "".join(chunks)
            self.assertGreater(len(full_response), 0)
            
            print(f"异步流式总共收到 {len(chunks)} 个chunks")
            print(f"异步流式完整响应长度: {len(full_response)}")
            print("✅ 异步流式输出测试通过")
        
        asyncio.run(run_async_streaming_test())
    
    def test_streaming_with_preprocessing(self) -> None:
        """
        测试带预处理的流式输出
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试带预处理的流式输出 ===")
        
        def preprocess_question(question: str) -> dict:
            processed = f"[预处理] {question.strip()}"
            return {"question": processed}
        
        preprocessing_chain = (RunnableLambda(preprocess_question) | 
                              self.prompt | 
                              self.model | 
                              self.str_parser)
        
        test_input = "  机器学习与深度学习的区别？  "
        
        chunks = []
        for chunk in preprocessing_chain.stream(test_input):
            chunks.append(chunk)
        
        self.assertGreater(len(chunks), 0)
        full_response = "".join(chunks)
        
        print(f"预处理输入: '{test_input}'")
        print(f"流式输出chunks数量: {len(chunks)}")
        print(f"预处理流式响应预览: {full_response[:100]}...")
        print("✅ 带预处理的流式输出测试通过")


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True) 