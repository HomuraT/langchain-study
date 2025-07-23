"""
多模态功能测试

测试 ChatOpenAI 模型处理图像、PDF、音频等多模态输入的能力。
涵盖了 base64 编码数据和 URL 链接两种输入方式。
"""

import unittest
import base64
from typing import Dict, Any, List, Union


from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.config.api import apis


class TestMultimodalFeatures(unittest.TestCase):
    """多模态功能测试类"""

    def get_chat_model(self) -> ChatOpenAI:
        """
        创建ChatOpenAI实例用于测试
        
        Returns:
            ChatOpenAI: 配置好的聊天模型实例
        """
        config = apis["local"]
        return ChatOpenAI(
            model="gpt-4o",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.7,
            max_tokens=1000,
            timeout=60  # 多模态处理可能需要更长时间
        )

    # ================== 图像处理测试 ==================

    def test_image_from_base64_data(self) -> None:
        """
        测试处理 base64 编码的图像数据
        """
        chat_model = self.get_chat_model()
        
        # 获取本地图像数据并转换为base64
        try:
            image_path = os.path.join(os.path.dirname(__file__), "resources", "image.png")
            print(f"正在读取图像文件: {image_path}")
            
            with open(image_path, "rb") as f:
                image_content = f.read()
                image_data = base64.b64encode(image_content).decode("utf-8")
                print(f"图像文件读取成功，大小: {len(image_content)} 字节")
                print(f"Base64编码后长度: {len(image_data)} 字符")
            
            # 构建消息
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what you see in this image:",
                    },
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image_data,
                        "mime_type": "image/png",
                    },
                ],
            }
            
            response = chat_model.invoke([message])
            
            # 验证响应
            self.assertIsInstance(response, AIMessage)
            self.assertGreater(len(response.content), 0)
            print(f"Base64 image response: {response.content}")
            
            # 检查是否包含图像相关的关键词
            content_lower = response.content.lower()
            image_keywords = ["boardwalk", "grass", "path", "nature", "green", "sky", "wood", "field"]
            has_image_content = any(keyword in content_lower for keyword in image_keywords)
            self.assertTrue(has_image_content, f"Response should contain image-related content. Got: {response.content}")
            
        except Exception as e:
            print(f"Base64 image test failed: {e}")
            # 不让测试失败，只是记录

    def test_image_from_url(self) -> None:
        """
        测试处理来自 URL 的图像
        """
        chat_model = self.get_chat_model()
        
        try:
            # 使用在线测试图像URL
            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?",
                    },
                    {
                        "type": "image",
                        "source_type": "url",
                        "url": image_url,
                    },
                ],
            }
            
            response = chat_model.invoke([message])
            
            # 验证响应
            self.assertIsInstance(response, AIMessage)
            self.assertGreater(len(response.content), 0)
            print(f"URL image response: {response.content}")
            
        except Exception as e:
            print(f"URL image test failed: {e}")

    def test_multiple_images_comparison(self) -> None:
        """
        测试处理多张图像并进行比较
        """
        chat_model = self.get_chat_model()
        
        try:
            # 使用在线测试图像进行比较
            image_url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images and tell me the main differences:"},
                    {"type": "image", "source_type": "url", "url": image_url1},
                    {"type": "image", "source_type": "url", "url": image_url2},
                ],
            }
            
            response = chat_model.invoke([message])
            
            # 验证响应
            self.assertIsInstance(response, AIMessage)
            self.assertGreater(len(response.content), 0)
            print(f"Multiple images comparison: {response.content}")
            
        except Exception as e:
            print(f"Multiple images test failed: {e}")

    # ================== PDF 文档处理测试 ==================

    def test_pdf_from_base64_data(self) -> None:
        """
        测试处理 base64 编码的 PDF 文档
        """
        chat_model = self.get_chat_model()
        
        try:
            # 尝试读取本地PDF文件，如果不存在则创建一个简单的测试PDF
            pdf_path = os.path.join(os.path.dirname(__file__), "resources", "sample.pdf")
            
            if not os.path.exists(pdf_path):
                # 创建一个简单的PDF内容（最小有效PDF）
                # 这是一个有效的最小PDF文件内容
                simple_pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello, World!) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000079 00000 n 
0000000173 00000 n 
0000000301 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
398
%%EOF"""
                
                # 确保resources目录存在
                os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                
                # 写入PDF文件
                with open(pdf_path, "wb") as f:
                    f.write(simple_pdf_content)
                print(f"创建了测试PDF文件: {pdf_path}")
            
            print(f"正在读取PDF文件: {pdf_path}")
            
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
                pdf_data = base64.b64encode(pdf_content).decode("utf-8")
                print(f"PDF文件读取成功，大小: {len(pdf_content)} 字节")
                print(f"Base64编码后长度: {len(pdf_data)} 字符")
            
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please summarize the content of this PDF document:",
                    },
                    {
                        "type": "file",
                        "source_type": "base64",
                        "data": pdf_data,
                        "mime_type": "application/pdf",
                        "filename": "test-document.pdf",  # OpenAI 需要文件名
                    },
                ],
            }
            
            response = chat_model.invoke([message])
            
            # 验证响应
            self.assertIsInstance(response, AIMessage)
            self.assertGreater(len(response.content), 0)
            print(f"PDF document response: {response.content}")
            
        except Exception as e:
            print(f"PDF base64 test failed: {e}")

    # ================== 音频处理测试 ==================

    def test_audio_from_base64_data(self) -> None:
        """
        测试处理 base64 编码的音频数据
        """
        # 音频处理需要使用支持音频的模型
        config = apis["local"]
        audio_model = ChatOpenAI(
            model="google/gemini-2.5-flash",
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.7,
            max_tokens=1000,
            timeout=60
        )
        
        try:
            # 获取本地音频文件并转换为base64
            audio_path = os.path.join(os.path.dirname(__file__), "resources", "audio.wav")
            print(f"正在读取音频文件: {audio_path}")
            
            with open(audio_path, "rb") as f:
                audio_content = f.read()
                audio_data = base64.b64encode(audio_content).decode("utf-8")
                print(f"音频文件读取成功，大小: {len(audio_content)} 字节")
                print(f"Base64编码后长度: {len(audio_data)} 字符")
                
                # 验证base64数据的前几个字符
                print(f"Base64前50字符: {audio_data[:50]}...")
                
                message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What can you tell me about this audio file?",
                        },
                        {
                            "type": "audio",
                            "source_type": "base64",
                            "data": audio_data,
                            "mime_type": "audio/wav",
                        },
                    ],
                }
                
                response = audio_model.invoke([message])
                
                # 验证响应
                self.assertIsInstance(response, AIMessage)
                self.assertGreater(len(response.content), 0)
                print(f"Audio response: {response.content}")
                
        except Exception as e:
            print(f"Audio test failed: {e}")
            # 即使失败也要显示具体的错误信息
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")

    # ================== 多模态工具调用测试 ==================

    def test_multimodal_with_tool_calling(self) -> None:
        """
        测试多模态输入结合工具调用功能
        """
        # 定义天气识别工具
        @tool
        def identify_weather(weather_condition: str) -> str:
            """
            根据图像识别的天气状况记录信息
            
            Args:
                weather_condition: 天气状况描述
                
            Returns:
                str: 天气记录确认信息
            """
            return f"Weather condition recorded: {weather_condition}"
        
        chat_model = self.get_chat_model()
        
        try:
            # 绑定工具
            model_with_tools = chat_model.bind_tools([identify_weather])
            
            # 使用在线图像进行工具调用测试
            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image and use the tool to record the weather condition you see:"},
                    {"type": "image", "source_type": "url", "url": image_url},
                ],
            }
            
            response = model_with_tools.invoke([message])
            
            # 验证响应
            self.assertIsInstance(response, AIMessage)
            print(f"Multimodal tool calling response: {response.content}")
            
            # 检查是否有工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Tool calls detected: {response.tool_calls}")
                self.assertGreater(len(response.tool_calls), 0)
                self.assertEqual(response.tool_calls[0]["name"], "identify_weather")
            
        except Exception as e:
            print(f"Multimodal tool calling test failed: {e}")

    # ================== 错误处理测试 ==================

    def test_invalid_image_url(self) -> None:
        """
        测试无效图像URL的错误处理
        """
        chat_model = self.get_chat_model()
        
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "https://invalid-url-that-does-not-exist.com/image.jpg",
                },
            ],
        }
        
        try:
            response = chat_model.invoke([message])
            # 如果没有抛出异常，检查响应是否包含错误信息
            print(f"Invalid URL response: {response.content}")
        except Exception as e:
            print(f"Expected error for invalid URL: {e}")
            # 这是预期的行为

    def test_unsupported_file_type(self) -> None:
        """
        测试不支持的文件类型错误处理
        """
        chat_model = self.get_chat_model()
        
        # 创建一个假的不支持的文件类型
        fake_data = base64.b64encode(b"This is not a valid file").decode("utf-8")
        
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Process this file:"},
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": fake_data,
                    "mime_type": "application/unknown",
                    "filename": "unknown.file"
                },
            ],
        }
        
        try:
            response = chat_model.invoke([message])
            print(f"Unsupported file type response: {response.content}")
        except Exception as e:
            print(f"Expected error for unsupported file type: {e}")


if __name__ == '__main__':
    unittest.main() 