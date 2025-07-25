# ChatModels 指南

ChatModels 是 LangChain 中用于与聊天模型（如 OpenAI 的 GPT 系列）进行交互的核心组件。本指南将全面介绍 ChatModels 的各种功能和使用方法。

详细测试样例以及源码，可以访问[github](https://github.com/HomuraT/langchain-study/tree/main/unitests/test_chatmodels)

## 📋 目录

1. [基础概念](#基础概念)
2. [基础聊天功能](#基础聊天功能)
3. [异步操作](#异步操作)
4. [流式输出](#流式输出)
5. [多模态功能](#多模态功能)
6. [工具调用](#工具调用)
7. [高级功能](#高级功能)
8. [错误处理](#错误处理)
9. [响应元数据](#响应元数据)

## 基础概念

### ChatModels 是什么？

ChatModels 是 LangChain 的聊天模型抽象，提供了与各种大语言模型进行对话的统一接口。它支持多种功能，从简单的文本对话到复杂的多模态交互和工具调用。

**相关链接**：
- [LangChain Chat Models 概念](https://python.langchain.com/docs/concepts/chat_models/)
- [如何使用聊天模型](https://python.langchain.com/docs/how_to/chat_models_universal_init/)

### 核心组件

- **ChatOpenAI**: OpenAI 聊天模型的实现
- **Messages**: 对话消息的数据结构
- **Tools**: 外部功能调用的抽象
- **Callbacks**: 回调处理机制

## 基础聊天功能

### 模型初始化

ChatModels 的使用从创建模型实例开始：

```python
from langchain_openai import ChatOpenAI

# 创建模型实例
model = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="http://localhost:8212",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)
```

**参数说明**：
- `model`: 使用的模型名称
- `temperature`: 控制输出的随机性（0-2，0最确定，2最随机）
- `max_tokens`: 最大输出token数量
- `timeout`: 请求超时时间（秒）

### 消息类型

LangChain 提供了多种消息类型来构建对话：

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 系统消息 - 定义AI的角色和行为
system_msg = SystemMessage(content="You are a helpful assistant.")

# 人类消息 - 用户输入
human_msg = HumanMessage(content="Hello, how are you?")

# AI消息 - AI的回复
ai_msg = AIMessage(content="I'm doing well, thank you!")
```

### 基础对话

**输入**: 消息列表
**输出**: AIMessage 对象
**原理**: 模型接收消息历史，生成下一个回复

```python
# 简单对话
response = model.invoke([HumanMessage(content="Hello!")])
print(response.content)

# 带系统消息的对话
messages = [
    SystemMessage(content="You are a math tutor."),
    HumanMessage(content="What is 2+2?")
]
response = model.invoke(messages)
print(response.content)
```

### 多轮对话

**原理**: 通过维护消息历史来保持对话上下文

```python
# 构建对话历史
conversation = [
    HumanMessage(content="My name is Alice."),
    AIMessage(content="Nice to meet you, Alice!"),
    HumanMessage(content="What's my name?")
]

response = model.invoke(conversation)
print(response.content)  # 应该包含 "Alice"
```

### 批处理

**输入**: 消息批次列表
**输出**: AIMessage 列表
**原理**: 并行处理多个独立的对话请求

```python
message_batches = [
    [HumanMessage(content="Hello 1")],
    [HumanMessage(content="Hello 2")],
    [HumanMessage(content="Hello 3")]
]

responses = model.batch(message_batches)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response.content}")
```

**相关链接**：
- [如何进行聊天对话](https://python.langchain.com/docs/how_to/chatbots/)

## 异步操作

### 概念定义

异步操作允许非阻塞式的模型调用，提高应用程序的并发性能和响应速度。

### 异步调用

**输入**: 消息列表
**输出**: AIMessage 对象（异步）
**原理**: 使用 Python asyncio 实现非阻塞调用

```python
import asyncio

async def async_chat():
    response = await model.ainvoke([HumanMessage(content="Hello async!")])
    return response.content

# 运行异步函数
result = asyncio.run(async_chat())
print(result)
```

### 异步流式输出

**原理**: 结合异步和流式输出，实现非阻塞的实时响应

```python
async def async_stream_chat():
    async for chunk in model.astream([HumanMessage(content="Count to 5")]):
        print(chunk.content, end="", flush=True)

asyncio.run(async_stream_chat())
```

### 并发请求处理

**原理**: 利用异步并发处理多个请求，显著提升吞吐量

```python
async def concurrent_requests():
    tasks = [
        model.ainvoke([HumanMessage(content=f"Request {i}")])
        for i in range(5)
    ]
    
    responses = await asyncio.gather(*tasks)
    return [r.content for r in responses]

results = asyncio.run(concurrent_requests())
```

### 异步批处理

```python
async def async_batch():
    message_batches = [
        [HumanMessage(content=f"Batch {i}")]
        for i in range(3)
    ]
    
    responses = await model.abatch(message_batches)
    return responses

results = asyncio.run(async_batch())
```

**相关链接**：
- [异步编程概念](https://python.langchain.com/docs/concepts/async/)

## 流式输出

### 概念定义

流式输出允许模型逐步生成和返回响应内容，而不是等待完整响应后一次性返回，从而提供更好的用户体验。

### 基础流式输出

**输入**: 消息列表
**输出**: ChatGenerationChunk 迭代器
**原理**: 模型生成内容时实时返回部分结果

```python
# 启用流式输出
streaming_model = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True
)

# 获取流式响应
for chunk in streaming_model.stream([HumanMessage(content="Tell me a story")]):
    print(chunk.content, end="", flush=True)
```

### 流式回调处理

**原理**: 通过回调函数自定义流式输出的处理逻辑

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

class CustomCallbackHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Token: {token}")

model_with_callback = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[CustomCallbackHandler()]
)
```

### 流式 vs 普通输出对比

```python
# 流式输出 - 实时显示
chunks = list(streaming_model.stream([HumanMessage(content="What is AI?")]))
streaming_response = "".join([chunk.content for chunk in chunks])

# 普通输出 - 等待完整响应
normal_response = model.invoke([HumanMessage(content="What is AI?")])

print(f"Streaming: {streaming_response}")
print(f"Normal: {normal_response.content}")
```

**相关链接**：
- [如何流式输出聊天模型响应](https://python.langchain.com/docs/how_to/streaming/)

## 多模态功能

### 概念定义

多模态功能允许 ChatModels 处理文本之外的其他数据类型，如图像、PDF文档、音频等，实现真正的多模态理解和交互。

### 图像处理

**输入**: 包含图像数据的消息
**输出**: 基于图像内容的文本描述
**原理**: 模型使用视觉编码器分析图像内容

```python
# 处理 base64 编码的图像
import base64

with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image:"},
        {
            "type": "image",
            "source_type": "base64",
            "data": image_data,
            "mime_type": "image/png"
        }
    ]
}

response = model.invoke([message])
print(response.content)
```

```python
# 处理 URL 图像
message = {
    "role": "user", 
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image",
            "source_type": "url", 
            "url": "https://example.com/image.jpg"
        }
    ]
}

response = model.invoke([message])
```

### PDF 文档处理

**原理**: 模型解析PDF文档结构和文本内容

```python
# 处理 PDF 文档
with open("document.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode("utf-8")

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Summarize this PDF:"},
        {
            "type": "file",
            "source_type": "base64",
            "data": pdf_data,
            "mime_type": "application/pdf",
            "filename": "document.pdf"
        }
    ]
}

response = model.invoke([message])
```

### 多图像对比

```python
# 对比多张图像
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Compare these images:"},
        {"type": "image", "source_type": "url", "url": "image1.jpg"},
        {"type": "image", "source_type": "url", "url": "image2.jpg"}
    ]
}

response = model.invoke([message])
```

**相关链接**：
- [如何传递多模态数据](https://python.langchain.com/docs/how_to/multimodal_inputs/)

## 工具调用

### 概念定义

工具调用（Tool Calling）是让 AI 模型能够调用外部函数和 API 的核心功能，极大扩展了模型的能力边界。

### 工具定义

**原理**: 使用装饰器定义工具，LangChain 自动生成工具描述供模型使用

```python
from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of the two numbers
    """
    return a + b

@tool
def get_weather(location: str) -> dict:
    """Get weather information for a location.
    
    Args:
        location: City name
        
    Returns:
        Weather data dictionary
    """
    return {
        "location": location,
        "temperature": 22,
        "description": "sunny"
    }
```

### 工具绑定和调用

**输入**: 绑定工具的模型和消息
**输出**: 包含工具调用的 AIMessage
**原理**: 模型识别需要使用工具的场景，生成工具调用指令

```python
# 绑定工具到模型
model_with_tools = model.bind_tools([add_numbers, get_weather])

# 请求计算
response = model_with_tools.invoke([
    HumanMessage(content="What is 15 + 27?")
])

# 检查工具调用
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
```

### 完整工具调用流程

**原理**: 完整的工具调用包括：请求→工具调用→执行→结果返回→最终回答

```python
from langchain_core.messages import ToolMessage

def complete_tool_calling(query: str):
    # 1. 发送查询
    messages = [HumanMessage(content=query)]
    
    # 2. 模型决定调用工具
    ai_response = model_with_tools.invoke(messages)
    messages.append(ai_response)
    
    # 3. 执行工具调用
    for tool_call in ai_response.tool_calls:
        if tool_call["name"] == "add_numbers":
            result = add_numbers.invoke(tool_call)
        elif tool_call["name"] == "get_weather":
            result = get_weather.invoke(tool_call)
        
        # 4. 添加工具结果
        tool_message = ToolMessage(
            content=str(result.content),
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        )
        messages.append(tool_message)
    
    # 5. 获取最终回答
    final_response = model_with_tools.invoke(messages)
    return final_response.content

# 使用示例
result = complete_tool_calling("Add 15 and 27, then tell me the weather in Beijing")
```

### 并行工具调用

**原理**: 模型可以同时调用多个工具提高效率

```python
response = model_with_tools.invoke([
    HumanMessage(content="Calculate 5+3 and get weather for Tokyo")
])

# 可能同时调用 add_numbers 和 get_weather
for tool_call in response.tool_calls:
    print(f"Parallel tool call: {tool_call['name']}")
```

### 复杂工具示例

```python
@tool
def search_database(query: str, table: str) -> list:
    """Search database for information.
    
    Args:
        query: Search query
        table: Database table name
        
    Returns:
        List of matching records
    """
    # 模拟数据库搜索
    return [{"id": 1, "name": "Result 1"}]

@tool  
def send_email(recipient: str, subject: str, body: str) -> bool:
    """Send an email.
    
    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body
        
    Returns:
        Success status
    """
    print(f"Email sent to {recipient}")
    return True
```

**相关链接**：
- [如何进行工具/函数调用](https://python.langchain.com/docs/how_to/tool_calling/)
- [如何创建工具](https://python.langchain.com/docs/how_to/custom_tools/)

## 高级功能

### 结构化输出

**概念**: 让模型返回符合特定数据结构的格式化响应
**原理**: 使用 Pydantic 模型约束输出格式

```python
from pydantic import BaseModel, Field

class WeatherInfo(BaseModel):
    """天气信息结构"""
    location: str = Field(description="地点名称")
    temperature: float = Field(description="温度")
    humidity: int = Field(description="湿度百分比")
    description: str = Field(description="天气描述")

# 创建结构化输出模型
structured_model = model.with_structured_output(WeatherInfo)

# 获取结构化响应
weather = structured_model.invoke([
    HumanMessage(content="Tell me about the weather in Beijing")
])

print(f"Location: {weather.location}")
print(f"Temperature: {weather.temperature}°C")
```

### 复杂结构化输出

```python
class Person(BaseModel):
    name: str = Field(description="人员姓名")
    age: int = Field(description="人员年龄")
    skills: list[str] = Field(description="技能列表")

class Team(BaseModel):
    team_name: str = Field(description="团队名称")
    members: list[Person] = Field(description="团队成员")
    project: str = Field(description="项目名称")

team_model = model.with_structured_output(Team)
result = team_model.invoke([
    HumanMessage(content="Create a 3-person development team for a web project")
])

print(f"Team: {result.team_name}")
for member in result.members:
    print(f"- {member.name}, {member.age}, Skills: {member.skills}")
```

### Pydantic 工具解析器

```python
from langchain_core.output_parsers import PydanticToolsParser

@tool
def calculate_area(length: float, width: float) -> dict:
    """Calculate rectangle area."""
    return {"area": length * width}

tools = [calculate_area]
parser = PydanticToolsParser(tools=tools)

model_with_parser = model.bind_tools(tools) | parser
```

### 上下文管理

**原理**: 智能管理长对话的上下文信息，避免超出模型限制

```python
def manage_long_conversation(messages: list, max_tokens: int = 4000):
    """管理长对话上下文"""
    # 估算token数量（简化计算）
    total_tokens = sum(len(msg.content.split()) for msg in messages)
    
    if total_tokens > max_tokens:
        # 保留系统消息和最近的对话
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        recent_msgs = messages[-10:]  # 保留最近10条消息
        messages = system_msgs + recent_msgs
    
    return model.invoke(messages)
```

**相关链接**：
- [如何返回结构化数据](https://python.langchain.com/docs/how_to/structured_output/)

## 错误处理

### 网络连接错误

**原理**: 处理网络不稳定、服务器不可达等连接问题

```python
import requests
from openai import APIConnectionError

def robust_invoke(messages, max_retries=3):
    """带重试机制的模型调用"""
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except APIConnectionError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Connection failed, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)  # 指数退避
```

### API 错误处理

```python
from openai import AuthenticationError, RateLimitError, BadRequestError

def handle_api_errors(messages):
    """处理各种API错误"""
    try:
        return model.invoke(messages)
    except AuthenticationError:
        print("API密钥无效，请检查配置")
    except RateLimitError:
        print("达到速率限制，请稍后重试")
    except BadRequestError as e:
        print(f"请求参数错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")
```

### 超时处理

```python
# 配置超时参数
timeout_model = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=30,  # 30秒超时
    max_retries=3
)

try:
    response = timeout_model.invoke(messages)
except TimeoutError:
    print("请求超时，请检查网络连接")
```

### 内容长度限制

```python
def check_context_length(messages, max_tokens=8000):
    """检查上下文长度是否超限"""
    # 简化的token计算
    total_tokens = sum(len(msg.content.split()) * 1.3 for msg in messages)
    
    if total_tokens > max_tokens:
        raise ValueError(f"Context too long: {total_tokens} > {max_tokens}")
    
    return model.invoke(messages)
```

**相关链接**：
- [如何处理速率限制](https://python.langchain.com/docs/how_to/rate_limits/)

## 响应元数据

### 概念定义

响应元数据包含了模型调用的详细信息，如token使用量、完成状态、模型版本等，对于监控、调试和成本控制非常重要。

### 访问元数据

**原理**: 每个 AIMessage 都包含 response_metadata 属性

```python
response = model.invoke([HumanMessage(content="Hello")])

# 访问元数据
metadata = response.response_metadata
print(f"Model: {metadata.get('model_name')}")
print(f"Finish reason: {metadata.get('finish_reason')}")

# Token使用信息
if 'token_usage' in metadata:
    usage = metadata['token_usage']
    print(f"Prompt tokens: {usage.get('prompt_tokens')}")
    print(f"Completion tokens: {usage.get('completion_tokens')}")
    print(f"Total tokens: {usage.get('total_tokens')}")
```

### 成本监控

```python
def calculate_cost(response):
    """计算API调用成本"""
    metadata = response.response_metadata
    token_usage = metadata.get('token_usage', {})
    
    # OpenAI pricing (示例价格)
    prompt_cost = token_usage.get('prompt_tokens', 0) * 0.00015 / 1000
    completion_cost = token_usage.get('completion_tokens', 0) * 0.0006 / 1000
    
    return {
        'prompt_cost': prompt_cost,
        'completion_cost': completion_cost,
        'total_cost': prompt_cost + completion_cost,
        'total_tokens': token_usage.get('total_tokens', 0)
    }

response = model.invoke([HumanMessage(content="Explain quantum computing")])
cost_info = calculate_cost(response)
print(f"Total cost: ${cost_info['total_cost']:.6f}")
```

### 性能监控

```python
import time

def monitor_performance(messages):
    """监控模型性能"""
    start_time = time.time()
    response = model.invoke(messages)
    end_time = time.time()
    
    response_time = end_time - start_time
    metadata = response.response_metadata
    token_usage = metadata.get('token_usage', {})
    
    return {
        'response_time': response_time,
        'tokens_per_second': token_usage.get('completion_tokens', 0) / response_time,
        'finish_reason': metadata.get('finish_reason'),
        'model': metadata.get('model_name')
    }

perf_data = monitor_performance([HumanMessage(content="Write a short story")])
print(f"Response time: {perf_data['response_time']:.2f}s")
print(f"Tokens/sec: {perf_data['tokens_per_second']:.1f}")
```

**相关链接**：
- [响应元数据概念](https://python.langchain.com/docs/how_to/response_metadata/)

## 消息格式转换

### LangChain 到 OpenAI 格式转换

**概念**: 将 LangChain 消息格式转换为 OpenAI API 兼容格式
**用途**: 与其他 OpenAI 兼容服务集成、日志记录、调试分析

```python
from langchain_core.messages import convert_to_openai_messages

# LangChain 消息
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hello!"),
    AIMessage("Hi there!")
]

# 转换为 OpenAI 格式
openai_messages = convert_to_openai_messages(messages)
print(openai_messages)
# 输出:
# [
#   {'role': 'system', 'content': 'You are a helpful assistant.'},
#   {'role': 'user', 'content': 'Hello!'},
#   {'role': 'assistant', 'content': 'Hi there!'}
# ]
```

### 工具调用格式转换

```python
# 包含工具调用的消息
tool_messages = [
    HumanMessage("Calculate 15 + 27"),
    AIMessage("", tool_calls=[{
        "name": "add_numbers",
        "args": {"a": 15, "b": 27},
        "id": "call_123"
    }]),
    ToolMessage("42", tool_call_id="call_123", name="add_numbers")
]

openai_format = convert_to_openai_messages(tool_messages)
```

## 其他

### 模型配置

```python
# 生产环境推荐配置
production_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,           # 平衡创造性和确定性
    max_tokens=1000,           # 控制响应长度
    timeout=60,                # 合理的超时时间
    max_retries=3,             # 自动重试
    streaming=True             # 提升用户体验
)
```

### 多模态

```python
def process_multimodal_safely(text, image_data=None):
    """安全处理多模态输入"""
    content = [{"type": "text", "text": text}]
    
    if image_data:
        # 检查图像大小
        if len(image_data) > 20 * 1024 * 1024:  # 20MB limit
            raise ValueError("Image too large")
        
        content.append({
            "type": "image",
            "source_type": "base64", 
            "data": image_data,
            "mime_type": "image/jpeg"
        })
    
    return model.invoke([{"role": "user", "content": content}])
```

## 相关链接

- [LangChain 官方文档](https://python.langchain.com/docs/)
- [Chat Models 概念指南](https://python.langchain.com/docs/concepts/chat_models/)
- [How-to 指南总览](https://python.langchain.com/docs/how_to/)
- [工具调用指南](https://python.langchain.com/docs/how_to/tool_calling/)
- [多模态输入指南](https://python.langchain.com/docs/how_to/multimodal_inputs/)
- [流式输出指南](https://python.langchain.com/docs/how_to/streaming/)
- [异步编程指南](https://python.langchain.com/docs/concepts/async/)
- [结构化输出指南](https://python.langchain.com/docs/how_to/structured_output/)
