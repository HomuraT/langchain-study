# ChatModels 测试套件

这是一个全面的 ChatOpenAI 模型测试套件，专注于测试基于 `gpt-4o-mini` 模型的各种功能和应用场景。本测试套件涵盖了从基础聊天到高级功能的完整测试范围，确保模型在各种实际应用场景下的可靠性和稳定性。

## 🚀 快速开始

### 安装依赖

```bash
# 安装测试依赖
pip install -r test_requirements.txt

# 或使用 uv
uv pip install -r test_requirements.txt
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest test_basic_chat.py
pytest test_tool_calling.py
pytest test_multimodal.py

# 运行特定测试类
pytest test_basic_chat.py::TestBasicChat
pytest test_tool_calling.py::TestToolCalling
pytest test_multimodal.py::TestMultimodalFeatures

# 运行特定测试方法
pytest test_basic_chat.py::TestBasicChat::test_simple_chat
pytest test_tool_calling.py::TestToolCalling::test_complete_tool_calling_flow
pytest test_multimodal.py::TestMultimodalFeatures::test_image_from_url

# 运行多模态相关的所有测试
pytest test_multimodal.py -k "image or pdf or audio"

# 运行工具调用相关的所有测试
pytest test_tool_calling.py test_advanced_features.py test_multimodal.py -k "tool"
```

## 📋 测试分类

### 🔧 单元测试（Unit Tests）
使用 mock 响应，不需要真实 API：

```bash
# 运行所有单元测试
pytest -m "unit"

# 或者运行除集成测试外的所有测试
pytest -m "not integration"
```

### 🌐 集成测试（Integration Tests）
需要真实的本地 API 服务：

```bash
# 运行集成测试（需要本地API服务）
pytest -m "integration"
```

### ⏱️ 性能测试（Slow Tests）
耗时较长的测试：

```bash
# 跳过慢速测试
pytest -m "not slow"

# 只运行慢速测试
pytest -m "slow"
```

### 🔄 异步测试（Async Tests）
异步功能测试：

```bash
# 运行异步测试
pytest -m "async_test"
```

## 🎯 功能测试详解

### 1. 基础聊天功能 (`test_basic_chat.py`)

#### 功能作用与应用场景
基础聊天功能是所有对话AI应用的核心，主要包括：
- **模型初始化和配置**: 确保模型能够正确配置各种参数（temperature、max_tokens等）
- **简单对话**: 处理用户的基础询问，是所有聊天应用的基本需求
- **系统消息**: 为AI设定角色和行为准则，常用于客服机器人、专业助手等场景
- **多轮对话**: 保持对话上下文，实现连续对话体验
- **参数调优**: 通过temperature等参数控制模型的创造性和确定性

**实际应用场景**:
- 📞 智能客服系统：处理用户咨询和问题解答
- 🎓 在线教育助手：回答学生问题，提供学习指导
- 🏥 医疗咨询助手：提供基础健康信息（需专业审核）
- 🛒 电商购物助手：产品推荐和购买指导
- 💼 企业内部助手：员工问题解答和信息查询

**相关测试样例**:
```python
# 模型初始化测试
def test_model_initialization() -> None:
    """测试模型的基础配置是否正确"""

# 简单对话测试  
def test_simple_chat() -> None:
    """测试基础的问答功能"""

# 系统消息测试
def test_system_message_chat() -> None:
    """测试角色设定和行为指导"""

# 多轮对话测试
def test_multi_turn_conversation() -> None:
    """测试上下文保持能力"""

# 参数配置测试
def test_different_temperatures() -> None:
    """测试不同创造性参数的效果"""

# 批处理测试
def test_batch_processing() -> None:
    """测试同时处理多个对话请求"""
```

### 2. 异步操作功能 (`test_async_operations.py`)

#### 功能作用与应用场景
异步操作是构建高性能、高并发AI应用的关键技术：
- **异步调用**: 非阻塞式API调用，提升应用响应性
- **并发处理**: 同时处理多个用户请求，提高系统吞吐量
- **异步流式输出**: 实时响应用户，改善用户体验
- **性能优化**: 通过并发减少总体响应时间
- **资源管理**: 合理利用系统资源，避免阻塞

**实际应用场景**:
- 🌐 高并发Web应用：同时服务数千用户
- 📱 移动应用后端：快速响应移动端请求
- 🤖 聊天机器人平台：处理多个对话会话
- 📊 数据分析平台：批量处理分析请求
- 🎮 游戏AI：实时响应玩家操作

**相关测试样例**:
```python
# 基础异步调用
def test_basic_async_invoke() -> None:
    """测试异步API调用功能"""

# 异步流式输出
def test_async_streaming() -> None:
    """测试异步实时响应功能"""

# 异步批处理
def test_async_batch_processing() -> None:
    """测试异步批量处理能力"""

# 并发请求处理
def test_async_concurrent_requests() -> None:
    """测试同时处理多个请求的能力"""

# 性能对比测试
def test_async_performance_timing() -> None:
    """对比并发与顺序执行的性能差异"""

# 操作取消测试
def test_async_cancellation() -> None:
    """测试异步操作的中断和取消"""
```

### 3. 流式输出功能 (`test_streaming.py`)

#### 功能作用与应用场景
流式输出提供实时响应体验，让用户看到AI"思考"的过程：
- **实时输出**: 逐步显示生成内容，减少等待感
- **用户体验优化**: 模拟真实对话的逐字显示效果
- **长内容处理**: 对于长文本生成，用户可以提前开始阅读
- **响应式设计**: 支持中断和实时反馈
- **回调处理**: 自定义输出处理逻辑

**实际应用场景**:
- 💬 即时聊天应用：模拟真人聊天的打字效果
- ✍️ 内容创作工具：实时显示文章、代码生成过程
- 📝 写作助手：让用户看到AI的创作思路
- 🎭 创意故事生成：增强互动性和娱乐性
- 🔍 实时搜索问答：边搜索边显示结果

**相关测试样例**:
```python
# 基础流式输出
def test_basic_streaming() -> None:
    """测试基本的流式响应功能"""

# 系统消息流式输出
def test_streaming_with_system_message() -> None:
    """测试带角色设定的流式输出"""

# 回调处理器
def test_streaming_callback_handler() -> None:
    """测试自定义流式输出处理逻辑"""

# 数据格式验证
def test_streaming_chunk_format() -> None:
    """验证流式输出数据的正确格式"""

# 长内容处理
def test_streaming_long_response() -> None:
    """测试长文本的流式输出效果"""

# 功能对比测试
def test_streaming_vs_normal() -> None:
    """对比流式与普通输出的差异"""
```

### 4. 错误处理功能 (`test_error_handling.py`)

#### 功能作用与应用场景
完善的错误处理确保应用在各种异常情况下的稳定性：
- **网络异常处理**: 应对网络不稳定、连接中断等情况
- **API错误处理**: 处理密钥错误、权限问题等API相关错误
- **超时管理**: 合理设置超时时间，避免长时间等待
- **资源限制**: 处理上下文长度超限、速率限制等问题
- **优雅降级**: 在出错时提供友好的用户体验

**实际应用场景**:
- 🌍 全球化应用：应对不同地区的网络环境差异
- 📱 移动应用：处理网络切换、信号不稳定情况
- 🏢 企业级应用：确保高可用性和稳定性
- ☁️ 云服务集成：处理各种云服务的异常情况
- 🔒 安全敏感应用：妥善处理权限和认证错误

**相关测试样例**:
```python
# 网络连接错误
def test_connection_error() -> None:
    """测试网络连接失败的处理"""

# 认证错误处理
def test_invalid_api_key_error() -> None:
    """测试API密钥错误的处理"""

# 超时错误处理
def test_timeout_error() -> None:
    """测试请求超时的处理机制"""

# 资源限制错误
def test_context_length_exceeded_error() -> None:
    """测试上下文长度超限的处理"""

# 流式中断处理
def test_streaming_interruption_error() -> None:
    """测试流式输出中断的恢复"""

# 系统韧性测试
def test_network_resilience() -> None:
    """测试系统的网络恢复能力"""

# 错误恢复测试
def test_error_recovery() -> None:
    """测试从错误状态恢复的能力"""
```

### 5. 高级功能 (`test_advanced_features.py`)

#### 功能作用与应用场景
高级功能支持复杂的AI应用开发需求：
- **结构化输出**: 返回格式化数据，便于程序处理和集成
- **上下文管理**: 智能管理长对话的上下文信息
- **批量处理**: 高效处理大量请求
- **创造性控制**: 通过参数调节模型的创造性和确定性

**实际应用场景**:
- 📊 数据分析：返回结构化的分析结果
- 📝 内容生成：创造性写作和内容创作
- 🏢 企业应用：批量处理业务请求
- 🎯 决策支持：综合多个数据源进行分析决策

**相关测试样例**:
```python
# 结构化输出解析
def test_pydantic_tools_parser() -> None:
    """测试结构化数据的解析处理"""

# 格式化输出
def test_structured_output_with_pydantic() -> None:
    """测试返回特定格式的结构化数据"""

# 上下文保持
def test_multi_turn_conversation_with_context() -> None:
    """测试复杂对话中的上下文管理"""

# 批量处理
def test_batch_processing() -> None:
    """测试批量处理功能"""

# 创造性控制
def test_temperature_and_creativity_effects() -> None:
    """测试参数对AI创造性的影响"""
```

### 6. 多模态功能 (`test_multimodal.py`)

#### 功能作用与应用场景
多模态功能让AI能够处理文本之外的多种数据类型，实现真正的多模态理解：
- **图像处理**: 分析和描述图片内容，支持base64和URL两种输入方式
- **PDF文档处理**: 解析和理解PDF文档内容
- **音频处理**: 分析音频文件内容和特征
- **多模态工具调用**: 结合多模态输入与工具调用功能
- **跨模态理解**: 综合多种数据类型进行分析和决策

**实际应用场景**:
- 🖼️ 图像分析系统：医疗影像诊断、产品质量检测、内容审核
- 📄 文档处理系统：合同分析、报告总结、表格数据提取
- 🎵 音频分析系统：语音转文字、音乐分析、声音识别
- 🤖 智能助手：处理用户上传的各种文件类型
- 🎯 决策支持：综合文本、图像、文档等多维度信息

**相关测试样例**:
```python
# 图像处理测试
def test_image_from_base64_data() -> None:
    """测试base64编码图像的处理能力"""

def test_image_from_url() -> None:
    """测试URL图像的处理能力"""

def test_multiple_images_comparison() -> None:
    """测试多图像对比分析"""

# PDF文档处理测试
def test_pdf_from_base64_data() -> None:
    """测试PDF文档内容解析"""

# 音频处理测试
def test_audio_from_base64_data() -> None:
    """测试音频文件分析"""

# 多模态工具调用测试
def test_multimodal_with_tool_calling() -> None:
    """测试多模态输入结合工具调用"""

# 错误处理测试
def test_invalid_image_url() -> None:
    """测试无效图像URL的错误处理"""

def test_unsupported_file_type() -> None:
    """测试不支持文件类型的错误处理"""
```

### 7. 工具调用功能 (`test_tool_calling.py`)

#### 功能作用与应用场景
工具调用是让AI能够调用外部函数和API的核心功能，极大扩展了AI的能力边界：
- **工具绑定**: 将外部函数绑定到AI模型，使其能够调用
- **单工具调用**: 调用单个工具完成特定任务
- **多工具协同**: 组合多个工具完成复杂任务
- **并行工具调用**: 同时调用多个工具提高效率
- **工具链执行**: 一个工具的输出作为另一个工具的输入
- **错误处理**: 妥善处理工具执行过程中的各种错误
- **异步工具调用**: 支持异步执行工具提高性能
- **流式工具调用**: 结合流式输出实时显示工具执行过程

**实际应用场景**:
- 🧠 智能助手：集成日历、邮件、天气、搜索等多种服务
- 🛠️ 开发工具：代码生成、测试执行、文档生成、API调用
- 🏪 电商平台：价格查询、库存检查、订单处理、支付集成
- 📊 数据分析：数据库查询、计算工具、图表生成
- 🏥 医疗系统：病历查询、药物信息、预约管理
- 🏦 金融服务：账户查询、交易执行、风险计算
- 🎮 游戏AI：游戏状态查询、动作执行、规则检查

**相关测试样例**:
```python
# ================== 基础工具调用测试 ==================

# 单工具绑定
def test_single_tool_binding() -> None:
    """测试单个工具的绑定功能"""

# 单工具调用
def test_single_tool_call() -> None:
    """测试单个工具的调用和参数传递"""

# 多工具绑定
def test_multiple_tools_binding() -> None:
    """测试多个工具的绑定功能"""

# 并行工具调用
def test_parallel_tool_calls() -> None:
    """测试同时调用多个工具"""

# ================== 完整工具调用流程测试 ==================

# 完整流程
def test_complete_tool_calling_flow() -> None:
    """测试完整的工具调用流程：请求->调用->执行->返回"""

# ID匹配验证
def test_tool_message_id_matching() -> None:
    """测试ToolMessage的tool_call_id匹配机制"""

# ================== 复杂工具调用测试 ==================

# 复杂参数工具
def test_complex_tool_with_multiple_parameters() -> None:
    """测试具有多个参数的复杂工具"""

# 可选参数工具
def test_tool_with_optional_parameters() -> None:
    """测试具有可选参数的工具"""

# 嵌套工具调用
def test_nested_tool_calls() -> None:
    """测试工具链式调用（一个工具的结果用于另一个工具）"""

# ================== 工具错误处理测试 ==================

# 执行错误处理
def test_tool_execution_error_handling() -> None:
    """测试工具执行错误的处理机制"""

# 无效参数处理
def test_invalid_tool_parameters() -> None:
    """测试无效工具参数的处理"""

# ================== 异步和流式工具调用测试 ==================

# 异步工具调用
def test_async_tool_calling() -> None:
    """测试异步工具调用功能"""

# 流式工具调用
def test_streaming_with_tools() -> None:
    """测试流式输出与工具调用的结合"""

# ================== 结构化工具输出测试 ==================

# 结构化输出
def test_structured_tool_output() -> None:
    """测试工具的结构化输出处理"""

# ================== 真实世界场景测试 ==================

# 多步骤工作流
def test_multi_step_workflow() -> None:
    """测试多步骤工作流（获取信息->执行操作）"""

# 数据分析场景
def test_data_analysis_scenario() -> None:
    """测试数据分析场景（计算->搜索->解释）"""

# 决策支持场景
def test_decision_support_scenario() -> None:
    """测试决策支持场景（多维度信息收集）"""

# ================== 性能和限制测试 ==================

# 工具调用限制
def test_tool_call_limits() -> None:
    """测试工具调用数量和复杂度限制"""

# 性能计时
def test_tool_performance_timing() -> None:
    """测试工具调用的性能表现"""
```

## ⚙️ 配置说明

### API 配置

测试使用 `src/config/api.py` 中的 `local` 配置：

```python
apis = {
    "local": {
        "base_url": "http://localhost:8212",
        "api_key": "sk-nsbaxS65nDJyGfA8wp5z7pbHxKUjEQBCpN5BKg7E19nLnOgL",
    }
}
```

### 模型配置

所有测试使用 `gpt-4o-mini` 模型，具体参数：

- **model**: `gpt-4o-mini`
- **temperature**: `0.7`（可配置）
- **max_tokens**: `1000`（可配置）
- **timeout**: `30`秒
- **max_retries**: `3`

## 🔧 自定义配置

### 修改测试配置

编辑 `conftest.py` 中的 `test_config` fixture：

```python
@pytest.fixture(scope="session")
def test_config() -> dict:
    return {
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 30,
        "max_retries": 3
    }
```

### 使用不同的 API 配置

创建临时配置用于测试：

```python
def test_with_custom_config():
    config = {
        "base_url": "http://alternative-server:8080",
        "api_key": "alternative-key"
    }
    model = ChatOpenAI(**config)
    # 测试...
```

## 🛠️ 工具调用使用示例

### 基础工具调用示例

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 定义工具
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# 创建模型并绑定工具
model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = model.bind_tools([add_numbers])

# 调用工具
messages = [HumanMessage(content="What is 15 + 27?")]
response = model_with_tools.invoke(messages)

if response.tool_calls:
    # 执行工具
    for tool_call in response.tool_calls:
        tool_output = add_numbers.invoke(tool_call)
        print(f"Result: {tool_output.content}")
```

### 完整工具调用流程示例

```python
# 完整的工具调用对话流程
def complete_tool_calling_example():
    messages = [HumanMessage(content="Calculate 8 * 7 and then add 15")]
    
    # 第一步：模型决定调用工具
    ai_response = model_with_tools.invoke(messages)
    messages.append(ai_response)
    
    # 第二步：执行工具调用
    for tool_call in ai_response.tool_calls:
        tool_output = selected_tool.invoke(tool_call)
        messages.append(tool_output)
    
    # 第三步：获取最终答案
    final_response = model_with_tools.invoke(messages)
    return final_response.content
```

### 多工具协同示例

```python
from datetime import datetime

@tool
def get_weather(location: str) -> dict:
    """Get weather information for a location."""
    return {"location": location, "temperature": 22, "condition": "sunny"}

@tool
def calculate_travel_time(distance: float, speed: float) -> dict:
    """Calculate travel time given distance and speed."""
    time_hours = distance / speed
    return {"time_hours": time_hours, "time_minutes": time_hours * 60}

# 绑定多个工具
tools = [get_weather, calculate_travel_time, add_numbers]
model_with_tools = model.bind_tools(tools)

# 复杂查询示例
query = "What's the weather in Beijing? Also, if I travel 120km at 60km/h, how long will it take?"
response = model_with_tools.invoke([HumanMessage(content=query)])

# 模型会自动选择合适的工具并调用
```

## 📧 消息格式转换

### LangChain消息转换为OpenAI格式

LangChain提供了 `convert_to_openai_messages` 函数，可以将LangChain的消息格式转换为OpenAI API兼容的格式。这在与其他OpenAI兼容的API服务集成时非常有用。

#### 功能说明

- **消息类型支持**: 支持SystemMessage、HumanMessage、AIMessage、ToolMessage等所有LangChain消息类型
- **内容格式处理**: 支持文本内容和多模态内容（图片、文件等）
- **工具调用转换**: 自动处理工具调用格式的转换
- **灵活的文本格式**: 支持字符串和块格式的内容处理

#### 使用示例

```python
from langchain_core.messages import (
    convert_to_openai_messages,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# 示例1: 基础消息转换
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hello, how are you?"),
    AIMessage("I'm doing well, thank you!")
]

# 转换为OpenAI格式
openai_messages = convert_to_openai_messages(messages)
print(openai_messages)
# 输出:
# [
#   {'role': 'system', 'content': 'You are a helpful assistant.'},
#   {'role': 'user', 'content': 'Hello, how are you?'},
#   {'role': 'assistant', 'content': "I'm doing well, thank you!"}
# ]
```

#### 多模态内容转换示例

```python
# 示例2: 多模态内容转换
multimodal_messages = [
    SystemMessage([{"type": "text", "text": "You are an image analysis assistant."}]),
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "What's in this image?"}, 
            {
                "type": "image_url", 
                "image_url": {"url": "data:image/png;base64,'/9j/4AAQSk'"}
            }
        ]
    },
    AIMessage("I can see a beautiful landscape in the image."),
]

openai_messages = convert_to_openai_messages(multimodal_messages)
print(openai_messages)
# 输出:
# [
#   {'role': 'system', 'content': 'You are an image analysis assistant.'},
#   {
#     'role': 'user', 
#     'content': [
#       {'type': 'text', 'text': "What's in this image?"}, 
#       {'type': 'image_url', 'image_url': {'url': "data:image/png;base64,'/9j/4AAQSk'"}}
#     ]
#   },
#   {'role': 'assistant', 'content': 'I can see a beautiful landscape in the image.'}
# ]
```

#### 工具调用转换示例

```python
# 示例3: 工具调用消息转换
tool_messages = [
    SystemMessage("You are a calculator assistant."),
    HumanMessage("Calculate 15 + 27"),
    AIMessage(
        "", 
        tool_calls=[{
            "name": "add_numbers", 
            "args": {"a": 15, "b": 27}, 
            "id": "call_123", 
            "type": "tool_call"
        }]
    ),
    ToolMessage("42", tool_call_id="call_123", name="add_numbers"),
    AIMessage("The result of 15 + 27 is 42."),
]

openai_messages = convert_to_openai_messages(tool_messages)
print(openai_messages)
# 输出:
# [
#   {'role': 'system', 'content': 'You are a calculator assistant.'},
#   {'role': 'user', 'content': 'Calculate 15 + 27'},
#   {
#     'role': 'assistant', 
#     'tool_calls': [{
#       'type': 'function', 
#       'id': 'call_123',
#       'function': {'name': 'add_numbers', 'arguments': '{"a": 15, "b": 27}'}
#     }], 
#     'content': ''
#   },
#   {'role': 'tool', 'name': 'add_numbers', 'content': '42'},
#   {'role': 'assistant', 'content': 'The result of 15 + 27 is 42.'}
# ]
```

#### 文本格式参数

`convert_to_openai_messages` 函数支持 `text_format` 参数来控制文本内容的格式化方式：

```python
# text_format="string" (默认): 尽可能保持字符串格式
messages_string = convert_to_openai_messages(messages, text_format="string")

# text_format="block": 将所有文本内容转换为块格式
messages_block = convert_to_openai_messages(messages, text_format="block")
```

#### 实际应用场景

**API集成场景**:
```python
def send_to_openai_compatible_api(messages):
    """将LangChain消息发送到OpenAI兼容的API"""
    # 转换消息格式
    openai_format = convert_to_openai_messages(messages)
    
    # 发送到第三方API
    response = requests.post(
        "https://api.example.com/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": openai_format,
            "temperature": 0.7
        },
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    return response.json()
```

**消息日志记录场景**:
```python
def log_conversation(messages):
    """记录对话日志为OpenAI格式便于分析"""
    openai_format = convert_to_openai_messages(messages)
    
    # 保存到日志文件
    with open("conversation_log.json", "a") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "messages": openai_format
        }, f)
```

#### 支持的消息类型转换对照表

| LangChain消息类型 | OpenAI格式role | 说明 |
|------------------|----------------|------|
| `SystemMessage` | `system` | 系统指令消息 |
| `HumanMessage` | `user` | 用户输入消息 |
| `AIMessage` | `assistant` | AI回复消息 |
| `ToolMessage` | `tool` | 工具执行结果消息 |
| `FunctionMessage` | `function` | 函数调用结果消息（已废弃） |

## 📊 响应元数据 (Response Metadata)

### 功能说明

许多模型提供商在其聊天生成响应中包含一些元数据信息。这些元数据可以通过 `AIMessage.response_metadata: Dict` 属性访问。根据模型提供商和模型配置的不同，这可能包含令牌计数、日志概率等信息。

### 实际应用场景

- **📈 成本监控**: 跟踪token使用量，计算API调用成本
- **🔍 性能分析**: 监控响应时间和完成状态
- **📋 日志记录**: 记录详细的调用信息用于调试和审计
- **⚡ 优化决策**: 基于元数据优化提示词和参数设置
- **🚨 错误诊断**: 通过finish_reason等信息诊断问题

### 不同提供商的响应元数据示例

#### OpenAI 元数据示例

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
msg = llm.invoke("What's the oldest known example of cuneiform")
print(msg.response_metadata)

# 输出示例:
{
    'token_usage': {
        'completion_tokens': 110,
        'prompt_tokens': 16,
        'total_tokens': 126,
        'completion_tokens_details': {
            'accepted_prediction_tokens': 0,
            'audio_tokens': 0,
            'reasoning_tokens': 0,
            'rejected_prediction_tokens': 0
        },
        'prompt_tokens_details': {
            'audio_tokens': 0, 
            'cached_tokens': 0
        }
    },
    'model_name': 'gpt-4o-mini-2024-07-18',
    'system_fingerprint': 'fp_b8bc95a0ac',
    'id': 'chatcmpl-BDrISvLar6AqcZngBmhajFZXVc2u9',
    'finish_reason': 'stop',
    'logprobs': None
}
```

#### Anthropic 元数据示例

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
msg = llm.invoke("What's the oldest known example of cuneiform")
print(msg.response_metadata)

# 输出示例:
{
    'id': 'msg_01JHnvPqgERY7MZwrvfkmq52',
    'model': 'claude-3-5-sonnet-20241022',
    'stop_reason': 'end_turn',
    'stop_sequence': None,
    'usage': {
        'cache_creation_input_tokens': 0,
        'cache_read_input_tokens': 0,
        'input_tokens': 17,
        'output_tokens': 221
    },
    'model_name': 'claude-3-5-sonnet-20241022'
}
```

#### Google VertexAI 元数据示例

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-2.0-flash-001")
msg = llm.invoke("What's the oldest known example of cuneiform")
print(msg.response_metadata)

# 输出示例:
{
    'is_blocked': False,
    'safety_ratings': [],
    'usage_metadata': {
        'prompt_token_count': 10,
        'candidates_token_count': 55,
        'total_token_count': 65,
        'prompt_tokens_details': [{'modality': 1, 'token_count': 10}],
        'candidates_tokens_details': [{'modality': 1, 'token_count': 55}],
        'cached_content_token_count': 0,
        'cache_tokens_details': []
    },
    'finish_reason': 'STOP',
    'avg_logprobs': -0.251378042047674,
    'model_name': 'gemini-2.0-flash-001'
}
```

#### AWS Bedrock (Anthropic) 元数据示例

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-sonnet-20240229-v1:0")
msg = llm.invoke("What's the oldest known example of cuneiform")
print(msg.response_metadata)

# 输出示例:
{
    'ResponseMetadata': {
        'RequestId': 'ea0ac2ad-3ad5-4a49-9647-274a0c73ac31',
        'HTTPStatusCode': 200,
        'HTTPHeaders': {
            'date': 'Sat, 22 Mar 2025 11:27:46 GMT',
            'content-type': 'application/json',
            'content-length': '1660',
            'connection': 'keep-alive',
            'x-amzn-requestid': 'ea0ac2ad-3ad5-4a49-9647-274a0c73ac31'
        },
        'RetryAttempts': 0
    },
    'stopReason': 'end_turn',
    'metrics': {'latencyMs': [11044]}
}
```

#### MistralAI 元数据示例

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-small-latest")
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
print(msg.response_metadata)

# 输出示例:
{
    'token_usage': {
        'prompt_tokens': 13,
        'total_tokens': 219,
        'completion_tokens': 206
    },
    'model_name': 'mistral-small-latest',
    'model': 'mistral-small-latest',
    'finish_reason': 'stop'
}
```

#### Groq 元数据示例

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")
msg = llm.invoke("What's the oldest known example of cuneiform")
print(msg.response_metadata)

# 输出示例:
{
    'token_usage': {
        'completion_tokens': 184,
        'prompt_tokens': 45,
        'total_tokens': 229,
        'completion_time': 0.245333333,
        'prompt_time': 0.002262803,
        'queue_time': 0.19315161,
        'total_time': 0.247596136
    },
    'model_name': 'llama-3.1-8b-instant',
    'system_fingerprint': 'fp_a56f6eea01',
    'finish_reason': 'stop',
    'logprobs': None
}
```

#### FireworksAI 元数据示例

```python
from langchain_fireworks import ChatFireworks

llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")
msg = llm.invoke("What's the oldest known example of cuneiform")
print(msg.response_metadata)

# 输出示例:
{
    'token_usage': {
        'prompt_tokens': 25,
        'total_tokens': 352,
        'completion_tokens': 327
    },
    'model_name': 'accounts/fireworks/models/llama-v3p1-70b-instruct',
    'system_fingerprint': '',
    'finish_reason': 'stop',
    'logprobs': None
}
```

### 元数据字段说明

#### 通用字段

| 字段名 | 说明 | 用途 |
|--------|------|------|
| `token_usage` | Token使用统计 | 成本计算和优化 |
| `model_name` | 实际使用的模型名称 | 版本跟踪 |
| `finish_reason` | 完成原因 | 判断响应是否完整 |
| `id` | 请求唯一标识符 | 日志追踪和调试 |

#### Token使用字段

| 字段名 | 说明 | 重要性 |
|--------|------|--------|
| `prompt_tokens` | 输入token数量 | 💰 成本计算 |
| `completion_tokens` | 输出token数量 | 💰 成本计算 |
| `total_tokens` | 总token数量 | 📊 使用量监控 |
| `cached_tokens` | 缓存token数量 | ⚡ 性能优化 |

#### 完成状态字段

| finish_reason | 说明 | 处理建议 |
|---------------|------|----------|
| `stop` | 正常完成 | ✅ 无需处理 |
| `length` | 达到最大长度限制 | ⚠️ 考虑增加max_tokens |
| `content_filter` | 内容被过滤 | 🚫 调整输入内容 |
| `tool_calls` | 需要工具调用 | 🔧 执行工具调用 |

### 实际使用示例

#### 成本监控示例

```python
def monitor_api_costs(messages: list) -> dict:
    """监控API调用成本"""
    response = llm.invoke(messages)
    
    # 获取token使用情况
    metadata = response.response_metadata
    token_usage = metadata.get('token_usage', {})
    
    # 计算成本（以OpenAI为例）
    prompt_cost = token_usage.get('prompt_tokens', 0) * 0.00015 / 1000
    completion_cost = token_usage.get('completion_tokens', 0) * 0.0006 / 1000
    total_cost = prompt_cost + completion_cost
    
    return {
        'response': response.content,
        'cost_info': {
            'prompt_tokens': token_usage.get('prompt_tokens', 0),
            'completion_tokens': token_usage.get('completion_tokens', 0),
            'total_tokens': token_usage.get('total_tokens', 0),
            'estimated_cost_usd': total_cost
        },
        'model_info': {
            'model': metadata.get('model_name'),
            'finish_reason': metadata.get('finish_reason')
        }
    }

# 使用示例
result = monitor_api_costs([HumanMessage("Explain quantum computing")])
print(f"Cost: ${result['cost_info']['estimated_cost_usd']:.6f}")
print(f"Tokens used: {result['cost_info']['total_tokens']}")
```

#### 性能分析示例

```python
import time
from datetime import datetime

def analyze_performance(messages: list) -> dict:
    """分析API调用性能"""
    start_time = time.time()
    timestamp = datetime.now()
    
    response = llm.invoke(messages)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    metadata = response.response_metadata
    
    return {
        'timestamp': timestamp.isoformat(),
        'response_time_seconds': response_time,
        'token_info': metadata.get('token_usage', {}),
        'model_info': metadata.get('model_name'),
        'finish_reason': metadata.get('finish_reason'),
        'request_id': metadata.get('id'),
        'tokens_per_second': metadata.get('token_usage', {}).get('completion_tokens', 0) / response_time if response_time > 0 else 0
    }

# 使用示例
perf_data = analyze_performance([HumanMessage("Write a short story")])
print(f"Response time: {perf_data['response_time_seconds']:.2f}s")
print(f"Tokens/second: {perf_data['tokens_per_second']:.1f}")
```

#### 错误诊断示例

```python
def diagnose_response(response) -> dict:
    """诊断响应状态"""
    metadata = response.response_metadata
    finish_reason = metadata.get('finish_reason')
    
    diagnosis = {
        'status': 'unknown',
        'message': '',
        'suggestions': []
    }
    
    if finish_reason == 'stop':
        diagnosis.update({
            'status': 'success',
            'message': '响应正常完成'
        })
    elif finish_reason == 'length':
        diagnosis.update({
            'status': 'warning',
            'message': '响应因长度限制被截断',
            'suggestions': [
                '增加max_tokens参数',
                '拆分成多个较短的请求',
                '优化提示词以获得更简洁的回答'
            ]
        })
    elif finish_reason == 'content_filter':
        diagnosis.update({
            'status': 'error',
            'message': '内容被安全过滤器拦截',
            'suggestions': [
                '修改输入内容避免敏感话题',
                '使用更中性的表达方式',
                '检查内容是否符合使用政策'
            ]
        })
    
    return diagnosis

# 使用示例
response = llm.invoke([HumanMessage("Your question here")])
diagnosis = diagnose_response(response)
print(f"Status: {diagnosis['status']}")
print(f"Message: {diagnosis['message']}")
if diagnosis['suggestions']:
    print("Suggestions:")
    for suggestion in diagnosis['suggestions']:
        print(f"  - {suggestion}")
```

### 测试中的元数据验证

在测试套件中，我们可以验证响应元数据的正确性：

```python
def test_response_metadata():
    """测试响应元数据的完整性"""
    response = llm.invoke([HumanMessage("Hello")])
    
    # 验证元数据存在
    assert hasattr(response, 'response_metadata')
    assert isinstance(response.response_metadata, dict)
    
    # 验证必要字段
    metadata = response.response_metadata
    assert 'model_name' in metadata
    assert 'finish_reason' in metadata
    
    # 验证token使用信息（如果提供商支持）
    if 'token_usage' in metadata:
        token_usage = metadata['token_usage']
        assert 'total_tokens' in token_usage
        assert token_usage['total_tokens'] > 0
        
    # 验证完成状态
    assert metadata['finish_reason'] in ['stop', 'length', 'content_filter', 'tool_calls']
    
    print(f"✅ 元数据验证通过: {metadata['model_name']}")
    print(f"   完成状态: {metadata['finish_reason']}")
    if 'token_usage' in metadata:
        print(f"   Token使用: {metadata['token_usage']['total_tokens']}")
```