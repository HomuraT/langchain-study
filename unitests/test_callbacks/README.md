# LangChain Callbacks 测试套件

这是一个全面的 LangChain 回调系统测试套件，专注于测试所有回调事件的**触发时机**、**参数传递**和**继承机制**。本测试套件深入验证回调系统在各种实际应用场景下的可靠性和正确性。

## 🚀 快速开始

### 安装依赖

```bash
# 使用项目的虚拟环境
source .venv/bin/activate

# 依赖已在pyproject.toml中配置
uv sync
```

### 运行测试

```bash
# 运行所有回调测试
python unitests/test_callbacks/run_all_tests.py

# 运行特定测试模块
python unitests/test_callbacks/run_all_tests.py --tests handlers
python unitests/test_callbacks/run_all_tests.py --tests inheritance

# 静默模式（只显示摘要）
python unitests/test_callbacks/run_all_tests.py --quiet

# 列出所有可用测试
python unitests/test_callbacks/run_all_tests.py --list
```

### 使用 unittest 直接运行

```bash
# 运行所有测试
python -m unittest discover unitests/test_callbacks -v

# 运行单个测试文件
python -m unittest unitests.test_callbacks.test_callback_handlers -v
python -m unittest unitests.test_callbacks.test_callback_inheritance -v

# 运行特定测试方法
python -m unittest unitests.test_callbacks.test_callback_handlers.TestCallbackHandlers.test_chat_model_callbacks -v
```

## 📋 回调事件完整列表

### 🤖 Chat Model 事件

| 事件名 | 触发时机 | 参数 | 用途 |
|--------|----------|------|------|
| `on_chat_model_start` | Chat模型开始处理时 | serialized, messages | 监控聊天开始，记录输入消息 |

### 🧠 LLM 事件

| 事件名 | 触发时机 | 参数 | 用途 |
|--------|----------|------|------|
| `on_llm_start` | LLM开始生成时 | serialized, prompts | 监控生成开始，记录提示词 |
| `on_llm_new_token` | LLM生成新token时 | token | 实时显示生成过程，流式输出 |
| `on_llm_end` | LLM生成结束时 | response | 记录最终结果，统计token使用 |
| `on_llm_error` | LLM出错时 | error | 错误处理，日志记录 |

### 🔗 Chain 事件

| 事件名 | 触发时机 | 参数 | 用途 |
|--------|----------|------|------|
| `on_chain_start` | Chain开始执行时 | serialized, inputs | 监控链执行，记录输入 |
| `on_chain_end` | Chain执行结束时 | outputs | 记录链输出，性能统计 |
| `on_chain_error` | Chain执行出错时 | error | 链错误处理 |

### 🛠️ Tool 事件

| 事件名 | 触发时机 | 参数 | 用途 |
|--------|----------|------|------|
| `on_tool_start` | 工具开始执行时 | serialized, input_str | 监控工具调用，记录输入 |
| `on_tool_end` | 工具执行结束时 | output | 记录工具输出 |
| `on_tool_error` | 工具执行出错时 | error | 工具错误处理 |

### 🤵 Agent 事件

| 事件名 | 触发时机 | 参数 | 用途 |
|--------|----------|------|------|
| `on_agent_action` | Agent执行动作时 | action | 监控Agent决策过程 |
| `on_agent_finish` | Agent完成任务时 | finish | 记录Agent最终结果 |

### 🔍 Retriever 事件

| 事件名 | 触发时机 | 参数 | 用途 |
|--------|----------|------|------|
| `on_retriever_start` | 检索器开始检索时 | serialized, query | 监控检索过程，记录查询 |
| `on_retriever_end` | 检索器检索结束时 | documents | 记录检索结果 |
| `on_retriever_error` | 检索器出错时 | error | 检索错误处理 |

### 📝 通用事件

| 事件名 | 触发时机 | 参数 | 用途 |
|--------|----------|------|------|
| `on_text` | 处理任意文本时 | text | 文本处理监控 |
| `on_retry` | 重试操作时 | retry_state | 重试逻辑监控 |

## 🎯 测试模块详解

### 1. 回调处理器测试 (`test_callback_handlers.py`)

#### 测试目标
验证所有回调事件的触发时机、参数完整性和事件顺序。

#### 核心测试用例

**`test_chat_model_callbacks`** - Chat Model回调测试
```python
# 验证点：
✅ on_chat_model_start 事件在模型开始时触发
✅ on_llm_end 事件在模型结束时触发  
✅ 事件顺序正确（开始 → 结束）
✅ 参数包含完整的消息信息
```

**`test_streaming_callbacks`** - 流式输出回调测试
```python
# 验证点：
✅ 流式模式下触发 on_llm_new_token 事件
✅ 每个token都有对应的回调事件
✅ token内容正确传递
✅ 流式和普通模式的回调区别
```

**`test_chain_callbacks`** - Chain回调测试
```python
# 验证点：
✅ on_chain_start 在chain开始时触发
✅ on_chain_end 在chain结束时触发
✅ 复杂chain（prompt | model | parser）的完整事件序列
✅ chain名称和输入输出信息正确传递
```

**`test_tool_callbacks`** - 工具回调测试
```python
# 验证点：
✅ on_tool_start 在工具执行前触发
✅ on_tool_end 在工具执行后触发
✅ 工具名称和参数正确传递
✅ 工具结果正确记录
```

**`test_error_callbacks`** - 错误回调测试
```python
# 验证点：
✅ 错误情况下触发相应的error事件
✅ 错误信息正确传递
✅ 错误不影响回调系统稳定性
```

**`test_callback_event_timing`** - 事件时序测试
```python
# 验证点：
✅ 事件时间戳递增
✅ 开始事件总是在结束事件之前
✅ 嵌套事件的时序关系正确
```

**`test_multiple_callbacks`** - 多回调处理器测试
```python
# 验证点：
✅ 多个回调处理器同时工作
✅ 每个处理器都能接收到相同事件
✅ 处理器之间不相互干扰
```

### 2. 回调继承测试 (`test_callback_inheritance.py`)

#### 测试目标
验证运行时回调 vs 构造函数回调的区别，以及回调在复杂组件中的传播机制。

#### 核心概念对比

| 特性 | 运行时回调 | 构造函数回调 |
|------|-----------|-------------|
| **传递方式** | `invoke(data, config={"callbacks": [handler]})` | `ChatOpenAI(callbacks=[handler])` |
| **作用范围** | 🔥 **传播到所有子组件** | ⚠️ **只作用于当前对象** |
| **灵活性** | ✅ 每次调用可以不同 | ❌ 创建时固定 |
| **继承性** | ✅ 自动继承到整个执行链 | ❌ 不继承 |
| **使用场景** | 临时调试、全链监控 | 特定组件监控 |

#### 核心测试用例

**`test_runtime_vs_constructor_callbacks`** - 两种回调方式对比
```python
# 测试验证：
✅ 构造函数回调只监控定义的对象
✅ 运行时回调传播到所有子对象
✅ 两种方式都能正确捕获事件
✅ 事件内容和时机一致
```

**`test_callback_propagation_in_chains`** - Chain中的回调传播
```python
# 测试场景：prompt | model | parser
✅ 构造函数回调：只监控model的事件
✅ 运行时回调：监控prompt、model、parser的所有事件
✅ 传播范围的明显区别
```

**`test_nested_callback_inheritance`** - 嵌套组件回调继承
```python
# 测试场景：preprocessing | prompt | model | postprocessing
✅ 运行时回调传播到所有嵌套层级
✅ 每个层级的事件都被正确捕获
✅ 事件来源信息正确标记
```

**`test_mixed_callback_scenarios`** - 混合使用场景
```python
# 测试场景：同时使用两种回调
✅ 构造函数回调和运行时回调可以共存
✅ 两种回调互不干扰
✅ 运行时回调覆盖范围更广
```

**`test_callback_scope_isolation`** - 回调作用域隔离
```python
# 验证点：
✅ 不同调用的回调相互隔离
✅ 回调处理器只接收对应调用的事件
✅ 无回调的调用不影响其他调用
```

## 💡 实际应用场景

### 🔍 调试和开发场景

```python
# 开发时调试整个chain的执行过程
debug_handler = DetailedCallbackHandler()
result = complex_chain.invoke(
    input_data, 
    config={"callbacks": [debug_handler]}
)

# 查看完整的执行轨迹
for event in debug_handler.events:
    print(f"[{event['timestamp']:.3f}s] {event['event']}")
```

### 📊 生产环境监控

```python
# 只监控特定模型的性能
performance_handler = PerformanceCallbackHandler()
model = ChatOpenAI(callbacks=[performance_handler])

# 监控模型调用次数、耗时、token使用等
```

### 🎥 实时显示进度

```python
# 流式输出到用户界面
class StreamingUIHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        ui.update_text(token)  # 实时更新界面

model = ChatOpenAI(streaming=True)
result = model.invoke(messages, config={"callbacks": [ui_handler]})
```

### 📝 日志记录

```python
# 记录所有API调用
class AuditLogHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        logger.info(f"API call started: {messages[0][0].content[:50]}")
    
    def on_llm_end(self, response, **kwargs):
        logger.info(f"API call completed: {response.generations[0][0].text[:50]}")
```

## ⚙️ 配置说明

### API 配置
使用 `src/config/api.py` 中的本地配置：

```python
apis = {
    "local": {
        "base_url": "http://localhost:8212/v1",
        "api_key": "sk-nsbaxS65nDJyGfA8wp5z7pbHxKUjEQBCpN5BKg7E19nLnOgL",
    }
}
```

### 模型配置
所有测试使用 `gpt-4o-mini` 模型：

```python
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=100-200,  # 测试用较小值
    timeout=30,
    max_retries=3
)
```

## 🛠️ 自定义回调处理器

### 基础模板

```python
from langchain_core.callbacks import BaseCallbackHandler

class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.events = []
    
    def on_chat_model_start(self, serialized, messages, **kwargs):
        # 处理聊天模型开始事件
        self.events.append({
            'type': 'chat_start',
            'timestamp': time.time(),
            'message_count': len(messages[0]) if messages else 0
        })
    
    def on_llm_end(self, response, **kwargs):
        # 处理LLM结束事件
        self.events.append({
            'type': 'llm_end',
            'timestamp': time.time(),
            'token_count': len(response.generations[0][0].text.split())
        })
```

### 高级示例

```python
class ComprehensiveCallbackHandler(BaseCallbackHandler):
    """全面的回调处理器示例"""
    
    def __init__(self, log_file: str = None):
        self.events = []
        self.start_time = time.time()
        self.log_file = log_file
        
    def _log(self, event_name: str, **kwargs):
        event = {
            'event': event_name,
            'timestamp': time.time() - self.start_time,
            'datetime': datetime.now().isoformat(),
            **kwargs
        }
        self.events.append(event)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{json.dumps(event)}\n")
    
    # 实现所有需要的回调方法...
```

## 🚨 常见问题和解决方案

### Q1: 回调没有被触发
**可能原因：**
- 使用了构造函数回调但期望传播到子组件
- 回调处理器没有实现对应的方法

**解决方案：**
```python
# ❌ 错误：构造函数回调不会传播
model = ChatOpenAI(callbacks=[handler])
chain = prompt | model | parser
result = chain.invoke(data)  # handler只监控model，不监控prompt和parser

# ✅ 正确：使用运行时回调
result = chain.invoke(data, config={"callbacks": [handler]})  # 监控所有组件
```

### Q2: 回调事件顺序不对
**可能原因：**
- 异步执行导致的时序问题
- 嵌套调用的复杂性

**解决方案：**
```python
# 检查时间戳确认真实顺序
for event in handler.events:
    print(f"[{event['timestamp']:.3f}s] {event['event']}")
```

### Q3: 流式输出没有token事件
**可能原因：**
- 模型没有启用streaming
- 回调处理器没有实现`on_llm_new_token`

**解决方案：**
```python
# 确保启用流式输出
model = ChatOpenAI(streaming=True)

# 确保实现token回调
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(f"Token: {token}")
```

## 🏗️ 项目结构

```
unitests/test_callbacks/
├── README.md                      # 📚 项目文档（本文件）
├── __init__.py                    # 📦 包初始化和配置
├── run_all_tests.py              # 🚀 主测试运行器
│
├── test_callback_handlers.py     # 🔔 回调处理器测试
│   ├── DetailedCallbackHandler   # 详细事件记录
│   ├── TestCallbackHandlers      # 测试类
│   ├── Chat Model事件测试
│   ├── LLM事件测试
│   ├── Chain事件测试
│   ├── Tool事件测试
│   ├── 错误回调测试
│   ├── 事件时序测试
│   └── 多回调处理器测试
│
└── test_callback_inheritance.py  # 🔗 回调继承测试
    ├── TrackingCallbackHandler    # 追踪回调处理器
    ├── TestCallbackInheritance    # 测试类
    ├── 运行时vs构造函数回调对比
    ├── Chain中的回调传播测试
    ├── 嵌套组件回调继承测试
    ├── 混合回调场景测试
    └── 回调作用域隔离测试
```

## 📚 参考资源

### LangChain 官方文档
- [Callbacks概念文档](https://python.langchain.com/docs/concepts/callbacks)
- [回调处理指南](https://python.langchain.com/docs/how_to/callbacks_at_runtime)
- [自定义回调处理器](https://python.langchain.com/docs/how_to/custom_callback_handlers)

### 相关测试参考
- `unitests/test_chatmodels/test_streaming.py` - 流式输出回调示例
- `unitests/test_lcel/test_chatopenai_applications.py` - Token追踪回调示例

### Python 回调模式
- [Python回调函数模式](https://docs.python.org/3/library/unittest.html#unittest.TestCase)
- [观察者模式实现](https://refactoring.guru/design-patterns/observer/python/example)

---

**测试口号**: *每个回调都有它的时机，每个事件都有它的故事* 🔔✨ 