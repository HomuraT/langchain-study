# LangChain

本指南全面介绍LangChain中结构化输出生成和数据解析的核心功能，包括Pydantic BaseModel构造、各种输出解析器的使用，以及高级错误处理机制。

详细测试样例和代码可参考如下两个链接：
- [test_output_parsers](https://github.com/HomuraT/langchain-study/tree/main/unitests/test_output_parsers)
- [test_pydantic_base_model](https://github.com/HomuraT/langchain-study/tree/main/unitests/test_pydantic_base_model)

## 📋 目录

1. [结构化输出概述](#结构化输出概述)
2. [Pydantic BaseModel 集成](#pydantic-basemodel-集成)
3. [输出解析器系统](#输出解析器系统)
4. [智能错误处理](#智能错误处理)
5. [高级应用模式](#高级应用模式)
6. [最佳实践指南](#最佳实践指南)

## 结构化输出概述

### 概念定义

**结构化输出**是指将大语言模型（LLM）的自然语言输出转换为具有明确数据类型和格式的结构化数据。这种转换使得AI应用能够可靠地处理和使用LLM的输出结果。

### 核心优势

- **🎯 类型安全**：确保数据符合预定义的结构和类型约束
- **🔄 一致性**：保证输出格式的稳定性和可预测性  
- **⚡ 可靠性**：通过验证和错误处理提高系统鲁棒性
- **🔧 易集成**：直接获得可在应用中使用的数据对象

### 技术实现方式

LangChain提供两种主要的结构化输出实现方式：

1. **结构化输出模式（Structured Output）**：基于约束解码技术，确保100%格式正确
2. **函数调用模式（Function Calling）**：基于工具调用机制，支持更复杂的数据结构

**相关链接**：
- [结构化输出概念](https://python.langchain.com/docs/how_to/structured_output/)
- [输出解析器概念](https://python.langchain.com/docs/concepts/output_parsers/)

## Pydantic BaseModel 集成

### 概念定义

**Pydantic BaseModel**是Python中最强大的数据验证和序列化库，LangChain深度集成了Pydantic来定义和验证结构化输出的数据模型。

### with_structured_output 方法

**输入**：Pydantic模型类或JSON Schema
**输出**：符合指定结构的数据对象
**原理**：将LLM输出约束到预定义的数据结构中

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class UserProfile(BaseModel):
    """用户档案数据模型"""
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄", ge=0, le=150)
    email: str = Field(description="邮箱地址")
    skills: List[str] = Field(description="技能列表")

# 创建结构化输出模型
model = ChatOpenAI(model="gpt-4o-mini")
structured_llm = model.with_structured_output(UserProfile)

# 使用结构化输出
result = structured_llm.invoke("提取用户信息：张三，25岁，邮箱zhang@example.com，擅长Python和数据分析")
# result 是 UserProfile 类型的对象
print(f"姓名: {result.name}, 年龄: {result.age}")
```

### 复杂数据结构支持

**嵌套模型**：支持模型之间的嵌套关系

```python
class Address(BaseModel):
    """地址信息"""
    street: str = Field(description="街道地址")
    city: str = Field(description="城市")
    country: str = Field(description="国家")

class Company(BaseModel):
    """公司信息"""
    name: str = Field(description="公司名称")
    address: Address = Field(description="公司地址")
    employees: List[UserProfile] = Field(description="员工列表")

# 处理复杂嵌套结构
company_llm = model.with_structured_output(Company)
```

### 枚举类型支持

```python
from enum import Enum

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Task(BaseModel):
    """任务信息"""
    title: str = Field(description="任务标题")
    status: TaskStatus = Field(description="任务状态")
    priority: int = Field(description="优先级", ge=1, le=5)

task_llm = model.with_structured_output(Task)
```

### Function Calling vs Structured Output

**技术选择指南**：

| 特性 | Structured Output | Function Calling |
|------|------------------|------------------|
| **动态字段支持** | ❌ 不支持 `Dict[str, Any]` | ✅ 完全支持 |
| **复杂嵌套** | 受限 | 无限制 |
| **格式可靠性** | 100%正确 | 高度可靠 |
| **首次延迟** | 有预处理开销 | 无额外开销 |

```python
# 包含动态字段的模型需要使用 function_calling
class FlexibleResponse(BaseModel):
    core_data: str = Field(description="核心数据")
    metadata: Dict[str, Any] = Field(description="动态元数据")

# 必须指定 method="function_calling"
flexible_llm = model.with_structured_output(
    FlexibleResponse,
    method="function_calling"
)
```

**相关链接**：
- [如何返回结构化数据](https://python.langchain.com/docs/how_to/structured_output/)
- [Pydantic模型验证](https://python.langchain.com/docs/how_to/output_parser_pydantic/)

## 输出解析器系统

### 概念定义

**输出解析器（Output Parsers）**是LangChain中负责将LLM的文本输出转换为特定数据类型的组件。它们提供了比结构化输出更灵活的数据处理能力。

### 基础解析器类型

#### StrOutputParser

**功能**：提取AI消息中的纯文本内容
**输入**：AIMessage对象
**输出**：字符串
**适用场景**：简单的文本提取和处理

```python
from langchain_core.output_parsers import StrOutputParser

# 文本提取链
text_chain = model | StrOutputParser()
result = text_chain.invoke("介绍一下人工智能")
# result 是纯字符串
```

#### JsonOutputParser

**功能**：解析JSON格式的模型输出
**输入**：包含JSON的文本
**输出**：Python字典或列表
**适用场景**：结构化数据提取

```python
from langchain_core.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()

# 创建JSON解析链
json_chain = model | json_parser
result = json_chain.invoke("用JSON格式返回用户信息：姓名张三，年龄25")
# result 是 Python 字典
```

#### XMLOutputParser

**功能**：解析XML格式的结构化数据
**输入**：XML格式文本
**输出**：解析后的数据结构
**适用场景**：处理层次化数据

```python
from langchain_core.output_parsers import XMLOutputParser

xml_parser = XMLOutputParser()
xml_chain = model | xml_parser
```

#### YAMLOutputParser

**功能**：解析YAML格式的配置数据
**输入**：YAML格式文本  
**输出**：Python数据结构
**适用场景**：配置文件处理

```python
from langchain_core.output_parsers import YamlOutputParser

yaml_parser = YamlOutputParser()
yaml_chain = model | yaml_parser
```

### PydanticOutputParser

**功能**：将文本输出解析为Pydantic模型实例
**输入**：符合模型格式的文本
**输出**：Pydantic模型对象
**原理**：结合格式指令和数据验证

```python
from langchain_core.output_parsers import PydanticOutputParser

# 创建解析器
parser = PydanticOutputParser(pydantic_object=UserProfile)

# 获取格式指令
format_instructions = parser.get_format_instructions()

# 构建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "提取用户信息：{query}\n{format_instructions}"
).partial(format_instructions=format_instructions)

# 创建完整链
chain = prompt | model | parser
result = chain.invoke({"query": "张三，工程师，28岁"})
# result 是 UserProfile 对象
```

### 流式解析支持

**概念**：支持实时处理模型的流式输出
**优势**：提供更好的用户体验和响应性

```python
# 流式JSON解析
from langchain_core.output_parsers import SimpleJsonOutputParser

streaming_parser = SimpleJsonOutputParser()

# 流式处理
for chunk in (model | streaming_parser).stream("生成用户数据"):
    print(chunk, end="", flush=True)
```

**相关链接**：
- [如何解析字符串输出](https://python.langchain.com/docs/how_to/output_parser_string/)
- [如何解析JSON输出](https://python.langchain.com/docs/how_to/output_parser_json/)
- [如何解析XML输出](https://python.langchain.com/docs/how_to/output_parser_xml/)
- [如何解析YAML输出](https://python.langchain.com/docs/how_to/output_parser_yaml/)

## 智能错误处理

### 概念定义

**智能错误处理**是LangChain提供的高级功能，能够自动检测、修复和重试解析错误，大幅提升系统的鲁棒性和可靠性。

### OutputFixingParser

**功能**：自动修复格式错误的LLM输出
**输入**：可能包含格式错误的文本
**输出**：修复后的正确数据结构
**原理**：使用另一个LLM理解并修复格式问题

```python
from langchain_core.output_parsers import OutputFixingParser

# 基础解析器
base_parser = JsonOutputParser()

# 包装为自动修复解析器
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(temperature=0.1)  # 使用低温度提高修复准确性
)

# 自动处理格式错误
broken_json = '{"name": "张三", "age": 25'  # 缺少闭合括号
fixed_result = fixing_parser.parse(broken_json)  # 自动修复成功
```

**应用场景**：
- JSON格式错误（缺少括号、多余逗号）
- XML标签不匹配
- YAML缩进问题
- Pydantic字段类型错误

### RetryWithErrorOutputParser

**功能**：解析失败时的智能重试机制
**输入**：原始提示和错误信息
**输出**：重新生成的正确格式数据
**原理**：将错误反馈给LLM，指导重新生成

```python
from langchain_core.output_parsers import RetryWithErrorOutputParser

# 创建重试解析器
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=UserProfile),
    llm=ChatOpenAI(temperature=0.1),
    max_retries=3  # 最多重试3次
)

# 自动处理解析失败并重试
chain = prompt | model | retry_parser
result = chain.invoke({"query": "用户信息提取"})
```

**重试流程**：
1. 初始解析尝试失败
2. 将错误信息反馈给LLM
3. LLM重新生成符合格式的输出
4. 重复直到成功或达到最大重试次数

### 组合错误处理策略

**组合使用**：将多种错误处理机制结合，实现最高可靠性

```python
# 构建三层错误处理
base_parser = PydanticOutputParser(pydantic_object=ComplexModel)

# 第一层：格式修复
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(temperature=0.1)
)

# 第二层：重试机制
ultimate_parser = RetryWithErrorOutputParser.from_llm(
    parser=fixing_parser,
    llm=ChatOpenAI(temperature=0.1),
    max_retries=2
)

ultimate_chain = prompt | model | ultimate_parser
```


**相关链接**：
- [如何处理解析重试](https://python.langchain.com/docs/how_to/output_parser_retry/)
- [如何修复解析错误](https://python.langchain.com/docs/how_to/output_parser_fixing/)

## 高级应用模式

### 自定义解析器开发

**概念**：基于BaseOutputParser创建专用的解析逻辑
**原理**：继承基类并实现parse和get_format_instructions方法

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class ListOutputParser(BaseOutputParser[List[str]]):
    """列表解析器 - 分隔符解析"""
    
    def __init__(self, separator: str = ","):
        self.separator = separator
    
    def parse(self, text: str) -> List[str]:
        """解析分隔符分隔的列表"""
        return [item.strip() for item in text.split(self.separator)]
    
    def get_format_instructions(self) -> str:
        """返回格式指令"""
        return f"请用{self.separator}分隔列表项目"

# 使用自定义解析器
list_parser = ListOutputParser(separator="|")
list_chain = model | list_parser
```

### 条件解析器

**功能**：根据内容特征自动选择解析策略

```python
class ConditionalOutputParser(BaseOutputParser):
    """条件解析器 - 智能格式识别"""
    
    def parse(self, text: str) -> Union[str, Dict, List]:
        """根据内容特征选择解析策略"""
        text = text.strip()
        
        if text.startswith('{') and text.endswith('}'):
            # JSON对象格式
            return json.loads(text)
        elif text.startswith('[') and text.endswith(']'):
            # JSON数组格式  
            return json.loads(text)
        elif ',' in text:
            # 逗号分隔的列表
            return [item.strip() for item in text.split(',')]
        else:
            # 纯文本
            return text
    
    def get_format_instructions(self) -> str:
        return "可以返回JSON、列表或纯文本格式"
```

### 链式解析器

**概念**：组合多个解析器，实现复杂的数据处理流程

```python
class ChainedOutputParser(BaseOutputParser[Dict[str, Any]]):
    """链式解析器 - 多策略组合"""
    
    def __init__(self, parsers: List[Tuple[str, BaseOutputParser]]):
        self.parsers = parsers
    
    def parse(self, text: str) -> Dict[str, Any]:
        """依次应用多个解析器"""
        results = {"original": text}
        
        for name, parser in self.parsers:
            try:
                results[name] = parser.parse(text)
            except Exception as e:
                results[f"{name}_error"] = str(e)
        
        return results

# 创建链式解析器
chained_parser = ChainedOutputParser([
    ("json", JsonOutputParser()),
    ("list", ListOutputParser()),
    ("yaml", YamlOutputParser())
])
```

### 验证解析器

**功能**：在解析后进行额外的数据验证和清理

```python
class ValidatingOutputParser(BaseOutputParser[Dict]):
    """验证解析器 - 数据质量保证"""
    
    def __init__(self, base_parser: BaseOutputParser, validators: List[callable]):
        self.base_parser = base_parser
        self.validators = validators
    
    def parse(self, text: str) -> Dict:
        """解析并验证数据"""
        result = self.base_parser.parse(text)
        
        # 应用验证规则
        for validator in self.validators:
            result = validator(result)
        
        return result

def validate_user_age(data: Dict) -> Dict:
    """验证用户年龄合理性"""
    if 'age' in data and not (0 <= data['age'] <= 150):
        data['age'] = max(0, min(data['age'], 150))
    return data

# 使用验证解析器
validating_parser = ValidatingOutputParser(
    base_parser=JsonOutputParser(),
    validators=[validate_user_age]
)
```

**相关链接**：
- [如何创建自定义解析器](https://python.langchain.com/docs/how_to/output_parser_custom/)

## 最佳实践指南

### 选择合适的方法

**决策流程图**：

```
需要结构化输出？
├─ 是 → 数据结构复杂（包含动态字段）？
│   ├─ 是 → 使用 with_structured_output(method="function_calling")
│   └─ 否 → 使用 with_structured_output() (默认模式)
└─ 否 → 需要格式转换？
    ├─ 是 → 使用合适的OutputParser (Json/XML/YAML)
    └─ 否 → 使用 StrOutputParser
```

### 错误处理最佳实践

**分层错误处理**：
1. **预防**：使用清晰的格式指令
2. **检测**：实现数据验证逻辑
3. **修复**：使用OutputFixingParser
4. **重试**：使用RetryWithErrorOutputParser
5. **降级**：提供备用解析策略

```python
def robust_structured_output(query: str, model_class: BaseModel, fallback_parser=None):
    """鲁棒的结构化输出处理"""
    try:
        # 主要方法：结构化输出
        structured_llm = model.with_structured_output(model_class)
        return structured_llm.invoke(query)
    except Exception as e:
        if fallback_parser:
            # 降级方法：使用解析器
            return fallback_parser.parse(model.invoke(query).content)
        else:
            raise e
```

### 监控和调试

**性能监控**：
```python
import time
from typing import Any

def monitored_parse(parser: BaseOutputParser, text: str) -> Dict[str, Any]:
    """监控解析性能"""
    start_time = time.time()
    try:
        result = parser.parse(text)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    end_time = time.time()
    
    return {
        "result": result,
        "success": success,
        "error": error,
        "parse_time": end_time - start_time,
        "input_length": len(text)
    }
```

### 提示工程优化

**格式指令优化**：
```python
def enhance_format_instructions(base_instructions: str, examples: List[str] = None) -> str:
    """增强格式指令"""
    enhanced = base_instructions
    
    if examples:
        enhanced += "\n\n示例格式:\n"
        for i, example in enumerate(examples, 1):
            enhanced += f"{i}. {example}\n"
    
    enhanced += "\n注意：严格按照上述格式返回，确保数据完整性。"
    return enhanced
```

### 类型安全增强

**泛型解析器**：
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class TypedOutputParser(BaseOutputParser[T], Generic[T]):
    """类型安全的解析器基类"""
    
    def __init__(self, target_type: Type[T]):
        self.target_type = target_type
    
    def parse(self, text: str) -> T:
        # 实现类型安全的解析逻辑
        pass
```

## 相关资源链接

### LangChain 官方文档
- [输出解析器概述](https://python.langchain.com/docs/concepts/output_parsers/)
- [如何解析结构化输出](https://python.langchain.com/docs/how_to/output_parser_structured/)
- [如何返回结构化数据](https://python.langchain.com/docs/how_to/structured_output/)
- [如何创建自定义解析器](https://python.langchain.com/docs/how_to/output_parser_custom/)
- [如何处理解析重试](https://python.langchain.com/docs/how_to/output_parser_retry/)
- [如何修复解析错误](https://python.langchain.com/docs/how_to/output_parser_fixing/)

### 数据验证和模型
- [Pydantic 官方文档](https://docs.pydantic.dev/)
- [Python Type Hints 指南](https://docs.python.org/3/library/typing.html)

### 高级功能
- [异步编程概念](https://python.langchain.com/docs/concepts/async/)
- [流式输出指南](https://python.langchain.com/docs/how_to/streaming/)
- [批处理操作](https://python.langchain.com/docs/how_to/batch/) 