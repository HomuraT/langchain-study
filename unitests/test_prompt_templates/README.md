# 提示模板测试套件

这是一个全面的LangChain提示模板测试套件，专注于测试**PromptTemplate**、**ChatPromptTemplate**、**MessagesPlaceholder**、**Jinja2PromptTemplate**和**ExampleSelectors**的各种功能和应用场景。本测试套件深入验证提示模板在实际AI应用中的可靠性和正确性。

## 🎯 测试成果概览

### ✅ 测试通过率
- **总测试方法数**: 28个核心测试方法
- **覆盖模块**: 3个主要测试文件
- **测试场景**: 涵盖基础功能到AI集成的完整链路
- **代码覆盖**: 包含正常流程、异常处理和边界条件

### 📊 测试模块分布
```
提示模板测试套件
├── test_prompt_templates.py (12个测试方法)
│   ├── PromptTemplate基础测试 (3个)
│   ├── ChatPromptTemplate测试 (3个) 
│   ├── MessagesPlaceholder测试 (3个)
│   └── AI模型集成测试 (4个)
├── test_jinja2_templates.py (9个测试方法)
│   ├── Jinja2基础功能 (2个)
│   ├── 高级语法特性 (4个)
│   ├── 复杂模板应用 (2个)
│   └── AI集成应用 (2个)
└── test_example_selectors.py (11个测试方法)
    ├── 自定义选择器 (3个)
    ├── 内置选择器 (4个)
    ├── 语义选择器 (2个)
    ├── 集成应用 (1个)
    └── 错误处理 (1个)
```

### 🔍 核心功能测试覆盖

#### 📝 PromptTemplate测试
- **基础创建**: `from_template()` vs 构造函数创建方式
- **格式化功能**: `format()` 和 `invoke()` 方法验证
- **多变量处理**: 复杂模板的变量识别和替换
- **AI集成**: 与ChatOpenAI的完整调用链测试

#### 💬 ChatPromptTemplate测试  
- **多角色消息**: System、Human、AI消息类型
- **消息组合**: `from_messages()` 方法和模板对象
- **复杂对话**: 多轮对话模板构建
- **AI集成**: 对话式AI交互测试

#### 📋 MessagesPlaceholder测试
- **历史消息插入**: 动态消息列表管理
- **替代语法**: 不同创建方式的兼容性
- **边界处理**: 空消息列表的正确处理
- **AI集成**: 上下文感知的对话测试

#### 🎨 Jinja2PromptTemplate测试
- **基础语法**: 变量替换和模板识别
- **条件逻辑**: if-else分支控制
- **循环功能**: for循环和loop变量
- **过滤器系统**: title、lower、default、round等内置过滤器
- **宏功能**: 可重用模板片段定义
- **复杂应用**: 多语法结合的实际场景
- **代码生成**: 基于模板的代码自动生成

#### 🔧 ExampleSelectors测试
- **自定义选择器**: BaseExampleSelector接口实现
- **长度基础选择器**: 基于token长度的动态选择
- **语义相似度选择器**: 使用嵌入模型的智能选择
- **MMR选择器**: 平衡相关性和多样性的选择策略
- **Few-Shot集成**: 在提示模板中的实际应用
- **错误处理**: 异常情况和边界条件的处理

## 🚀 快速开始

### 🔧 环境准备

```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 确保已安装必要依赖
# 核心依赖
pip install langchain-core langchain-openai langchain-community

# 可选依赖（用于示例选择器）
pip install faiss-cpu numpy

# 模板引擎
pip install jinja2
```

### ⚙️ API配置

在 `src/config/api.py` 中配置API设置：

```python
apis = {
    "local": {
        "base_url": "http://localhost:11434/v1",  # Ollama本地服务
        "api_key": "ollama",
        "model": "qwen2.5:latest"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "your-openai-api-key",
        "model": "gpt-4o-mini"
    }
}
```

### 🏃‍♂️ 快速运行

```bash
# 一键运行所有测试（推荐）
python unitests/test_prompt_templates/run_all_tests.py

# 查看详细输出
python unitests/test_prompt_templates/run_all_tests.py --verbose

# 只运行特定模块测试
python unitests/test_prompt_templates/run_all_tests.py --tests prompt_templates
python unitests/test_prompt_templates/run_all_tests.py --tests jinja2_templates  
python unitests/test_prompt_templates/run_all_tests.py --tests example_selectors

# 静默模式（只显示摘要）
python unitests/test_prompt_templates/run_all_tests.py --quiet

# 列出所有可用测试
python unitests/test_prompt_templates/run_all_tests.py --list
```

### 🧪 使用unittest直接运行

```bash
# 运行所有测试
python -m unittest discover unitests/test_prompt_templates -v

# 运行单个测试文件
python -m unittest unitests.test_prompt_templates.test_prompt_templates -v
python -m unittest unitests.test_prompt_templates.test_jinja2_templates -v  
python -m unittest unitests.test_prompt_templates.test_example_selectors -v

# 运行特定测试方法
python -m unittest unitests.test_prompt_templates.test_prompt_templates.TestPromptTemplates.test_prompt_template_creation -v
```

### 🎯 使用pytest运行（高级）

```bash
# 并行运行测试（提升速度）
python -m pytest unitests/test_prompt_templates/ -n auto -v

# 生成HTML测试报告
python -m pytest unitests/test_prompt_templates/ --html=test_report.html

# 只运行失败的测试
python -m pytest unitests/test_prompt_templates/ --lf

# 运行特定标记的测试
python -m pytest unitests/test_prompt_templates/ -m "not slow" -v
```

## 📋 核心功能详解

### 🔧 PromptTemplate - 字符串模板核心

#### 实际应用场景
```python
from langchain_core.prompts import PromptTemplate

# 1. 动态内容生成
email_template = PromptTemplate.from_template(
    "写一封{tone}的邮件给{recipient}，主题是{subject}。内容包括：{content}"
)

# 2. 多语言支持
i18n_template = PromptTemplate.from_template(
    "请用{language}回答以下问题：{question}"
)

# 3. 任务指令构建
task_template = PromptTemplate.from_template(
    "作为{role}，请帮助{target_user}完成{task_type}任务：{specific_task}"
)
```

#### 测试覆盖要点
- ✅ **创建方式验证**: `from_template()` vs 构造函数
- ✅ **变量识别**: 自动识别模板中的变量占位符
- ✅ **格式化方法**: `format()` 和 `invoke()` 的功能对等性
- ✅ **多变量处理**: 复杂模板的完整变量替换

### 💬 ChatPromptTemplate - 对话模板系统

#### 实际应用场景
```python
from langchain_core.prompts import ChatPromptTemplate

# 1. 智能客服对话
customer_service = ChatPromptTemplate.from_messages([
    ("system", "你是专业的{company}客服代表，始终保持{tone}和耐心。"),
    ("user", "用户问题：{user_question}"),
    ("assistant", "我理解您的问题。让我为您查找相关信息..."),
    ("user", "补充信息：{additional_info}")
])

# 2. 教学助手对话
tutor_template = ChatPromptTemplate.from_messages([
    ("system", "你是{subject}老师，学生水平：{level}，教学风格：{style}"),
    ("user", "我想学习：{topic}"),
    ("user", "{follow_up_question}")
])
```

#### 测试覆盖要点
- ✅ **消息类型**: System、Human、AI消息的正确创建
- ✅ **模板组合**: 使用`from_messages()`和模板对象
- ✅ **变量管理**: 跨多个消息的变量识别和替换
- ✅ **复杂对话**: 多轮对话的完整模板构建

### 📋 MessagesPlaceholder - 动态消息管理

#### 实际应用场景
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 上下文感知对话
context_aware_chat = ChatPromptTemplate([
    ("system", "你是有用的助手，能基于对话历史提供连贯回答"),
    MessagesPlaceholder("conversation_history"),
    ("user", "{new_question}")
])

# 历史消息动态插入
history = [
    HumanMessage(content="我想学习Python"),
    AIMessage(content="很好！从哪个方面开始？"),
    HumanMessage(content="数据类型")
]
```

#### 测试覆盖要点
- ✅ **历史插入**: 动态消息列表的正确插入
- ✅ **空列表处理**: 无历史消息时的正确行为
- ✅ **消息类型保持**: 插入消息的类型完整性
- ✅ **语法兼容性**: 不同创建方式的功能一致性

### 🎨 Jinja2PromptTemplate - 高级模板引擎

#### 核心功能特性
```python
from langchain_core.prompts import PromptTemplate

# 1. 条件逻辑
conditional_template = PromptTemplate.from_template(
    """
{%- if user.is_vip -%}
尊敬的VIP用户 {{ user.name }}，您享有专属服务！
{%- else -%}
您好 {{ user.name }}，欢迎使用我们的服务！
{%- endif -%}
    """,
    template_format="jinja2"
)

# 2. 循环处理
loop_template = PromptTemplate.from_template(
    """
任务清单：
{%- for task in tasks %}
{{ loop.index }}. {{ task.name }} - 优先级：{{ task.priority }}
{%- endfor %}
    """,
    template_format="jinja2"
)

# 3. 过滤器应用
filter_template = PromptTemplate.from_template(
    """
- 姓名：{{ name | title }}
- 邮箱：{{ email | lower }}
- 分数：{{ score | round(2) }}
- 标签：{{ tags | join(", ") }}
    """,
    template_format="jinja2"
)
```

#### 测试覆盖要点
- ✅ **基础语法**: 变量替换和模板格式识别
- ✅ **条件逻辑**: if-else分支的正确执行
- ✅ **循环功能**: for循环和loop变量的使用
- ✅ **过滤器系统**: 内置过滤器的功能验证
- ✅ **宏定义**: 可重用模板片段的创建和调用
- ✅ **复杂场景**: 多种语法结合的实际应用
- ✅ **代码生成**: 基于模板的代码自动生成

### 🔧 ExampleSelectors - 智能示例选择

#### 支持的选择器类型

##### 1. 自定义示例选择器 (CustomExampleSelector)
```python
from langchain_core.example_selectors.base import BaseExampleSelector

class CustomExampleSelector(BaseExampleSelector):
    """基于输入长度选择最相似长度的示例"""
    
    def select_examples(self, input_variables):
        # 根据自定义逻辑选择示例
        return selected_examples
```

##### 2. 长度基础选择器 (LengthBasedExampleSelector)
```python
from langchain_core.example_selectors import LengthBasedExampleSelector

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100  # token限制
)
```

##### 3. 语义相似度选择器 (SemanticSimilarityExampleSelector)
```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vectorstore_cls=FAISS,
    k=2  # 选择最相似的2个示例
)
```

##### 4. 最大边际相关性选择器 (MaxMarginalRelevanceExampleSelector)
```python
mmr_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vectorstore_cls=FAISS,
    k=2  # 平衡相关性和多样性
)
```

#### 测试覆盖要点
- ✅ **选择器创建**: 各种类型选择器的正确初始化
- ✅ **示例选择**: 选择逻辑的准确性验证
- ✅ **动态添加**: 运行时添加新示例的功能
- ✅ **Few-Shot集成**: 在提示模板中的实际应用
- ✅ **多选择器对比**: 不同策略的选择结果对比
- ✅ **嵌入服务集成**: 语义选择器的API调用测试
- ✅ **错误处理**: 异常情况和边界条件的处理

## 🛠️ 配置与定制

### ⚙️ API配置详解

```python
# src/config/api.py
apis = {
    "local": {
        "base_url": "http://localhost:11434/v1",  # Ollama本地服务
        "api_key": "ollama",
        "model": "qwen2.5:latest"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "your-openai-api-key", 
        "model": "gpt-4o-mini"
    }
}
```

### 🔧 自定义测试扩展

```python
# 添加自定义测试用例
import unittest
from langchain_core.prompts import PromptTemplate

class CustomPromptTest(unittest.TestCase):
    
    def test_business_scenario(self):
        """测试特定业务场景"""
        template = PromptTemplate.from_template(
            "为{company}的{product}设计{marketing_type}营销方案"
        )
        
        result = template.format(
            company="科技公司",
            product="AI助手",
            marketing_type="社交媒体"
        )
        
        self.assertIn("科技公司", result)
        self.assertIn("AI助手", result)
        self.assertIn("社交媒体", result)
```

## 🎯 最佳实践指南

### ✅ 推荐做法

#### 1. 模板设计原则
```python
# ✅ 好的做法：变量命名清晰语义化
PromptTemplate.from_template("分析{user_input}的{analysis_type}，生成{output_format}报告")

# ❌ 避免：变量名模糊不清
PromptTemplate.from_template("分析{x}的{y}，生成{z}")
```

#### 2. Jinja2模板优化
```jinja2
{# ✅ 好的做法：使用白空间控制 #}
{%- for item in items -%}
{{ item.name }}{% if not loop.last %}, {% endif %}
{%- endfor %}

{# ❌ 避免：不控制输出格式 #}
{% for item in items %}
{{ item.name }}
{% endfor %}
```

#### 3. 错误处理策略
```python
# ✅ 好的做法：完整的异常处理
try:
    result = template.format(**user_data)
except KeyError as e:
    print(f"缺少必需变量: {e}")
    # 提供默认值或提示用户
except Exception as e:
    print(f"模板处理错误: {e}")
```

### ⚠️ 注意事项

#### 安全性考虑
- **输入验证**: 避免直接使用未验证的用户输入作为模板内容
- **注入防护**: 对Jinja2模板进行安全检查，防止模板注入攻击
- **敏感信息**: 不在模板中硬编码API密钥或敏感数据

#### 性能优化
- **模板缓存**: 对于复杂的Jinja2模板，考虑预编译缓存
- **批量处理**: 使用批量API调用减少网络延迟
- **示例选择**: 合理控制示例数量，避免token限制

#### 调试技巧
```python
# 查看模板结构
template = ChatPromptTemplate.from_messages([...])
print(template.pretty_print())

# 验证变量识别
print(f"输入变量: {template.input_variables}")

# 测试格式化结果
test_data = {"var1": "value1", "var2": "value2"}
messages = template.format_messages(**test_data)
for msg in messages:
    print(f"{type(msg).__name__}: {msg.content}")
```

## 🐛 故障排除

### 常见问题及解决方案

#### ❓ Jinja2模板语法错误
**问题**: `TemplateSyntaxError: unexpected char '{'`

**解决方案**: 
```python
# ❌ 错误：未指定template_format
template = PromptTemplate.from_template("{{ name }}")

# ✅ 正确：明确指定jinja2格式
template = PromptTemplate.from_template("{{ name }}", template_format="jinja2")
```

#### ❓ API连接失败
**问题**: 连接超时或认证失败

**解决方案**:
1. 检查API配置文件 `src/config/api.py`
2. 验证网络连接和API密钥有效性
3. 确认模型服务正在运行

```python
# 测试API连接
from src.config.api import apis
config = apis["local"]
print(f"连接地址: {config['base_url']}")

# 验证Ollama服务
curl http://localhost:11434/api/tags
```

#### ❓ 示例选择器嵌入服务问题
**问题**: 语义相似度选择器创建失败

**解决方案**:
1. 确认嵌入服务可用性
2. 检查API配置和网络连接
3. 验证依赖包安装（faiss-cpu, numpy）

```bash
# 安装缺失依赖
pip install faiss-cpu numpy langchain-community

# 测试嵌入服务
python -c "from langchain_openai import OpenAIEmbeddings; print('嵌入服务可用')"
```

#### ❓ 内存使用过高
**问题**: 大规模测试时内存不足

**解决方案**:
```bash
# 单独运行测试模块
python -m unittest unitests.test_prompt_templates.test_jinja2_templates -v

# 限制并发数（使用pytest）
python -m pytest unitests/test_prompt_templates/ -n 2

# 清理测试数据
def tearDown(self):
    self.large_template = None
    self.test_data = None
```


## 📚 文档资源

- **LangChain官方文档**: [https://python.langchain.com/docs/](https://python.langchain.com/docs/)
- **Jinja2文档**: [https://jinja.palletsprojects.com/](https://jinja.palletsprojects.com/)
- **FAISS文档**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)