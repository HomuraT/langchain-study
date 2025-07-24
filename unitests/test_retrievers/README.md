# LangChain 检索器功能详解与测试

本测试套件深入测试LangChain检索器的核心功能，涵盖向量存储检索、多查询检索、上下文压缩等关键技术。本文档详细介绍这些LangChain检索器组件的功能、用法和最佳实践。

## 目录结构

```
test_retrievers/
├── README.md                       # 本文件，LangChain检索器功能详解
├── run_tests.py                   # 测试运行脚本
├── test_basic_retrievers.py       # 基础检索器功能测试
├── test_multiquery_retrievers.py  # 多查询检索器测试
├── test_compression_retrievers.py # 上下文压缩检索器测试
└── test_search_strategies.py      # 搜索策略和参数测试
```

## LangChain 检索器核心组件功能详解

### 1. 基础向量存储检索器 (`VectorStoreRetriever`)

向量存储检索器是最基础的检索器类型，它使用向量存储来检索文档。

#### 1.1 核心方法

**`as_retriever(search_type="similarity", search_kwargs={})`**
- **功能**: 从向量存储创建检索器
- **输入**: 搜索类型和搜索参数
- **输出**: VectorStoreRetriever实例
- **应用场景**: 
  - 基础文档检索：根据相似性查找相关文档
  - 知识库问答：检索相关知识进行问答
  - 文档推荐：基于内容相似性推荐文档

**主要搜索类型**:
- `similarity`: 相似性搜索（默认）
- `mmr`: 最大边际相关性搜索，减少结果冗余
- `similarity_score_threshold`: 基于相似性分数阈值的搜索

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

# 创建基础检索器
retriever = vectorstore.as_retriever()

# 创建MMR检索器
mmr_retriever = vectorstore.as_retriever(search_type="mmr")

# 创建阈值检索器
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)
```

#### 1.2 搜索参数配置

**`search_kwargs` 常用参数**:
- `k`: 返回文档数量（默认4）
- `score_threshold`: 相似性分数阈值
- `fetch_k`: MMR搜索中获取的候选文档数
- `lambda_mult`: MMR多样性参数

```python
# 限制返回文档数量
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 设置相似性阈值
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)

# MMR参数配置
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)
```

### 2. 多查询检索器 (`MultiQueryRetriever`)

多查询检索器通过LLM生成多个不同视角的查询，然后合并检索结果，以提高检索的全面性。

#### 2.1 核心功能

**自动查询生成**:
- 使用LLM从用户查询生成多个不同视角的查询
- 对每个生成的查询进行独立检索
- 合并去重所有检索结果

**优势**:
- 克服单一查询表达限制
- 减少因查询表述不准确导致的检索遗漏
- 提供更全面的检索结果

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# 创建多查询检索器
llm = ChatOpenAI(temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 使用检索器
docs = retriever.invoke("什么是任务分解的方法？")
```

#### 2.2 自定义查询生成

可以通过自定义提示模板来控制查询生成的行为：

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser

# 自定义输出解析器
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))

# 自定义提示模板
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一个AI助手。你的任务是生成5个不同版本的用户问题，
    以便从向量数据库中检索相关文档。通过从不同角度生成问题，
    帮助用户克服基于距离的相似性搜索的局限性。
    请用换行符分隔这些替代问题。
    原始问题: {question}"""
)

# 创建自定义检索器
llm_chain = QUERY_PROMPT | llm | LineListOutputParser()
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(),
    llm_chain=llm_chain,
    parser_key="lines"
)
```

### 3. 上下文压缩检索器 (`ContextualCompressionRetriever`)

上下文压缩检索器通过压缩和过滤检索到的文档，只返回与查询最相关的内容。

**📖 本节内容导航**:
- 3.1 核心组件概览
- **3.2 LLMChainFilter 工作原理详解** ⭐ *新增*
- 3.3 压缩器管道使用

#### 3.1 核心组件

**文档压缩器类型**:
- `LLMChainExtractor`: 使用LLM提取相关内容
- `LLMChainFilter`: 使用LLM过滤无关文档
- `LLMListwiseRerank`: 使用LLM重新排序文档
- `EmbeddingsFilter`: 基于嵌入相似性过滤

#### 3.2 LLMChainFilter 工作原理详解

LLMChainFilter 是一个智能文档过滤器，利用大语言模型的理解能力来判断文档与查询的相关性。

**🔍 核心工作流程**:

1. **输入构造阶段**
   - 使用 `get_input` 函数将查询和每个文档组合成 LLM 输入
   - 默认会创建包含查询和文档内容的提示词

2. **批量 LLM 推理**
   - 调用 `llm_chain.batch()` 同时处理多个文档
   - LLM 分析每个文档与查询的语义相关性

3. **结果解析与过滤**
   - 解析 LLM 输出为布尔值（相关/不相关）
   - 只保留被判定为相关的文档

**💡 工作特点**:
- **语义理解**: 基于深度语言理解，不仅仅是关键词匹配
- **内容保持**: 只做过滤，不压缩文档内容
- **准确性高**: 相比简单的相似度计算，判断更准确
- **成本较高**: 需要调用 LLM API，速度和成本比嵌入过滤器高

**🆚 性能对比**:

| 压缩器类型 | 速度 | 成本 | 准确性 | 适用场景 |
|------------|------|------|--------|----------|
| **LLMChainFilter** | 慢 | 高 | 很高 | 精确过滤，质量优先 |
| **EmbeddingsFilter** | 快 | 低 | 中等 | 快速过滤，效率优先 |
| **LLMChainExtractor** | 慢 | 高 | 高 | 内容压缩，精确提取 |

**🎯 最佳应用场景**:
- 需要高精度文档过滤的 RAG 系统
- 对准确性要求高于速度的知识问答
- 复杂语义判断的文档筛选任务

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
    EmbeddingsFilter
)

# LLM提取器 - 提取相关内容
llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# LLM过滤器 - 使用语言模型智能过滤无关文档
filter_compressor = LLMChainFilter.from_llm(llm)
filter_retriever = ContextualCompressionRetriever(
    base_compressor=filter_compressor,
    base_retriever=retriever
)

# 实际使用示例：过滤效果对比
# 相关查询 - 会保留更多相关文档
relevant_docs = filter_retriever.invoke("Ketanji Brown Jackson nomination")

# 不相关查询 - 会过滤掉大部分无关文档  
irrelevant_docs = filter_retriever.invoke("cooking recipes and food preparation")

print(f"相关查询结果数: {len(relevant_docs)}")
print(f"不相关查询结果数: {len(irrelevant_docs)}")
# 预期: relevant_docs 数量 > irrelevant_docs 数量

# 嵌入过滤器 - 基于相似性过滤
embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.76
)
embedding_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever
)
```

#### 3.3 压缩器管道

可以组合多个压缩器形成处理管道：

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter

# 创建处理管道
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

# 组装管道：分割 -> 去重 -> 相关性过滤
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever
)
```

### 4. 检索器性能优化

#### 4.1 搜索策略选择

**相似性搜索 vs MMR**:
- 相似性搜索：速度快，适合精确匹配
- MMR搜索：结果多样化，避免冗余，适合探索性查询

**阈值设置**:
- 高阈值（0.8+）：精确匹配，结果少但质量高
- 中阈值（0.5-0.8）：平衡精度和召回率
- 低阈值（0.3-0.5）：广泛匹配，结果多但可能有噪声

#### 4.2 批量检索优化

```python
# 异步批量检索
import asyncio

async def batch_retrieve(retriever, queries: List[str]):
    """批量异步检索"""
    tasks = [retriever.ainvoke(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results

# 缓存检索结果
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieve(query: str):
    """缓存检索结果"""
    return retriever.invoke(query)
```

## 使用最佳实践

### 1. 检索器选择指南

**场景1: 精确问答**
- 使用基础相似性检索 + 高阈值
- 适合事实查询、定义查询

**场景2: 探索性搜索**
- 使用MMR检索或多查询检索
- 适合开放性问题、研究性查询

**场景3: 长文档处理**
- 使用上下文压缩检索器
- 减少无关内容，提高LLM处理效率

**场景4: 实时应用**
- 使用嵌入过滤器而非LLM压缩器
- 平衡精度和速度

### 2. 参数调优建议

**k值设置**:
- 问答系统：k=3-5
- 文档推荐：k=10-20
- 内容生成：k=5-10

**阈值设置**:
- 严格匹配：threshold=0.8+
- 一般用途：threshold=0.6-0.8
- 广泛搜索：threshold=0.3-0.6

**MMR参数**:
- 多样性优先：lambda_mult=0.3-0.5
- 相关性优先：lambda_mult=0.7-0.9
- 平衡模式：lambda_mult=0.5-0.7

### 3. 错误处理和监控

```python
import logging
from typing import Optional

# 设置检索器日志
logging.basicConfig()
logging.getLogger("langchain.retrievers").setLevel(logging.INFO)

def robust_retrieve(retriever, query: str, fallback_k: int = 10) -> List[Document]:
    """
    健壮的检索函数
    
    Args:
        retriever: 检索器实例
        query: 查询字符串
        fallback_k: 降级时的文档数量
        
    Returns:
        检索到的文档列表
    """
    try:
        # 尝试正常检索
        docs = retriever.invoke(query)
        if not docs:
            # 如果没有结果，降低阈值重试
            fallback_retriever = vectorstore.as_retriever(search_kwargs={"k": fallback_k})
            docs = fallback_retriever.invoke(query)
        return docs
    except Exception as e:
        logging.error(f"检索失败: {e}")
        # 返回空结果或默认文档
        return []
```

## 性能基准测试

测试套件包含以下性能评估：

1. **检索延迟测试**: 不同检索器的响应时间对比
2. **检索质量测试**: 基于人工标注的相关性评估
3. **缓存效果测试**: 缓存对性能提升的量化分析
4. **批量处理测试**: 大批量查询的处理能力
5. **资源消耗测试**: 内存和CPU使用情况监控

## LLMChainFilter 实战技巧 💡

### 🎯 优化策略

1. **温度参数调优**
   ```python
   # 低温度确保稳定的过滤判断
   llm = ChatOpenAI(temperature=0.1)  # 推荐值：0.1-0.3
   filter_compressor = LLMChainFilter.from_llm(llm)
   ```

2. **自定义提示词优化**
   ```python
   from langchain_core.prompts import PromptTemplate
   
   # 自定义过滤提示词
   custom_prompt = PromptTemplate.from_template("""
   查询: {query}
   文档: {document}
   
   请判断该文档是否与查询直接相关？
   只回答"是"或"否"。
   """)
   
   filter_compressor = LLMChainFilter.from_llm(llm, prompt=custom_prompt)
   ```

3. **性能与成本平衡**
   ```python
   # 混合策略：先用快速过滤，再用精确过滤
   pipeline = DocumentCompressorPipeline([
       EmbeddingsFilter(similarity_threshold=0.3),  # 初步过滤
       LLMChainFilter.from_llm(llm)                # 精确过滤
   ])
   ```

### ⚠️ 注意事项

1. **API配额管理**: 多查询检索器和LLM压缩器会增加API调用
2. **延迟考虑**: LLM压缩器会增加检索延迟
3. **成本控制**: 权衡检索质量和API成本
4. **结果缓存**: 对频繁查询使用缓存机制
5. **降级策略**: 准备检索失败时的降级方案 