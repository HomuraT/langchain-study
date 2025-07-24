# 嵌入模型测试套件

这个测试套件提供了对LangChain嵌入模型功能的全面测试，包括基础功能测试、缓存功能测试和性能评估。

## 目录结构

```
test_embeddings/
├── README.md                    # 本文件，使用说明
├── run_tests.py                # 测试运行脚本
├── test_basic_embeddings.py    # 基础嵌入功能测试
└── test_cached_embeddings.py   # 缓存嵌入功能测试
```

## 测试模块说明

### 1. 基础嵌入测试 (`test_basic_embeddings.py`)

#### TestBasicEmbeddings 类
- `test_embed_documents_basic`: 测试基础文档嵌入功能
- `test_embed_query_basic`: 测试基础查询嵌入功能
- `test_embedding_consistency`: 测试嵌入结果的一致性
- `test_embedding_similarity`: 测试嵌入相似性计算
- `test_empty_text_handling`: 测试空文本处理
- `test_long_text_handling`: 测试长文本处理
- `test_batch_vs_individual_embeddings`: 测试批量与单独嵌入的一致性

#### TestEmbeddingModels 类
- `test_small_model_embedding`: 测试小型嵌入模型
- `test_model_dimension_consistency`: 测试模型维度一致性

### 2. 缓存嵌入测试 (`test_cached_embeddings.py`)

#### TestCachedEmbeddings 类
- `test_local_file_store_cache`: 测试本地文件存储缓存
- `test_in_memory_store_cache`: 测试内存字节存储缓存
- `test_query_caching`: 测试查询嵌入缓存
- `test_faiss_integration`: 测试与FAISS向量存储的集成
- `test_namespace_isolation`: 测试命名空间隔离
- `test_batch_processing`: 测试批处理功能

#### TestEmbeddingPerformance 类
- `test_cache_vs_no_cache_performance`: 缓存与非缓存性能对比测试

## 快速开始

### 1. 环境配置

确保你的环境中已安装必要的依赖：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖（如果还未安装）
pip install langchain-openai langchain-community faiss-cpu numpy
```

### 2. 配置API

确保在 `src/config/api.py` 中正确配置了API信息：

```python
apis = {
    "local": {
        "base_url": "http://localhost:8212/v1",
        "api_key": "your-api-key-here",
    }
}
```

### 3. 运行测试

#### 运行所有测试
```bash
cd unitests/test_embeddings
python run_tests.py
```

#### 运行特定类型的测试
```bash
# 只运行基础嵌入测试
python run_tests.py basic

# 只运行缓存嵌入测试
python run_tests.py cache
```

#### 运行特定测试类
```bash
# 运行基础嵌入测试类
python run_tests.py TestBasicEmbeddings

# 运行缓存嵌入测试类
python run_tests.py TestCachedEmbeddings
```

#### 运行特定测试方法
```bash
# 运行特定的测试方法
python run_tests.py TestBasicEmbeddings test_embed_documents_basic
```

### 4. 直接运行单个测试文件

```bash
# 运行基础嵌入测试
python test_basic_embeddings.py

# 运行缓存嵌入测试
python test_cached_embeddings.py
```

## 测试功能详解

### 基础嵌入功能测试

1. **文档嵌入测试**: 验证 `embed_documents` 方法能正确处理多个文档
2. **查询嵌入测试**: 验证 `embed_query` 方法能正确处理单个查询
3. **一致性测试**: 确保相同输入产生相同输出
4. **相似性测试**: 验证语义相似的文本具有更高的向量相似度
5. **边界情况测试**: 测试空文本、长文本等特殊情况

### 缓存嵌入功能测试

1. **本地文件缓存**: 使用 `LocalFileStore` 进行持久化缓存
2. **内存缓存**: 使用 `InMemoryByteStore` 进行临时缓存
3. **查询缓存**: 测试查询嵌入的缓存功能
4. **向量存储集成**: 测试与FAISS等向量数据库的集成
5. **命名空间隔离**: 确保不同模型的缓存不会冲突
6. **性能测试**: 对比缓存前后的性能差异

## 测试输出示例

运行测试时，你会看到详细的输出信息：

```
=== 测试基础文档嵌入功能 ===
文档数量: 4
嵌入向量维度: 1536
每个嵌入向量前5个值示例: [0.0123, -0.0456, 0.0789, ...]

=== 测试本地文件存储缓存 ===
初始缓存键数量: 0
第一次嵌入耗时: 1.234秒
缓存中的键数量: 4
第二次嵌入耗时: 0.012秒
性能提升: 102.83倍
```

## 故障排除

### 常见问题

1. **API连接失败**
   - 检查 `src/config/api.py` 中的API配置
   - 确保API服务正在运行

2. **导入错误**
   - 确保在正确的目录下运行测试
   - 检查Python路径配置

3. **依赖缺失**
   - 运行 `pip install -r requirements.txt` 安装依赖
   - 确保虚拟环境已激活

### 调试技巧

1. **增加详细输出**：测试已设置 `verbosity=2`，会显示详细信息
2. **单独运行测试**：可以单独运行特定测试方法来定位问题
3. **检查日志**：观察测试输出中的错误信息和性能数据

## 扩展测试

### 添加新的嵌入模型测试

1. 在 `TestEmbeddingModels` 类中添加新的测试方法
2. 配置新的嵌入模型实例
3. 编写相应的测试逻辑

### 添加新的缓存存储测试

1. 在 `TestCachedEmbeddings` 类中添加新的测试方法
2. 实现新的存储后端（如Redis、数据库等）
3. 验证缓存功能的正确性

## 性能基准

测试套件包含性能测试，可以用来：

1. **评估缓存效果**: 对比有无缓存的性能差异
2. **模型对比**: 比较不同嵌入模型的性能
3. **批处理优化**: 验证批处理相对于单个处理的优势

## 贡献指南

如果你想为测试套件做出贡献：

1. 确保新测试遵循现有的命名约定
2. 添加适当的文档字符串和类型注解
3. 在测试中包含合适的断言和错误消息
4. 更新README文档以反映新增功能

## 许可证

本测试套件遵循项目的整体许可证协议。 