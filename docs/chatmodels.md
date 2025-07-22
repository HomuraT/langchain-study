# 聊天模型

## 概述

大型语言模型（LLMs）是先进的机器学习模型，在广泛的语言相关任务中表现出色，如文本生成、翻译、摘要、问答等，无需针对每种场景进行特定的任务微调。

现代LLMs通常通过聊天模型接口访问，该接口接受[消息](https://python.langchain.com/docs/concepts/messages/)列表作为输入并返回[消息](https://python.langchain.com/docs/concepts/messages/)作为输出。

最新一代的聊天模型提供了额外的功能：

- **[工具调用](https://python.langchain.com/docs/concepts/tool_calling/)**：许多流行的聊天模型提供原生的[工具调用](https://python.langchain.com/docs/concepts/tool_calling/)API。该API允许开发者构建丰富的应用程序，使LLMs能够与外部服务、API和数据库交互。工具调用还可用于从非结构化数据中提取结构化信息并执行各种其他任务。
- **[结构化输出](https://python.langchain.com/docs/concepts/structured_outputs/)**：一种使聊天模型以结构化格式（如匹配给定模式的JSON）响应的技术。
- **[多模态性](https://python.langchain.com/docs/concepts/multimodality/)**：处理文本以外数据的能力；例如，图像、音频和视频。

## 特性

LangChain为使用来自不同提供商的聊天模型提供了一致的接口，同时为监控、调试和优化使用LLMs的应用程序性能提供了额外功能。

- 与许多聊天模型提供商的集成（例如，Anthropic、OpenAI、Ollama、Microsoft Azure、Google Vertex、Amazon Bedrock、Hugging Face、Cohere、Groq）。请查看[聊天模型集成](https://python.langchain.com/docs/integrations/chat/)以获取最新的支持模型列表。
- 使用LangChain的[消息](https://python.langchain.com/docs/concepts/messages/)格式或OpenAI格式。
- 标准[工具调用API](https://python.langchain.com/docs/concepts/tool_calling/)：用于将工具绑定到模型、访问模型发出的工具调用请求以及将工具结果发送回模型的标准接口。
- 通过`with_structured_output`方法[构建输出](https://python.langchain.com/docs/concepts/structured_outputs/#structured-output-method)的标准API。
- 提供对[异步编程](https://python.langchain.com/docs/concepts/async/)、[高效批处理](https://python.langchain.com/docs/concepts/runnables/#optimized-parallel-execution-batch)、[丰富流式API](https://python.langchain.com/docs/concepts/streaming/)的支持。
- 与[LangSmith](https://docs.smith.langchain.com)集成，用于监控和调试基于LLMs的生产级应用程序。
- 额外功能，如标准化[令牌使用](https://python.langchain.com/docs/concepts/messages/#aimessage)、[速率限制](#速率限制)、[缓存](#缓存)等。

## 集成

LangChain有许多聊天模型集成，允许您使用来自不同提供商的各种模型。

这些集成有两种类型：

- **官方模型**：这些是LangChain和/或模型提供商官方支持的模型。您可以在`langchain-<provider>`包中找到这些模型。
- **社区模型**：这些模型主要由社区贡献和支持。您可以在`langchain-community`包中找到这些模型。

LangChain聊天模型遵循在类名前加"Chat"前缀的命名约定（例如，`ChatOllama`、`ChatAnthropic`、`ChatOpenAI`等）。

请查看[聊天模型集成](https://python.langchain.com/docs/integrations/chat/)以获取支持模型的列表。

> **注意**：名称中**不**包含"Chat"前缀或名称中包含"LLM"后缀的模型通常指不遵循聊天模型接口的较旧模型，而是使用接受字符串作为输入并返回字符串作为输出的接口。

## 接口

LangChain聊天模型实现了[BaseChatModel](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html)接口。由于`BaseChatModel`也实现了[Runnable接口](https://python.langchain.com/docs/concepts/runnables/)，聊天模型支持[标准流式接口](https://python.langchain.com/docs/concepts/streaming/)、[异步编程](https://python.langchain.com/docs/concepts/async/)、优化[批处理](https://python.langchain.com/docs/concepts/runnables/#optimized-parallel-execution-batch)等。请参阅[Runnable接口](https://python.langchain.com/docs/concepts/runnables/)了解更多详细信息。

聊天模型的许多关键方法以[消息](https://python.langchain.com/docs/concepts/messages/)作为输入并返回消息作为输出。

聊天模型提供一套标准参数，可用于配置模型。这些参数通常用于控制模型的行为，如输出的温度、响应中的最大令牌数以及等待响应的最大时间。请参阅[标准参数](#标准参数)部分了解更多详细信息。

> **注意**：在文档中，我们经常互换使用"LLM"和"聊天模型"这两个术语。这是因为大多数现代LLMs通过聊天模型接口向用户公开。
>
> 但是，LangChain也有一些不遵循聊天模型接口的较旧LLMs的实现，而是使用接受字符串作为输入并返回字符串作为输出的接口。这些模型通常没有"Chat"前缀（例如，`Ollama`、`Anthropic`、`OpenAI`等）。这些模型实现[BaseLLM](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.BaseLLM.html#langchain_core.language_models.llms.BaseLLM)接口，可能以"LLM"后缀命名（例如，`OllamaLLM`、`AnthropicLLM`、`OpenAILLM`等）。通常，用户不应使用这些模型。

## 关键方法

聊天模型的关键方法包括：

- **invoke**：与聊天模型交互的主要方法。它接受[消息](https://python.langchain.com/docs/concepts/messages/)列表作为输入并返回消息列表作为输出。
- **stream**：允许您在生成聊天模型输出时流式传输输出的方法。
- **batch**：允许您将对聊天模型的多个请求批处理在一起以进行更高效处理的方法。
- **bind_tools**：允许您将工具绑定到聊天模型以在模型的执行上下文中使用的方法。
- **with_structured_output**：围绕`invoke`方法的包装器，用于原生支持[结构化输出](https://python.langchain.com/docs/concepts/structured_outputs/)的模型。

其他重要方法可在[BaseChatModel API参考](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html)中找到。

## 输入和输出

现代LLMs通常通过聊天模型接口访问，该接口接受[消息](https://python.langchain.com/docs/concepts/messages/)作为输入并返回[消息](https://python.langchain.com/docs/concepts/messages/)作为输出。消息通常与角色（例如，"system"、"human"、"assistant"）和一个或多个包含文本或潜在多模态数据（例如，图像、音频、视频）的内容块相关联。

LangChain支持两种消息格式与聊天模型交互：

- **LangChain消息格式**：LangChain自己的消息格式，默认使用，由LangChain内部使用。
- **OpenAI消息格式**：OpenAI的消息格式。

## 标准参数

许多聊天模型具有可用于配置模型的标准化参数：

| 参数 | 描述 |
|------|------|
| model | 您要使用的特定AI模型的名称或标识符（例如，"gpt-3.5-turbo"或"gpt-4"） |
| temperature | 控制模型输出的随机性。较高的值（例如，1.0）使响应更具创造性，而较低的值（例如，0.0）使它们更具确定性和专注性 |
| timeout | 在取消请求之前等待模型响应的最大时间（以秒为单位）。确保请求不会无限期挂起 |
| max_tokens | 限制响应中令牌（单词和标点符号）的总数。这控制输出的长度 |
| stop | 指定停止序列，指示模型何时应停止生成令牌。例如，您可能使用特定字符串来表示响应的结束 |
| max_retries | 如果由于网络超时或速率限制等问题而失败，系统将尝试重新发送请求的最大次数 |
| api_key | 与模型提供商进行身份验证所需的API密钥。这通常在您注册访问模型时发出 |
| base_url | 发送请求的API端点的URL。这通常由模型提供商提供，对于定向您的请求是必需的 |
| rate_limiter | 可选的[BaseRateLimiter](https://python.langchain.com/api_reference/core/rate_limiters/langchain_core.rate_limiters.BaseRateLimiter.html#langchain_core.rate_limiters.BaseRateLimiter)来间隔请求以避免超过速率限制。有关更多详细信息，请参阅[速率限制](#速率限制) |

需要注意的一些重要事项：

- 标准参数仅适用于公开具有预期功能的参数的模型提供商。例如，某些提供商不公开最大输出令牌的配置，因此这些提供商不能支持`max_tokens`。
- 标准参数目前仅在具有自己集成包的集成上强制执行（例如`langchain-openai`、`langchain-anthropic`等），它们不在`langchain-community`中的模型上强制执行。

聊天模型还接受特定于该集成的其他参数。要查找聊天模型支持的所有参数，请转到该模型的相应[API参考](https://python.langchain.com/api_reference/)。

## 工具调用

聊天模型可以调用[工具](https://python.langchain.com/docs/concepts/tools/)来执行任务，如从数据库获取数据、进行API请求或运行自定义代码。请参阅[工具调用](https://python.langchain.com/docs/concepts/tool_calling/)指南了解更多信息。

## 结构化输出

可以要求聊天模型以特定格式（例如，JSON或匹配特定模式）响应。此功能对于信息提取任务极其有用。请在[结构化输出](https://python.langchain.com/docs/concepts/structured_outputs/)指南中阅读有关该技术的更多信息。

## 多模态性

大型语言模型（LLMs）不仅限于处理文本。它们还可以用于处理其他类型的数据，如图像、音频和视频。这被称为[多模态性](https://python.langchain.com/docs/concepts/multimodality/)。

目前，只有一些LLMs支持多模态输入，几乎没有支持多模态输出。请查阅特定模型文档了解详细信息。

## 上下文窗口

聊天模型的上下文窗口是指模型一次可以处理的输入序列的最大大小。虽然现代LLMs的上下文窗口相当大，但它们仍然存在开发人员在使用聊天模型时必须牢记的限制。

如果输入超过上下文窗口，模型可能无法处理整个输入并可能引发错误。在对话应用程序中，这特别重要，因为上下文窗口决定了模型在整个对话中可以"记住"多少信息。开发人员通常需要在上下文窗口内管理输入，以维持连贯的对话而不超过限制。有关在对话中处理内存的更多详细信息，请参阅[内存](https://langchain-ai.github.io/langgraph/concepts/memory/)。

输入的大小以[令牌](https://python.langchain.com/docs/concepts/tokens/)为单位测量，令牌是模型使用的处理单元。

## 高级主题

### 速率限制

许多聊天模型提供商对在给定时间段内可以发出的请求数量施加限制。

如果您遇到速率限制，您通常会从提供商收到速率限制错误响应，并需要等待后再发出更多请求。

您有几个选项来处理速率限制：

- **通过间隔请求尝试避免遇到速率限制**：聊天模型接受可在初始化期间提供的`rate_limiter`参数。此参数用于控制向模型提供商发出请求的速率。间隔对给定模型的请求是在基准测试模型以评估其性能时特别有用的策略。请参阅[如何处理速率限制](https://python.langchain.com/docs/how_to/chat_model_rate_limiting/)以获取有关如何使用此功能的更多信息。
- **尝试从速率限制错误中恢复**：如果您收到速率限制错误，您可以在重试请求之前等待一定时间。每次后续速率限制错误可以增加等待时间。聊天模型有一个`max_retries`参数，可用于控制重试次数。有关更多信息，请参阅[标准参数](#标准参数)部分。
- **回退到另一个聊天模型**：如果您在一个聊天模型上遇到速率限制，您可以切换到另一个没有速率限制的聊天模型。

### 缓存

聊天模型API可能很慢，因此一个自然的问题是是否缓存先前对话的结果。理论上，缓存可以通过减少向模型提供商发出的请求数量来帮助提高性能。实际上，缓存聊天模型响应是一个复杂的问题，应谨慎处理。

原因是，如果依靠缓存模型的确切输入，在对话中第一次或第二次交互后获得缓存命中的可能性很小。例如，您认为多个对话以完全相同的消息开始的可能性有多大？完全相同的三条消息呢？

一种替代方法是使用语义缓存，您根据输入的含义而不是确切的输入本身来缓存响应。这在某些情况下可能有效，但在其他情况下则不然。

语义缓存在应用程序的关键路径上引入了对另一个模型的依赖（例如，语义缓存可能依靠[嵌入模型](https://python.langchain.com/docs/concepts/embedding_models/)将文本转换为向量表示），并且不能保证准确捕获输入的含义。

但是，可能存在缓存聊天模型响应有益的情况。例如，如果您有一个用于回答常见问题的聊天模型，缓存响应可以帮助减少模型提供商的负载、成本并改善响应时间。

请参阅[如何缓存聊天模型响应](https://python.langchain.com/docs/how_to/chat_model_caching/)指南了解更多详细信息。

## 相关资源

- 使用聊天模型的操作指南：[操作指南](https://python.langchain.com/docs/how_to/#chat-models)
- 支持的聊天模型列表：[聊天模型集成](https://python.langchain.com/docs/integrations/chat/)

### 概念指南

- [消息](https://python.langchain.com/docs/concepts/messages/)
- [工具调用](https://python.langchain.com/docs/concepts/tool_calling/)
- [多模态性](https://python.langchain.com/docs/concepts/multimodality/)
- [结构化输出](https://python.langchain.com/docs/concepts/structured_outputs/)
- [令牌](https://python.langchain.com/docs/concepts/tokens/)
