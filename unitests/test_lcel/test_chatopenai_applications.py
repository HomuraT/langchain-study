"""
LangChain Expression Language (LCEL) 与 ChatOpenAI 应用测试

测试ChatOpenAI与LCEL结合的实际应用场景：
- 智能问答助手
- 文本分析与总结
- 角色扮演对话
- 多步骤推理链
- 条件对话流
- 内容生成管道

作者: AI Assistant
创建时间: 2025年
"""

import unittest
import asyncio
from typing import Dict, Any, List, Optional, Union
from langchain_core.runnables import (
    RunnableSequence, 
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch
)
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.config.api import apis


class TestChatOpenAIApplications(unittest.TestCase):
    """ChatOpenAI应用场景测试类"""
    
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
            max_tokens=1000,
            timeout=30
        )
        
        # 创建不同温度的模型用于不同场景
        cls.creative_model = ChatOpenAI(
            base_url=cls.config["base_url"],
            api_key=cls.config["api_key"],
            model="gpt-4o-mini",
            temperature=0.9,  # 高创造性
            max_tokens=800,
            timeout=30
        )
        
        cls.analytical_model = ChatOpenAI(
            base_url=cls.config["base_url"],
            api_key=cls.config["api_key"],
            model="gpt-4o-mini",
            temperature=0.1,  # 低创造性，更精确
            max_tokens=1200,
            timeout=30
        )
    
    def setUp(self) -> None:
        """
        每个测试方法前的设置
        
        输入: 无
        输出: 无
        """
        self.str_parser = StrOutputParser()
        self.json_parser = JsonOutputParser()
    
    def test_intelligent_qa_assistant(self) -> None:
        """
        测试智能问答助手应用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试智能问答助手 ===")
        
        try:
            # 1. 构建智能问答链
            system_prompt = SystemMessagePromptTemplate.from_template(
                "你是一个专业的智能助手，能够准确回答各种问题。"
                "请根据问题类型调整回答风格：技术问题要详细，日常问题要简洁。"
            )
            human_prompt = HumanMessagePromptTemplate.from_template("问题：{question}")
            
            qa_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            
            # 添加问题分类和答案优化
            question_classifier = RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "type": "技术问题" if any(word in x["question"].lower() 
                                         for word in ["编程", "代码", "算法", "技术", "开发", "python", "ai"]) 
                           else "日常问题"
                }
            )
            
            # 根据问题类型调整回答
            answer_formatter = RunnableLambda(
                lambda x: f"【{x['type']}】\n{x['answer']}\n\n💡 提示：{'需要详细解释时请告诉我' if x['type'] == '技术问题' else '还有其他问题吗？'}"
            )
            
            # 构建完整的问答链
            qa_chain = (
                question_classifier
                | RunnableParallel({
                    "type": lambda x: x["type"],
                    "answer": qa_prompt | self.model | self.str_parser
                })
                | answer_formatter
            )
            
            # 测试不同类型的问题
            test_questions = [
                {"question": "什么是Python中的装饰器？"},
                {"question": "今天天气怎么样？"},
                {"question": "如何实现二分查找算法？"}
            ]
            
            for q in test_questions:
                result = qa_chain.invoke(q)
                print(f"\n问题: {q['question']}")
                print(f"回答: {result}")
                self.assertIsInstance(result, str)
                self.assertIn("【", result)  # 检查分类标签
            
            print("✅ 智能问答助手测试通过!")
            
        except Exception as e:
            print(f"❌ 智能问答助手测试失败: {e}")
    
    def test_text_analysis_and_summary(self) -> None:
        """
        测试文本分析与总结应用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试文本分析与总结 ===")
        
        try:
            # 构建文本分析管道
            analysis_prompt = ChatPromptTemplate.from_template(
                "请对以下文本进行详细分析，包括主题、情感、关键信息：\n\n{text}"
            )
            
            summary_prompt = ChatPromptTemplate.from_template(
                "请用3句话总结以下内容的核心要点：\n\n{text}"
            )
            
            keywords_prompt = ChatPromptTemplate.from_template(
                "请提取以下文本的5个关键词（用逗号分隔）：\n\n{text}"
            )
            
            # 文本预处理
            text_preprocessor = RunnableLambda(
                lambda x: {
                    "text": x["text"],
                    "word_count": len(x["text"].split()),
                    "char_count": len(x["text"])
                }
            )
            
            # 并行分析
            analysis_parallel = RunnableParallel({
                "analysis": analysis_prompt | self.analytical_model | self.str_parser,
                "summary": summary_prompt | self.analytical_model | self.str_parser,
                "keywords": keywords_prompt | self.analytical_model | self.str_parser,
                "metadata": RunnablePassthrough()
            })
            
            # 结果整合
            result_formatter = RunnableLambda(
                lambda x: f"""📊 文本分析报告
                
📈 基本信息：
- 字数：{x['metadata']['word_count']} 词
- 字符数：{x['metadata']['char_count']} 字符

🔍 详细分析：
{x['analysis']}

📝 核心总结：
{x['summary']}

🏷️ 关键词：
{x['keywords']}
"""
            )
            
            # 完整分析链
            analysis_chain = (
                text_preprocessor
                | analysis_parallel
                | result_formatter
            )
            
            # 测试文本
            test_text = """
            人工智能技术正在快速发展，特别是大语言模型的出现，为自然语言处理领域带来了革命性的变化。
            这些模型不仅能够理解复杂的语言结构，还能够生成高质量的文本内容。
            然而，随着AI技术的普及，我们也需要关注其带来的伦理和安全问题，确保技术的发展能够造福人类社会。
            """
            
            result = analysis_chain.invoke({"text": test_text})
            print(f"分析结果:\n{result}")
            
            self.assertIsInstance(result, str)
            self.assertIn("文本分析报告", result)
            self.assertIn("字数", result)
            
            print("✅ 文本分析与总结测试通过!")
            
        except Exception as e:
            print(f"❌ 文本分析与总结测试失败: {e}")
    
    def test_role_playing_dialogue(self) -> None:
        """
        测试角色扮演对话应用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试角色扮演对话 ===")
        
        try:
            # 定义不同角色的prompt
            roles = {
                "teacher": "你是一位耐心的老师，善于用简单易懂的方式解释复杂概念，喜欢举例说明。",
                "scientist": "你是一位严谨的科学家，回答问题时会提供科学依据和数据支持。",
                "poet": "你是一位富有想象力的诗人，习惯用优美的语言和比喻来表达观点。",
                "coach": "你是一位激励型教练，总是积极正面，善于鼓励和指导他人。"
            }
            
            # 角色选择器
            role_selector = RunnableLambda(
                lambda x: {
                    "role": x["role"],
                    "question": x["question"],
                    "system_message": roles.get(x["role"], roles["teacher"])
                }
            )
            
            # 动态创建角色prompt
            def create_role_prompt(data):
                return ChatPromptTemplate.from_messages([
                    SystemMessage(content=data["system_message"]),
                    HumanMessage(content=data["question"])
                ])
            
            # 角色对话链
            role_dialogue_chain = (
                role_selector
                | RunnableLambda(lambda x: create_role_prompt(x).format_messages())
                | self.creative_model
                | self.str_parser
            )
            
            # 测试不同角色对同一问题的回答
            test_cases = [
                {"role": "teacher", "question": "什么是机器学习？"},
                {"role": "scientist", "question": "什么是机器学习？"},
                {"role": "poet", "question": "什么是机器学习？"},
                {"role": "coach", "question": "如何学好编程？"}
            ]
            
            for case in test_cases:
                result = role_dialogue_chain.invoke(case)
                print(f"\n角色: {case['role']}")
                print(f"问题: {case['question']}")
                print(f"回答: {result[:200]}...")  # 只显示前200字符
                
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 50)  # 确保有实质性内容
            
            print("✅ 角色扮演对话测试通过!")
            
        except Exception as e:
            print(f"❌ 角色扮演对话测试失败: {e}")
    
    def test_multi_step_reasoning_chain(self) -> None:
        """
        测试多步骤推理链应用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试多步骤推理链 ===")
        
        try:
            # 步骤1：问题分解
            decompose_prompt = ChatPromptTemplate.from_template(
                "请将以下复杂问题分解为3-5个具体的子问题：\n问题：{question}\n\n"
                "请用编号列表格式回答。"
            )
            
            # 步骤2：逐步分析
            analyze_prompt = ChatPromptTemplate.from_template(
                "原问题：{original_question}\n"
                "子问题列表：{sub_questions}\n\n"
                "请逐一分析每个子问题，提供详细解答。"
            )
            
            # 步骤3：综合结论
            synthesize_prompt = ChatPromptTemplate.from_template(
                "基于以下分析内容，请提供一个综合性的最终答案：\n\n"
                "原问题：{original_question}\n"
                "详细分析：{analysis}\n\n"
                "请给出清晰、完整的最终答案。"
            )
            
            # 构建多步推理链
            reasoning_chain = (
                RunnablePassthrough.assign(
                    sub_questions=decompose_prompt | self.analytical_model | self.str_parser
                )
                | RunnablePassthrough.assign(
                    analysis=analyze_prompt | self.analytical_model | self.str_parser
                )
                | RunnablePassthrough.assign(
                    final_answer=synthesize_prompt | self.analytical_model | self.str_parser
                )
                | RunnableLambda(lambda x: f"""🧠 多步骤推理结果

📋 原问题：
{x['question']}

🔍 问题分解：
{x['sub_questions']}

📊 详细分析：
{x['analysis']}

🎯 最终答案：
{x['final_answer']}
""")
            )
            
            # 测试复杂问题
            complex_question = "如何设计一个高效且用户友好的在线学习平台？"
            
            result = reasoning_chain.invoke({"question": complex_question})
            print(f"推理结果:\n{result}")
            
            self.assertIsInstance(result, str)
            self.assertIn("原问题", result)
            self.assertIn("问题分解", result)
            self.assertIn("最终答案", result)
            
            print("✅ 多步骤推理链测试通过!")
            
        except Exception as e:
            print(f"❌ 多步骤推理链测试失败: {e}")
    
    def test_conditional_dialogue_flow(self) -> None:
        """
        测试条件对话流应用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试条件对话流 ===")
        
        try:
            # 情感检测
            sentiment_detector = RunnableLambda(
                lambda x: {
                    "text": x["text"],
                    "sentiment": "positive" if any(word in x["text"].lower() 
                                                 for word in ["好", "棒", "喜欢", "满意", "开心", "优秀"]) 
                               else "negative" if any(word in x["text"].lower() 
                                                    for word in ["差", "糟", "讨厌", "不满", "难过", "问题"])
                               else "neutral"
                }
            )
            
            # 不同情感的回应策略
            positive_response = ChatPromptTemplate.from_template(
                "用户表达了积极情感：{text}\n请给出友好、鼓励的回应。"
            )
            
            negative_response = ChatPromptTemplate.from_template(
                "用户表达了消极情感：{text}\n请给出同理心、解决方案导向的回应。"
            )
            
            neutral_response = ChatPromptTemplate.from_template(
                "用户表达了中性观点：{text}\n请给出信息丰富、有帮助的回应。"
            )
            
            # 条件分支
            def route_by_sentiment(data):
                sentiment = data["sentiment"]
                if sentiment == "positive":
                    return positive_response | self.model | self.str_parser
                elif sentiment == "negative":
                    return negative_response | self.model | self.str_parser
                else:
                    return neutral_response | self.model | self.str_parser
            
            # 构建条件对话流
            conditional_flow = (
                sentiment_detector
                | RunnableLambda(route_by_sentiment)
            )
            
            # 测试不同情感的输入
            test_inputs = [
                {"text": "这个产品真的很棒，我很满意！"},
                {"text": "这个服务有很多问题，我很不满意。"},
                {"text": "请介绍一下这个功能的使用方法。"}
            ]
            
            for input_data in test_inputs:
                result = conditional_flow.invoke(input_data)
                print(f"\n输入: {input_data['text']}")
                print(f"回应: {result}")
                
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 20)
            
            print("✅ 条件对话流测试通过!")
            
        except Exception as e:
            print(f"❌ 条件对话流测试失败: {e}")
    
    def test_content_generation_pipeline(self) -> None:
        """
        测试内容生成管道应用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试内容生成管道 ===")
        
        try:
            # 主题扩展
            topic_expander = ChatPromptTemplate.from_template(
                "给定主题：{topic}\n请扩展成一个详细的大纲，包含3-5个主要部分。"
            )
            
            # 内容生成
            content_generator = ChatPromptTemplate.from_template(
                "基于以下大纲，写一篇结构完整的文章：\n{outline}\n\n"
                "要求：\n1. 每个部分都要有具体内容\n2. 语言流畅自然\n3. 逻辑清晰"
            )
            
            # 内容优化
            content_optimizer = ChatPromptTemplate.from_template(
                "请优化以下文章，使其更加生动有趣：\n{content}\n\n"
                "优化要求：\n1. 增加具体例子\n2. 使用更生动的描述\n3. 保持原有结构"
            )
            
            # 添加元数据
            metadata_adder = RunnableLambda(
                lambda x: {
                    "final_content": x["optimized_content"],
                    "word_count": len(x["optimized_content"].split()),
                    "reading_time": f"{len(x['optimized_content'].split()) // 200 + 1}分钟",
                    "generation_chain": "主题扩展 → 内容生成 → 内容优化"
                }
            )
            
            # 最终格式化
            final_formatter = RunnableLambda(
                lambda x: f"""📝 内容生成报告

📊 文章统计：
- 字数：{x['word_count']} 词
- 预计阅读时间：{x['reading_time']}
- 生成流程：{x['generation_chain']}

📄 正文内容：
{x['final_content']}

🔧 生成说明：
本内容通过AI多步骤管道自动生成，包含主题扩展、内容生成和优化三个阶段。
"""
            )
            
            # 构建完整的内容生成管道
            content_pipeline = (
                RunnablePassthrough.assign(
                    outline=topic_expander | self.creative_model | self.str_parser
                )
                | RunnablePassthrough.assign(
                    content=content_generator | self.creative_model | self.str_parser
                )
                | RunnablePassthrough.assign(
                    optimized_content=content_optimizer | self.creative_model | self.str_parser
                )
                | metadata_adder
                | final_formatter
            )
            
            # 测试主题
            test_topic = "人工智能在教育中的应用"
            
            result = content_pipeline.invoke({"topic": test_topic})
            print(f"生成结果:\n{result[:500]}...")  # 只显示前500字符
            
            self.assertIsInstance(result, str)
            self.assertIn("内容生成报告", result)
            self.assertIn("字数", result)
            self.assertIn("正文内容", result)
            
            print("✅ 内容生成管道测试通过!")
            
        except Exception as e:
            print(f"❌ 内容生成管道测试失败: {e}")

    def test_content_generation_pipeline_with_details(self) -> None:
        """
        测试内容生成管道应用（包含所有中间结果和token使用详情）

        输入: 无
        输出: 无
        """
        print("\n=== 测试内容生成管道（详细版本 + Token追踪） ===")

        try:
            from langchain_core.callbacks import UsageMetadataCallbackHandler
            
            # 创建token追踪器
            callback = UsageMetadataCallbackHandler()
            
            # 主题扩展
            topic_expander = ChatPromptTemplate.from_template(
                "给定主题：{topic}\n请扩展成一个详细的大纲，包含3-5个主要部分。"
            )

            # 内容生成
            content_generator = ChatPromptTemplate.from_template(
                "基于以下大纲，写一篇结构完整的文章：\n{step1_outline}\n\n"
                "要求：\n1. 每个部分都要有具体内容\n2. 语言流畅自然\n3. 逻辑清晰"
            )

            # 内容优化
            content_optimizer = ChatPromptTemplate.from_template(
                "请优化以下文章，使其更加生动有趣：\n{step2_content}\n\n"
                "优化要求：\n1. 增加具体例子\n2. 使用更生动的描述\n3. 保持原有结构"
            )

            # 自定义token追踪函数
            def track_step_tokens(step_name: str, step_callback: UsageMetadataCallbackHandler):
                """追踪单个步骤的token使用情况"""
                def wrapper(x):
                    # 执行对应的链
                    if step_name == "outline":
                        chain = topic_expander | self.creative_model | self.str_parser
                    elif step_name == "content":
                        chain = content_generator | self.creative_model | self.str_parser
                    elif step_name == "optimized":
                        chain = content_optimizer | self.creative_model | self.str_parser
                    
                    # 使用独立的callback追踪这一步
                    result = chain.invoke(x, config={"callbacks": [step_callback]})
                    
                    # 打印这一步的token使用情况
                    print(f"\n🔍 步骤 [{step_name}] Token使用情况:")
                    if step_callback.usage_metadata:
                        for model, usage in step_callback.usage_metadata.items():
                            print(f"  模型: {model}")
                            print(f"  输入tokens: {usage.get('input_tokens', 0)}")
                            print(f"  输出tokens: {usage.get('output_tokens', 0)}")
                            print(f"  总tokens: {usage.get('total_tokens', 0)}")
                    
                    return result
                
                return wrapper

            # 创建分步token追踪器
            step1_callback = UsageMetadataCallbackHandler()
            step2_callback = UsageMetadataCallbackHandler()  
            step3_callback = UsageMetadataCallbackHandler()

            # 构建包含token追踪的管道
            detailed_pipeline = (
                # 步骤1：生成大纲并追踪token
                RunnablePassthrough.assign(
                    step1_outline=RunnableLambda(track_step_tokens("outline", step1_callback))
                )
                # 步骤2：生成内容并追踪token
                | RunnablePassthrough.assign(
                    step2_content=RunnableLambda(track_step_tokens("content", step2_callback))
                )
                # 步骤3：优化内容并追踪token
                | RunnablePassthrough.assign(
                    step3_optimized=RunnableLambda(track_step_tokens("optimized", step3_callback))
                )
                # 步骤4：汇总所有信息包括token统计
                | RunnablePassthrough.assign(
                    metadata=RunnableLambda(lambda x: {
                        "word_count_outline": len(x["step1_outline"].split()),
                        "word_count_content": len(x["step2_content"].split()),
                        "word_count_optimized": len(x["step3_optimized"].split()),
                        "processing_steps": ["topic_expansion", "content_generation", "content_optimization"],
                        "token_usage": {
                            "step1_outline": dict(step1_callback.usage_metadata),
                            "step2_content": dict(step2_callback.usage_metadata),
                            "step3_optimized": dict(step3_callback.usage_metadata)
                        }
                    })
                )
            )

            # 执行管道
            test_topic = "人工智能在教育中的应用"
            all_results = detailed_pipeline.invoke({"topic": test_topic})

            # 现在 all_results 包含了所有中间结果和token使用情况
            print("\n=== 完整的处理结果 ===")
            print(f"原始主题: {all_results['topic']}")
            print(f"步骤1大纲字数: {len(all_results['step1_outline'].split())}")
            print(f"步骤2内容字数: {len(all_results['step2_content'].split())}")
            print(f"步骤3优化字数: {len(all_results['step3_optimized'].split())}")

            # 打印详细的token使用统计
            print("\n📊 总体Token使用统计:")
            token_usage = all_results['metadata']['token_usage']
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens = 0
            
            for step_name, step_usage in token_usage.items():
                print(f"\n  {step_name}:")
                for model, usage in step_usage.items():
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    step_total = usage.get('total_tokens', 0)
                    
                    print(f"    模型: {model}")
                    print(f"    输入: {input_tokens} tokens")
                    print(f"    输出: {output_tokens} tokens")
                    print(f"    小计: {step_total} tokens")
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_tokens += step_total
            
            print(f"\n🎯 全流程汇总:")
            print(f"  总输入tokens: {total_input_tokens}")
            print(f"  总输出tokens: {total_output_tokens}")
            print(f"  总计tokens: {total_tokens}")

            # 你可以访问任何中间结果和token信息
            outline = all_results["step1_outline"]
            content = all_results["step2_content"]
            optimized = all_results["step3_optimized"]
            token_stats = all_results["metadata"]["token_usage"]

            # 验证所有数据都存在
            self.assertIn("topic", all_results)
            self.assertIn("step1_outline", all_results)
            self.assertIn("step2_content", all_results)
            self.assertIn("step3_optimized", all_results)
            self.assertIn("token_usage", all_results["metadata"])

            print("\n✅ 详细版内容生成管道（含Token追踪）测试通过!")

        except Exception as e:
            print(f"❌ 详细版内容生成管道测试失败: {e}")

    def test_content_generation_with_token_tracking_v2(self) -> None:
        """
        测试内容生成管道 - 使用context manager方式追踪token
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试内容生成管道（Context Manager Token追踪） ===")
        
        try:
            from langchain_core.callbacks import get_usage_metadata_callback
            
            # 主题扩展
            topic_expander = ChatPromptTemplate.from_template(
                "给定主题：{topic}\n请扩展成一个详细的大纲，包含3-5个主要部分。"
            )

            # 内容生成
            content_generator = ChatPromptTemplate.from_template(
                "基于以下大纲，写一篇结构完整的文章：\n{step1_outline}\n\n"
                "要求：\n1. 每个部分都要有具体内容\n2. 语言流畅自然\n3. 逻辑清晰"
            )

            # 内容优化
            content_optimizer = ChatPromptTemplate.from_template(
                "请优化以下文章，使其更加生动有趣：\n{step2_content}\n\n"
                "优化要求：\n1. 增加具体例子\n2. 使用更生动的描述\n3. 保持原有结构"
            )

            # 使用context manager追踪所有token使用
            with get_usage_metadata_callback() as cb:
                # 构建简化的管道
                simple_pipeline = (
                    RunnablePassthrough.assign(
                        step1_outline=topic_expander | self.creative_model | self.str_parser
                    )
                    | RunnablePassthrough.assign(
                        step2_content=content_generator | self.creative_model | self.str_parser
                    )
                    | RunnablePassthrough.assign(
                        step3_optimized=content_optimizer | self.creative_model | self.str_parser
                    )
                )
                
                # 执行管道，所有token使用都会被自动追踪
                test_topic = "人工智能在教育中的应用"
                results = simple_pipeline.invoke({"topic": test_topic})
                
                # 获取总的token使用情况
                total_usage = cb.usage_metadata
            
            # 显示结果
            print("\n=== 处理结果 ===")
            print(f"原始主题: {results['topic']}")
            print(f"步骤1大纲字数: {len(results['step1_outline'].split())}")
            print(f"步骤2内容字数: {len(results['step2_content'].split())}")
            print(f"步骤3优化字数: {len(results['step3_optimized'].split())}")
            
            # 显示详细的token使用统计
            print("\n📊 Token使用统计:")
            total_input = 0
            total_output = 0
            total_all = 0
            
            for model_name, usage_data in total_usage.items():
                input_tokens = usage_data.get('input_tokens', 0)
                output_tokens = usage_data.get('output_tokens', 0)
                total_tokens = usage_data.get('total_tokens', 0)
                
                print(f"\n模型: {model_name}")
                print(f"  输入tokens: {input_tokens}")
                print(f"  输出tokens: {output_tokens}")
                print(f"  总tokens: {total_tokens}")
                
                # 如果有详细信息，也显示出来
                if 'input_token_details' in usage_data:
                    print(f"  输入详情: {usage_data['input_token_details']}")
                if 'output_token_details' in usage_data:
                    print(f"  输出详情: {usage_data['output_token_details']}")
                
                total_input += input_tokens
                total_output += output_tokens  
                total_all += total_tokens
            
            print(f"\n🎯 整个管道汇总:")
            print(f"  总输入tokens: {total_input}")
            print(f"  总输出tokens: {total_output}")
            print(f"  总计tokens: {total_all}")
            
            # 验证数据
            self.assertIn("topic", results)
            self.assertIn("step1_outline", results)
            self.assertIn("step2_content", results)
            self.assertIn("step3_optimized", results)
            self.assertGreater(total_all, 0, "应该有token使用记录")
            
            print("\n✅ Context Manager Token追踪测试通过!")
            
            # 返回详细结果供进一步分析
            return {
                "results": results,
                "token_usage": total_usage,
                "summary": {
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "total_tokens": total_all
                }
            }
            
        except Exception as e:
            print(f"❌ Context Manager Token追踪测试失败: {e}")
            raise

    def test_content_generation_step_by_step_tokens(self) -> None:
        """
        测试内容生成管道 - 分步实时追踪每个步骤的token使用
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试内容生成管道（分步实时Token追踪） ===")
        
        try:
            from langchain_core.callbacks import get_usage_metadata_callback
            
            # 主题扩展
            topic_expander = ChatPromptTemplate.from_template(
                "给定主题：{topic}\n请扩展成一个详细的大纲，包含3-5个主要部分。"
            )

            # 内容生成
            content_generator = ChatPromptTemplate.from_template(
                "基于以下大纲，写一篇结构完整的文章：\n{outline}\n\n"
                "要求：\n1. 每个部分都要有具体内容\n2. 语言流畅自然\n3. 逻辑清晰"
            )

            # 内容优化
            content_optimizer = ChatPromptTemplate.from_template(
                "请优化以下文章，使其更加生动有趣：\n{content}\n\n"
                "优化要求：\n1. 增加具体例子\n2. 使用更生动的描述\n3. 保持原有结构"
            )
            
            # 创建单独的链
            outline_chain = topic_expander | self.creative_model | self.str_parser
            content_chain = content_generator | self.creative_model | self.str_parser
            optimize_chain = content_optimizer | self.creative_model | self.str_parser
            
            # 存储每步结果和token使用情况
            step_results = {}
            step_tokens = {}
            
            test_topic = "人工智能在教育中的应用"
            
            # 步骤1：生成大纲
            print("\n🚀 步骤1: 生成主题大纲...")
            with get_usage_metadata_callback() as cb1:
                outline = outline_chain.invoke({"topic": test_topic})
                step_results["step1_outline"] = outline
                step_tokens["step1"] = dict(cb1.usage_metadata)
                
                print(f"✅ 大纲生成完成 ({len(outline.split())} 词)")
                if cb1.usage_metadata:
                    for model, usage in cb1.usage_metadata.items():
                        print(f"   Token使用 - 输入: {usage.get('input_tokens', 0)}, "
                              f"输出: {usage.get('output_tokens', 0)}, "
                              f"总计: {usage.get('total_tokens', 0)}")
            
            # 步骤2：生成内容
            print("\n🚀 步骤2: 基于大纲生成文章内容...")
            with get_usage_metadata_callback() as cb2:
                content = content_chain.invoke({"outline": outline})
                step_results["step2_content"] = content
                step_tokens["step2"] = dict(cb2.usage_metadata)
                
                print(f"✅ 文章内容生成完成 ({len(content.split())} 词)")
                if cb2.usage_metadata:
                    for model, usage in cb2.usage_metadata.items():
                        print(f"   Token使用 - 输入: {usage.get('input_tokens', 0)}, "
                              f"输出: {usage.get('output_tokens', 0)}, "
                              f"总计: {usage.get('total_tokens', 0)}")
            
            # 步骤3：优化内容
            print("\n🚀 步骤3: 优化文章内容...")
            with get_usage_metadata_callback() as cb3:
                optimized_content = optimize_chain.invoke({"content": content})
                step_results["step3_optimized"] = optimized_content
                step_tokens["step3"] = dict(cb3.usage_metadata)
                
                print(f"✅ 内容优化完成 ({len(optimized_content.split())} 词)")
                if cb3.usage_metadata:
                    for model, usage in cb3.usage_metadata.items():
                        print(f"   Token使用 - 输入: {usage.get('input_tokens', 0)}, "
                              f"输出: {usage.get('output_tokens', 0)}, "
                              f"总计: {usage.get('total_tokens', 0)}")
            
            # 汇总统计
            print("\n📊 完整Token使用分析:")
            print("=" * 50)
            
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens = 0
            
            step_names = {
                "step1": "主题扩展为大纲",
                "step2": "大纲生成文章", 
                "step3": "文章内容优化"
            }
            
            for step_id, step_name in step_names.items():
                print(f"\n📝 {step_name}:")
                if step_id in step_tokens:
                    for model, usage in step_tokens[step_id].items():
                        input_t = usage.get('input_tokens', 0)
                        output_t = usage.get('output_tokens', 0)
                        total_t = usage.get('total_tokens', 0)
                        
                        print(f"   模型: {model}")
                        print(f"   输入tokens: {input_t}")
                        print(f"   输出tokens: {output_t}")
                        print(f"   步骤总计: {total_t}")
                        
                        total_input_tokens += input_t
                        total_output_tokens += output_t
                        total_tokens += total_t
                else:
                    print("   无token使用数据")
            
            print(f"\n🎯 全流程汇总:")
            print(f"   总输入tokens: {total_input_tokens}")
            print(f"   总输出tokens: {total_output_tokens}")
            print(f"   流程总计tokens: {total_tokens}")
            
            # 计算效率指标
            if total_input_tokens > 0:
                efficiency_ratio = total_output_tokens / total_input_tokens
                print(f"   输出/输入比率: {efficiency_ratio:.2f}")
            
            # 内容统计
            print(f"\n📄 内容统计:")
            print(f"   原始主题: {test_topic}")
            print(f"   大纲字数: {len(step_results['step1_outline'].split())} 词")
            print(f"   文章字数: {len(step_results['step2_content'].split())} 词")
            print(f"   优化后字数: {len(step_results['step3_optimized'].split())} 词")
            
            # 验证数据完整性
            self.assertIn("step1_outline", step_results)
            self.assertIn("step2_content", step_results)
            self.assertIn("step3_optimized", step_results)
            self.assertGreater(total_tokens, 0, "应该有token使用记录")
            
            print("\n✅ 分步实时Token追踪测试通过!")
            
            # 返回完整的分析结果
            return {
                "topic": test_topic,
                "step_results": step_results,
                "step_tokens": step_tokens,
                "summary": {
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": total_tokens,
                    "efficiency_ratio": total_output_tokens / total_input_tokens if total_input_tokens > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"❌ 分步实时Token追踪测试失败: {e}")
            raise

    def test_async_batch_applications(self) -> None:
        """
        测试异步批处理应用场景
        
        输入: 无
        输出: 无
        """
        print("\n=== 测试异步批处理应用 ===")
        
        async def run_async_test():
            try:
                # 简单的翻译链
                translation_prompt = ChatPromptTemplate.from_template(
                    "请将以下中文翻译成英文：{text}"
                )
                
                translation_chain = translation_prompt | self.model | self.str_parser
                
                # 批量翻译任务
                texts_to_translate = [
                    {"text": "你好，世界"},
                    {"text": "人工智能技术"},
                    {"text": "机器学习算法"},
                    {"text": "自然语言处理"},
                    {"text": "深度学习模型"}
                ]
                
                # 异步批处理
                results = await translation_chain.abatch(texts_to_translate)
                
                print("批量翻译结果:")
                for i, (original, translated) in enumerate(zip(texts_to_translate, results)):
                    print(f"{i+1}. {original['text']} → {translated}")
                
                self.assertEqual(len(results), len(texts_to_translate))
                for result in results:
                    self.assertIsInstance(result, str)
                    self.assertGreater(len(result), 0)
                
                print("✅ 异步批处理应用测试通过!")
                
            except Exception as e:
                print(f"❌ 异步批处理应用测试失败: {e}")
        
        # 运行异步测试
        asyncio.run(run_async_test())


if __name__ == "__main__":
    unittest.main() 