"""
检索器测试运行脚本

用于运行所有检索器相关的测试，包括：
- 基础检索器功能测试
- 多查询检索器测试
- 上下文压缩检索器测试
- 性能对比测试

作者: LinRen
创建时间: 2025年
"""

import unittest
import sys
import os
import time
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# 导入测试模块
from test_basic_retrievers import TestBasicRetrievers, TestRetrieverEdgeCases
from test_multiquery_retrievers import (
    TestMultiQueryRetriever, 
    TestMultiQueryRetrieverEdgeCases,
    TestCustomOutputParser
)
from test_compression_retrievers import (
    TestContextualCompressionRetriever,
    TestCompressionEdgeCases,
    TestCustomCompressor
)
from test_search_strategies import (
    TestSearchStrategies,
    TestParameterTuning,
    TestRealWorldScenarios
)


def create_test_suite() -> unittest.TestSuite:
    """
    创建测试套件
    
    Args:
        None
        
    Returns:
        完整的测试套件
    """
    suite = unittest.TestSuite()
    
    # 添加基础检索器测试
    suite.addTest(unittest.makeSuite(TestBasicRetrievers))
    suite.addTest(unittest.makeSuite(TestRetrieverEdgeCases))
    
    # 添加多查询检索器测试
    suite.addTest(unittest.makeSuite(TestMultiQueryRetriever))
    suite.addTest(unittest.makeSuite(TestMultiQueryRetrieverEdgeCases))
    suite.addTest(unittest.makeSuite(TestCustomOutputParser))
    
    # 添加上下文压缩检索器测试
    suite.addTest(unittest.makeSuite(TestContextualCompressionRetriever))
    suite.addTest(unittest.makeSuite(TestCompressionEdgeCases))
    suite.addTest(unittest.makeSuite(TestCustomCompressor))
    
    # 添加搜索策略测试
    suite.addTest(unittest.makeSuite(TestSearchStrategies))
    suite.addTest(unittest.makeSuite(TestParameterTuning))
    suite.addTest(unittest.makeSuite(TestRealWorldScenarios))
    
    return suite


def run_basic_retriever_tests() -> None:
    """
    运行基础检索器测试
    
    Args:
        None
        
    Returns:
        None
    """
    print("=" * 80)
    print("运行基础检索器测试")
    print("=" * 80)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBasicRetrievers))
    suite.addTest(unittest.makeSuite(TestRetrieverEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_multiquery_retriever_tests() -> None:
    """
    运行多查询检索器测试
    
    Args:
        None
        
    Returns:
        None
    """
    print("=" * 80)
    print("运行多查询检索器测试")
    print("=" * 80)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMultiQueryRetriever))
    suite.addTest(unittest.makeSuite(TestMultiQueryRetrieverEdgeCases))
    suite.addTest(unittest.makeSuite(TestCustomOutputParser))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_compression_retriever_tests() -> None:
    """
    运行上下文压缩检索器测试
    
    Args:
        None
        
    Returns:
        None
    """
    print("=" * 80)
    print("运行上下文压缩检索器测试")
    print("=" * 80)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestContextualCompressionRetriever))
    suite.addTest(unittest.makeSuite(TestCompressionEdgeCases))
    suite.addTest(unittest.makeSuite(TestCustomCompressor))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_performance_comparison() -> None:
    """
    运行性能对比测试
    
    Args:
        None
        
    Returns:
        None
    """
    print("=" * 80)
    print("检索器性能对比测试")
    print("=" * 80)
    
    from test_basic_retrievers import TestBasicRetrievers
    from test_multiquery_retrievers import TestMultiQueryRetriever
    from test_compression_retrievers import TestContextualCompressionRetriever
    
    # 创建测试实例
    basic_test = TestBasicRetrievers()
    basic_test.setUp()
    
    multiquery_test = TestMultiQueryRetriever()
    multiquery_test.setUp()
    
    compression_test = TestContextualCompressionRetriever()
    compression_test.setUp()
    
    # 运行性能测试
    print("\n1. 基础检索器性能测试")
    basic_test.test_retriever_performance()
    
    print("\n2. 多查询检索器性能测试")
    multiquery_test.test_performance_comparison()
    
    print("\n3. 压缩检索器性能测试")
    compression_test.test_performance_comparison()
    
    print("\n性能对比完成")


def run_search_strategy_tests() -> bool:
    """
    运行搜索策略测试
    
    Args:
        None
        
    Returns:
        测试是否成功
    """
    print("=" * 80)
    print("运行搜索策略测试")
    print("=" * 80)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSearchStrategies))
    suite.addTest(unittest.makeSuite(TestParameterTuning))
    suite.addTest(unittest.makeSuite(TestRealWorldScenarios))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_selected_tests(test_categories: List[str]) -> Dict[str, bool]:
    """
    运行选定的测试类别
    
    Args:
        test_categories: 测试类别列表
        
    Returns:
        各类别测试结果
    """
    results = {}
    
    for category in test_categories:
        if category == "basic":
            results["basic"] = run_basic_retriever_tests()
        elif category == "multiquery":
            results["multiquery"] = run_multiquery_retriever_tests()
        elif category == "compression":
            results["compression"] = run_compression_retriever_tests()
        elif category == "strategies":
            results["strategies"] = run_search_strategy_tests()
        elif category == "performance":
            run_performance_comparison()
            results["performance"] = True
        else:
            print(f"未知的测试类别: {category}")
            results[category] = False
    
    return results


def run_all_tests() -> bool:
    """
    运行所有检索器测试
    
    Args:
        None
        
    Returns:
        所有测试是否成功
    """
    print("=" * 80)
    print("LangChain 检索器功能全面测试")
    print("=" * 80)
    
    start_time = time.time()
    
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 打印测试摘要
    print("\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)
    print(f"运行时间: {total_time:.2f} 秒")
    print(f"总测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")
    
    return result.wasSuccessful()


def main():
    """
    主函数，处理命令行参数
    
    Args:
        None
        
    Returns:
        None
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain检索器测试运行器")
    parser.add_argument(
        "--category",
        choices=["basic", "multiquery", "compression", "strategies", "performance", "all"],
        default="all",
        help="选择要运行的测试类别"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    if args.category == "all":
        success = run_all_tests()
    elif args.category == "performance":
        run_performance_comparison()
        success = True
    else:
        results = run_selected_tests([args.category])
        success = all(results.values())
    
    # 退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 