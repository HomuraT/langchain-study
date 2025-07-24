"""
嵌入模型测试运行脚本

用于运行所有嵌入模型相关的测试，包括：
- 基础嵌入功能测试
- 缓存嵌入功能测试
- 性能对比测试

作者: LinRen
创建时间: 2025年
"""

import unittest
import sys
import os
from typing import List

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# 导入测试模块
from test_basic_embeddings import TestBasicEmbeddings, TestEmbeddingModels
from test_cached_embeddings import TestCachedEmbeddings, TestEmbeddingPerformance


def create_test_suite() -> unittest.TestSuite:
    """
    创建测试套件
    
    Args:
        None
        
    Returns:
        完整的测试套件
    """
    suite = unittest.TestSuite()
    
    # 添加基础嵌入测试
    suite.addTest(unittest.makeSuite(TestBasicEmbeddings))
    suite.addTest(unittest.makeSuite(TestEmbeddingModels))
    
    # 添加缓存嵌入测试
    suite.addTest(unittest.makeSuite(TestCachedEmbeddings))
    suite.addTest(unittest.makeSuite(TestEmbeddingPerformance))
    
    return suite


def run_specific_test(test_class_name: str = None, test_method_name: str = None) -> None:
    """
    运行特定的测试类或测试方法
    
    Args:
        test_class_name: 测试类名
        test_method_name: 测试方法名
        
    Returns:
        None
    """
    if test_class_name and test_method_name:
        # 运行特定测试方法
        suite = unittest.TestSuite()
        test_class = globals()[test_class_name]
        suite.addTest(test_class(test_method_name))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    elif test_class_name:
        # 运行特定测试类
        suite = unittest.TestSuite()
        test_class = globals()[test_class_name]
        suite.addTest(unittest.makeSuite(test_class))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        print("请指定测试类名或测试方法名")


def run_basic_tests() -> None:
    """
    只运行基础嵌入测试
    
    Args:
        None
        
    Returns:
        None
    """
    print("=== 运行基础嵌入测试 ===")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBasicEmbeddings))
    suite.addTest(unittest.makeSuite(TestEmbeddingModels))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n测试结果: 运行 {result.testsRun} 个测试")
    if result.failures:
        print(f"失败: {len(result.failures)} 个")
    if result.errors:
        print(f"错误: {len(result.errors)} 个")


def run_cache_tests() -> None:
    """
    只运行缓存嵌入测试
    
    Args:
        None
        
    Returns:
        None
    """
    print("=== 运行缓存嵌入测试 ===")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCachedEmbeddings))
    suite.addTest(unittest.makeSuite(TestEmbeddingPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n测试结果: 运行 {result.testsRun} 个测试")
    if result.failures:
        print(f"失败: {len(result.failures)} 个")
    if result.errors:
        print(f"错误: {len(result.errors)} 个")


def run_all_tests() -> None:
    """
    运行所有嵌入测试
    
    Args:
        None
        
    Returns:
        None
    """
    print("=== 运行所有嵌入模型测试 ===")
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n最终测试结果: 运行 {result.testsRun} 个测试")
    if result.failures:
        print(f"失败: {len(result.failures)} 个")
        for test, traceback in result.failures:
            print(f"  失败测试: {test}")
    if result.errors:
        print(f"错误: {len(result.errors)} 个")
        for test, traceback in result.errors:
            print(f"  错误测试: {test}")
    
    if not result.failures and not result.errors:
        print("所有测试通过! ✅")
    else:
        print("部分测试失败 ❌")


def main() -> None:
    """
    主函数，根据命令行参数运行不同的测试
    
    Args:
        None
        
    Returns:
        None
    """
    if len(sys.argv) == 1:
        # 没有参数，运行所有测试
        run_all_tests()
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == "basic":
            run_basic_tests()
        elif arg == "cache":
            run_cache_tests()
        elif arg == "all":
            run_all_tests()
        else:
            # 尝试作为测试类名运行
            try:
                run_specific_test(arg)
            except KeyError:
                print(f"未知的测试类: {arg}")
                print_usage()
    elif len(sys.argv) == 3:
        # 运行特定测试方法
        test_class = sys.argv[1]
        test_method = sys.argv[2]
        try:
            run_specific_test(test_class, test_method)
        except KeyError:
            print(f"未找到测试类: {test_class}")
            print_usage()
    else:
        print_usage()


def print_usage() -> None:
    """
    打印使用说明
    
    Args:
        None
        
    Returns:
        None
    """
    print("\n使用说明:")
    print("python run_tests.py                    # 运行所有测试")
    print("python run_tests.py all               # 运行所有测试")
    print("python run_tests.py basic             # 只运行基础嵌入测试")
    print("python run_tests.py cache             # 只运行缓存嵌入测试")
    print("python run_tests.py TestBasicEmbeddings                           # 运行特定测试类")
    print("python run_tests.py TestBasicEmbeddings test_embed_documents_basic # 运行特定测试方法")
    print("\n可用的测试类:")
    print("- TestBasicEmbeddings: 基础嵌入功能测试")
    print("- TestEmbeddingModels: 嵌入模型对比测试")
    print("- TestCachedEmbeddings: 缓存嵌入功能测试")
    print("- TestEmbeddingPerformance: 嵌入性能测试")


if __name__ == '__main__':
    main() 