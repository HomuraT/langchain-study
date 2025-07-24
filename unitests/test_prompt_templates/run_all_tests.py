#!/usr/bin/env python3
"""
提示模板测试运行器

全自动化的提示模板功能测试运行器，支持：
- 批量测试执行
- 详细测试报告
- 命令行接口
- 错误分析

作者: AI Assistant
创建时间: 2025年
"""

import unittest
import sys
import time
import argparse
from typing import Dict, List, Optional
from io import StringIO


class PromptTemplateTestRunner:
    """提示模板测试运行器类"""
    
    def __init__(self) -> None:
        """
        初始化测试运行器
        
        输入: 无
        输出: 无
        """
        self.test_modules = {
            'prompt_templates': 'unitests.test_prompt_templates.test_prompt_templates',
            'jinja2_templates': 'unitests.test_prompt_templates.test_jinja2_templates',
            'example_selectors': 'unitests.test_prompt_templates.test_example_selectors'
        }
        
        self.test_descriptions = {
            'prompt_templates': '提示模板功能测试 (PromptTemplate, ChatPromptTemplate, MessagesPlaceholder)',
            'jinja2_templates': 'Jinja2模板功能测试 (Jinja2PromptTemplate, 条件逻辑, 循环, 过滤器, 宏)',
            'example_selectors': '示例选择器功能测试 (LengthBased, SemanticSimilarity, NGram, MMR, Custom)'
        }
    
    def run_specific_tests(self, test_names: List[str], verbose: bool = True, quiet: bool = False) -> Dict[str, bool]:
        """
        运行指定的测试模块
        
        输入:
            test_names: 要运行的测试名称列表
            verbose: 是否显示详细输出
            quiet: 是否静默模式
        输出:
            Dict[str, bool]: 测试名称到成功状态的映射
        """
        results = {}
        
        if not quiet:
            print(f"\n🚀 开始运行提示模板测试")
            print("=" * 60)
        
        for test_name in test_names:
            if test_name not in self.test_modules:
                print(f"❌ 未知的测试模块: {test_name}")
                results[test_name] = False
                continue
            
            if not quiet:
                print(f"\n📋 运行测试: {self.test_descriptions[test_name]}")
                print("-" * 50)
            
            try:
                # 加载测试模块
                import os
                # 确保当前目录在Python路径中
                current_dir = os.getcwd()
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                
                # 捕获输出
                if quiet:
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = StringIO()
                    sys.stderr = StringIO()
                
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromName(self.test_modules[test_name])
                
                # 运行测试
                runner = unittest.TextTestRunner(
                    verbosity=2 if verbose and not quiet else 0,
                    stream=sys.stdout if not quiet else StringIO()
                )
                
                start_time = time.time()
                result = runner.run(suite)
                end_time = time.time()
                
                success = result.wasSuccessful()
                results[test_name] = success
                
                if not quiet:
                    status = "✅ 通过" if success else "❌ 失败"
                    print(f"\n{status} - 耗时: {end_time - start_time:.2f}秒")
                    print(f"运行测试: {result.testsRun}")
                    print(f"失败: {len(result.failures)}")
                    print(f"错误: {len(result.errors)}")
                    
                    if result.failures:
                        print("\n失败详情:")
                        for test, traceback in result.failures:
                            print(f"  - {test}: {traceback}")
                    
                    if result.errors:
                        print("\n错误详情:")
                        for test, traceback in result.errors:
                            print(f"  - {test}: {traceback}")
                
            except Exception as e:
                results[test_name] = False
                if not quiet:
                    print(f"❌ 测试执行失败: {e}")
            
            finally:
                if quiet:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
        
        return results
    
    def run_all_tests(self, verbose: bool = True, quiet: bool = False) -> Dict[str, bool]:
        """
        运行所有测试
        
        输入:
            verbose: 是否显示详细输出
            quiet: 是否静默模式
        输出:
            Dict[str, bool]: 测试结果
        """
        return self.run_specific_tests(list(self.test_modules.keys()), verbose, quiet)
    
    def print_summary(self, results: Dict[str, bool]) -> None:
        """
        打印测试摘要
        
        输入:
            results: 测试结果字典
        输出: 无
        """
        total_tests = len(results)
        passed_tests = sum(1 for success in results.values() if success)
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 60)
        print(f"📊 测试摘要")
        print("=" * 60)
        print(f"总测试模块: {total_tests}")
        print(f"✅ 通过: {passed_tests}")
        print(f"❌ 失败: {failed_tests}")
        print(f"成功率: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
        
        if failed_tests > 0:
            print(f"\n❌ 失败的测试:")
            for test_name, success in results.items():
                if not success:
                    print(f"  - {test_name}: {self.test_descriptions[test_name]}")
        
        print("=" * 60)
    
    def list_available_tests(self) -> None:
        """
        列出所有可用的测试
        
        输入: 无
        输出: 无
        """
        print("📋 可用的提示模板测试:")
        print("=" * 60)
        for test_name, description in self.test_descriptions.items():
            print(f"  {test_name:20} - {description}")
        print("=" * 60)


def main() -> int:
    """
    主函数 - 命令行接口
    
    输入: 无
    输出: 退出码
    """
    parser = argparse.ArgumentParser(description="提示模板测试运行器")
    parser.add_argument(
        "--tests",
        nargs="+",
        help="要运行的测试名称 (可选: prompt_templates)",
        default=None
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用测试"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只显示摘要"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="关闭详细输出"
    )
    
    args = parser.parse_args()
    
    runner = PromptTemplateTestRunner()
    
    if args.list:
        runner.list_available_tests()
        return 0
    
    # 确定要运行的测试
    if args.tests:
        # 验证测试名称
        valid_tests = []
        for test in args.tests:
            if test in runner.test_modules:
                valid_tests.append(test)
            else:
                print(f"❌ 未知的测试: {test}")
                print("可用测试:", list(runner.test_modules.keys()))
                return 1
        
        if not valid_tests:
            print("❌ 没有有效的测试可运行")
            return 1
        
        results = runner.run_specific_tests(
            valid_tests, 
            verbose=not args.no_verbose,
            quiet=args.quiet
        )
    else:
        # 运行所有测试
        results = runner.run_all_tests(
            verbose=not args.no_verbose,
            quiet=args.quiet
        )
    
    # 打印摘要
    if not args.quiet:
        runner.print_summary(results)
    
    # 返回适当的退出码
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit(main()) 