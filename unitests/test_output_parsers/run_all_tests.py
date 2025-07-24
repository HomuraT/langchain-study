#!/usr/bin/env python3
"""
输出解析器测试运行器

运行所有输出解析器测试，并提供详细的测试报告和统计信息
"""

import sys
import unittest
import time
from typing import List, Dict, Any
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from unitests.test_output_parsers.test_basic_parsers import TestBasicOutputParsers
from unitests.test_output_parsers.test_pydantic_parsers import TestPydanticOutputParsers
from unitests.test_output_parsers.test_custom_parsers import TestCustomOutputParsers
from unitests.test_output_parsers.test_error_handling import TestOutputParserErrorHandling


class OutputParserTestRunner:
    """输出解析器测试运行器"""
    
    def __init__(self):
        self.test_suites = {
            'basic': TestBasicOutputParsers,
            'pydantic': TestPydanticOutputParsers,
            'custom': TestCustomOutputParsers,
            'error_handling': TestOutputParserErrorHandling
        }
        self.results = {}
    
    def run_basic_tests(self) -> unittest.TestResult:
        """运行基础解析器测试"""
        print("\n" + "="*60)
        print("🔧 运行基础输出解析器测试")
        print("="*60)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestBasicOutputParsers)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        self.results['basic'] = {
            'result': result,
            'duration': end_time - start_time,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        }
        
        return result
    
    def run_pydantic_tests(self) -> unittest.TestResult:
        """运行Pydantic解析器测试"""
        print("\n" + "="*60)
        print("🏗️ 运行Pydantic输出解析器测试")
        print("="*60)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestPydanticOutputParsers)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        self.results['pydantic'] = {
            'result': result,
            'duration': end_time - start_time,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        }
        
        return result
    
    def run_custom_tests(self) -> unittest.TestResult:
        """运行自定义解析器测试"""
        print("\n" + "="*60)
        print("🛠️ 运行自定义输出解析器测试")
        print("="*60)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestCustomOutputParsers)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        self.results['custom'] = {
            'result': result,
            'duration': end_time - start_time,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        }
        
        return result
    
    def run_error_handling_tests(self) -> unittest.TestResult:
        """运行错误处理测试"""
        print("\n" + "="*60)
        print("🚨 运行错误处理和高级功能测试")
        print("="*60)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestOutputParserErrorHandling)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        self.results['error_handling'] = {
            'result': result,
            'duration': end_time - start_time,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        }
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("\n" + "🎯" * 30)
        print("🎯 输出解析器测试套件 - 全面测试开始")
        print("🎯" * 30)
        
        overall_start_time = time.time()
        
        # 运行各类测试
        try:
            self.run_basic_tests()
        except Exception as e:
            print(f"⚠️ 基础解析器测试异常: {e}")
        
        try:
            self.run_pydantic_tests()
        except Exception as e:
            print(f"⚠️ Pydantic解析器测试异常: {e}")
            
        try:
            self.run_custom_tests()
        except Exception as e:
            print(f"⚠️ 自定义解析器测试异常: {e}")
            
        try:
            self.run_error_handling_tests()
        except Exception as e:
            print(f"⚠️ 错误处理测试异常: {e}")
        
        overall_end_time = time.time()
        
        # 生成测试报告
        summary = self._generate_summary(overall_end_time - overall_start_time)
        self._print_summary(summary)
        
        return summary
    
    def run_specific_tests(self, test_types: List[str]) -> Dict[str, Any]:
        """运行指定类型的测试"""
        overall_start_time = time.time()
        
        for test_type in test_types:
            if test_type == 'basic':
                self.run_basic_tests()
            elif test_type == 'pydantic':
                self.run_pydantic_tests()
            elif test_type == 'custom':
                self.run_custom_tests()
            elif test_type == 'error_handling':
                self.run_error_handling_tests()
            else:
                print(f"⚠️ 未知的测试类型: {test_type}")
        
        overall_end_time = time.time()
        summary = self._generate_summary(overall_end_time - overall_start_time)
        self._print_summary(summary)
        
        return summary
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """生成测试总结"""
        total_tests = sum(data.get('tests_run', 0) for data in self.results.values())
        total_failures = sum(data.get('failures', 0) for data in self.results.values())
        total_errors = sum(data.get('errors', 0) for data in self.results.values())
        total_success = total_tests - total_failures - total_errors
        
        return {
            'total_duration': total_duration,
            'total_tests': total_tests,
            'total_success': total_success,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'overall_success_rate': (total_success / total_tests * 100) if total_tests > 0 else 0,
            'module_results': self.results
        }
    
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """打印测试总结"""
        print("\n" + "📊" * 30)
        print("📊 输出解析器测试套件 - 测试报告")
        print("📊" * 30)
        
        print(f"\n⏱️ 总耗时: {summary['total_duration']:.2f}秒")
        print(f"🧪 总测试数: {summary['total_tests']}")
        print(f"✅ 成功: {summary['total_success']}")
        print(f"❌ 失败: {summary['total_failures']}")
        print(f"💥 错误: {summary['total_errors']}")
        print(f"📈 总成功率: {summary['overall_success_rate']:.1f}%")
        
        print("\n📋 模块详细结果:")
        print("-" * 80)
        print(f"{'模块':<20} {'测试数':<10} {'成功':<10} {'失败':<10} {'错误':<10} {'成功率':<10} {'耗时':<10}")
        print("-" * 80)
        
        module_names = {
            'basic': '基础解析器',
            'pydantic': 'Pydantic解析器', 
            'custom': '自定义解析器',
            'error_handling': '错误处理'
        }
        
        for module, data in summary['module_results'].items():
            module_name = module_names.get(module, module)
            tests_run = data.get('tests_run', 0)
            failures = data.get('failures', 0)
            errors = data.get('errors', 0)
            success = tests_run - failures - errors
            success_rate = data.get('success_rate', 0)
            duration = data.get('duration', 0)
            
            print(f"{module_name:<20} {tests_run:<10} {success:<10} {failures:<10} {errors:<10} {success_rate:<9.1f}% {duration:<9.2f}s")
        
        print("-" * 80)
        
        # 性能评估
        print("\n⚡ 性能评估:")
        avg_time_per_test = summary['total_duration'] / summary['total_tests'] if summary['total_tests'] > 0 else 0
        print(f"   平均每测试耗时: {avg_time_per_test:.3f}秒")
        
        if avg_time_per_test < 0.5:
            print("   🟢 性能优秀")
        elif avg_time_per_test < 2.0:
            print("   🟡 性能良好")
        else:
            print("   🔴 性能需要优化")
        
        # 可靠性评估
        print("\n🔒 可靠性评估:")
        if summary['overall_success_rate'] >= 95:
            print("   🟢 可靠性优秀")
        elif summary['overall_success_rate'] >= 85:
            print("   🟡 可靠性良好")
        else:
            print("   🔴 可靠性需要改进")
        
        # 测试覆盖评估
        print("\n📊 测试覆盖评估:")
        total_modules = len(self.test_suites)
        tested_modules = len([m for m in summary['module_results'] if summary['module_results'][m].get('tests_run', 0) > 0])
        coverage_rate = tested_modules / total_modules * 100
        
        print(f"   模块覆盖率: {coverage_rate:.1f}% ({tested_modules}/{total_modules})")
        
        if coverage_rate >= 90:
            print("   🟢 覆盖率优秀")
        elif coverage_rate >= 75:
            print("   🟡 覆盖率良好")
        else:
            print("   🔴 覆盖率需要提升")
        
        # 推荐建议
        print("\n💡 推荐建议:")
        if summary['total_failures'] > 0:
            print("   - 检查失败的测试用例，优化模型配置")
        if summary['total_errors'] > 0:
            print("   - 检查错误的测试用例，确认环境配置")
        if avg_time_per_test > 1.0:
            print("   - 考虑优化测试性能，减少LLM调用")
        if summary['overall_success_rate'] < 90:
            print("   - 提升测试稳定性，加强错误处理")
        
        print("\n🎉 测试完成！感谢使用输出解析器测试套件！")
    
    def list_available_tests(self) -> None:
        """列出所有可用的测试"""
        print("📋 可用的测试模块:")
        print("-" * 50)
        
        descriptions = {
            'basic': '基础输出解析器测试 - StrOutputParser, JsonOutputParser, XMLOutputParser, YAMLOutputParser',
            'pydantic': 'Pydantic输出解析器测试 - PydanticOutputParser, PydanticToolsParser, 复杂模型解析',
            'custom': '自定义输出解析器测试 - 列表解析器, 正则解析器, 条件解析器, 链式解析器',
            'error_handling': '错误处理和高级功能测试 - OutputFixingParser, RetryWithErrorOutputParser, 回退策略'
        }
        
        for test_type, test_class in self.test_suites.items():
            desc = descriptions.get(test_type, "")
            print(f"🔹 {test_type}: {test_class.__name__}")
            print(f"   {desc}")
            print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="输出解析器测试运行器")
    parser.add_argument(
        '--tests', 
        nargs='*', 
        choices=['basic', 'pydantic', 'custom', 'error_handling', 'all'],
        default=['all'],
        help="要运行的测试类型 (默认: all)"
    )
    parser.add_argument(
        '--list', 
        action='store_true',
        help="列出所有可用的测试"
    )
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help="静默模式，只显示摘要"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help="运行性能基准测试"
    )
    
    args = parser.parse_args()
    
    runner = OutputParserTestRunner()
    
    if args.list:
        runner.list_available_tests()
        return 0
    
    if args.quiet:
        # 重定向输出到静默模式
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        if 'all' in args.tests:
            summary = runner.run_all_tests()
        else:
            summary = runner.run_specific_tests(args.tests)
        
        if args.quiet:
            # 恢复输出并只显示摘要
            sys.stdout = old_stdout
            runner._print_summary(summary)
        
        if args.benchmark:
            print("\n📊 运行性能基准测试...")
            # 这里可以添加专门的性能测试
            
        # 返回适当的退出码
        if summary['total_failures'] == 0 and summary['total_errors'] == 0:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        if args.quiet:
            sys.stdout = old_stdout
        print("\n\n❌ 测试被用户中断")
        return 130
    except Exception as e:
        if args.quiet:
            sys.stdout = old_stdout
        print(f"\n\n💥 测试运行器发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 