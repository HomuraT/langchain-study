#!/usr/bin/env python3
"""
LangChain Expression Language (LCEL) 测试运行器

全自动化的LCEL功能测试运行器，支持：
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


class LCELTestRunner:
    """LCEL测试运行器类"""
    
    def __init__(self) -> None:
        """
        初始化测试运行器
        
        输入: 无
        输出: 无
        """
        self.test_modules = {
            'basic': 'unitests.test_lcel.test_basic_composition',
            'syntax': 'unitests.test_lcel.test_syntax_operators', 
            'coercion': 'unitests.test_lcel.test_type_coercion',
            'async': 'unitests.test_lcel.test_async_operations',
            'streaming': 'unitests.test_lcel.test_streaming',
            'parallel': 'unitests.test_lcel.test_parallel_execution',
            'error': 'unitests.test_lcel.test_error_handling',
            'applications': 'unitests.test_lcel.test_chatopenai_applications'
        }
        
        self.test_descriptions = {
            'basic': 'LCEL基础组合功能测试 (RunnableSequence, RunnableParallel)',
            'syntax': 'LCEL语法操作符测试 (| 操作符, .pipe 方法)',
            'coercion': 'LCEL类型转换测试 (字典到RunnableParallel, 函数到RunnableLambda)',
            'async': 'LCEL异步操作测试 (ainvoke, astream, abatch)',
            'streaming': 'LCEL流式传输测试 (stream, astream)',
            'parallel': 'LCEL并行执行测试 (batch, abatch, 性能优化)',
            'error': 'LCEL错误处理测试 (错误传播, 异步错误, 错误恢复)',
            'applications': 'ChatOpenAI应用场景测试 (智能问答, 文本分析, 角色扮演, 推理链)'
        }
    
    def run_single_test(self, module_name: str, verbose: bool = True) -> Dict[str, any]:
        """
        运行单个测试模块
        
        输入:
            module_name: str - 测试模块名称
            verbose: bool - 是否显示详细输出
        输出:
            Dict[str, any] - 测试结果统计
        """
        if module_name not in self.test_modules:
            raise ValueError(f"未知的测试模块: {module_name}")
        
        print(f"\n{'='*60}")
        print(f"🧪 运行测试模块: {module_name}")
        print(f"📋 描述: {self.test_descriptions[module_name]}")
        print(f"{'='*60}")
        
        # 重定向输出以捕获测试结果
        if verbose:
            stream = sys.stdout
        else:
            stream = StringIO()
        
        # 加载并运行测试
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(self.test_modules[module_name])
        
        start_time = time.time()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2 if verbose else 1,
            buffer=True
        )
        result = runner.run(suite)
        end_time = time.time()
        
        # 计算统计信息
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        success = total_tests - failures - errors - skipped
        execution_time = end_time - start_time
        
        stats = {
            'module': module_name,
            'total': total_tests,
            'success': success,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'time': execution_time,
            'success_rate': (success / total_tests * 100) if total_tests > 0 else 0,
            'result_object': result
        }
        
        # 打印模块统计
        if verbose:
            self._print_module_stats(stats)
        
        return stats
    
    def run_all_tests(self, test_filter: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, any]:
        """
        运行所有测试或指定的测试集合
        
        输入:
            test_filter: Optional[List[str]] - 要运行的测试模块列表，None表示全部
            verbose: bool - 是否显示详细输出
        输出:
            Dict[str, any] - 总体测试结果
        """
        if test_filter is None:
            modules_to_run = list(self.test_modules.keys())
        else:
            modules_to_run = [m for m in test_filter if m in self.test_modules]
            if not modules_to_run:
                raise ValueError(f"没有找到有效的测试模块: {test_filter}")
        
        print(f"\n🚀 开始运行LCEL测试套件")
        print(f"📦 待运行模块: {', '.join(modules_to_run)}")
        print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = []
        overall_start_time = time.time()
        
        # 运行所有选定的测试模块
        for module in modules_to_run:
            try:
                result = self.run_single_test(module, verbose)
                all_results.append(result)
            except Exception as e:
                print(f"❌ 模块 {module} 运行失败: {e}")
                all_results.append({
                    'module': module,
                    'total': 0,
                    'success': 0,
                    'failures': 0,
                    'errors': 1,
                    'skipped': 0,
                    'time': 0,
                    'success_rate': 0,
                    'error_msg': str(e)
                })
        
        overall_end_time = time.time()
        
        # 计算总体统计
        total_stats = self._calculate_overall_stats(all_results, overall_end_time - overall_start_time)
        
        # 打印总体报告
        self._print_overall_report(total_stats, all_results)
        
        return total_stats
    
    def _calculate_overall_stats(self, results: List[Dict], total_time: float) -> Dict[str, any]:
        """
        计算总体统计信息
        
        输入:
            results: List[Dict] - 各模块测试结果
            total_time: float - 总执行时间
        输出:
            Dict[str, any] - 总体统计
        """
        total_tests = sum(r['total'] for r in results)
        total_success = sum(r['success'] for r in results)
        total_failures = sum(r['failures'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        total_skipped = sum(r['skipped'] for r in results)
        
        return {
            'modules_run': len(results),
            'total_tests': total_tests,
            'total_success': total_success,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'overall_success_rate': (total_success / total_tests * 100) if total_tests > 0 else 0,
            'total_time': total_time,
            'module_results': results
        }
    
    def _print_module_stats(self, stats: Dict[str, any]) -> None:
        """
        打印单个模块的统计信息
        
        输入:
            stats: Dict[str, any] - 模块统计信息
        输出: 无
        """
        print(f"\n📊 模块 '{stats['module']}' 测试结果:")
        print(f"   总测试数: {stats['total']}")
        print(f"   ✅ 成功: {stats['success']}")
        print(f"   ❌ 失败: {stats['failures']}")
        print(f"   💥 错误: {stats['errors']}")
        print(f"   ⏭️  跳过: {stats['skipped']}")
        print(f"   📈 成功率: {stats['success_rate']:.1f}%")
        print(f"   ⏱️  执行时间: {stats['time']:.2f}秒")
        
        # 如果有失败或错误，显示详细信息
        if stats['failures'] > 0 or stats['errors'] > 0:
            result = stats['result_object']
            if result.failures:
                print(f"\n❌ 失败详情:")
                for test, traceback in result.failures[:3]:  # 只显示前3个
                    print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
            
            if result.errors:
                print(f"\n💥 错误详情:")
                for test, traceback in result.errors[:3]:  # 只显示前3个
                    print(f"   - {test}: {traceback.split('Error:')[-1].strip()}")
    
    def _print_overall_report(self, total_stats: Dict[str, any], module_results: List[Dict]) -> None:
        """
        打印总体测试报告
        
        输入:
            total_stats: Dict[str, any] - 总体统计
            module_results: List[Dict] - 各模块结果
        输出: 无
        """
        print(f"\n{'='*80}")
        print(f"🎯 LCEL测试套件总体报告")
        print(f"{'='*80}")
        print(f"🏃 运行模块数: {total_stats['modules_run']}")
        print(f"🧪 总测试数: {total_stats['total_tests']}")
        print(f"✅ 总成功数: {total_stats['total_success']}")
        print(f"❌ 总失败数: {total_stats['total_failures']}")
        print(f"💥 总错误数: {total_stats['total_errors']}")
        print(f"⏭️  总跳过数: {total_stats['total_skipped']}")
        print(f"📈 总体成功率: {total_stats['overall_success_rate']:.1f}%")
        print(f"⏱️  总执行时间: {total_stats['total_time']:.2f}秒")
        
        # 各模块成功率概览
        print(f"\n📋 各模块成功率概览:")
        for result in module_results:
            status_icon = "✅" if result['success_rate'] == 100 else "⚠️" if result['success_rate'] >= 80 else "❌"
            print(f"   {status_icon} {result['module']:12} | {result['success_rate']:6.1f}% | {result['time']:6.2f}s | {result['success']:2}/{result['total']:2}")
        
        # 总体评价
        if total_stats['overall_success_rate'] == 100:
            print(f"\n🎉 恭喜！所有LCEL测试都通过了！")
        elif total_stats['overall_success_rate'] >= 90:
            print(f"\n👍 很好！大部分LCEL测试通过，仅有少量问题需要关注。")
        elif total_stats['overall_success_rate'] >= 70:
            print(f"\n⚠️  警告：有较多测试失败，建议检查LCEL实现。")
        else:
            print(f"\n❌ 严重：大量测试失败，LCEL功能可能存在重大问题。")
        
        print(f"⏰ 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
    
    def list_available_tests(self) -> None:
        """
        列出所有可用的测试模块
        
        输入: 无
        输出: 无
        """
        print("\n📋 可用的LCEL测试模块:")
        print("=" * 60)
        for key, desc in self.test_descriptions.items():
            print(f"🔹 {key:12} - {desc}")
        print("\n💡 使用示例:")
        print("   python run_all_tests.py --tests basic syntax")
        print("   python run_all_tests.py --tests all")
        print("   python run_all_tests.py --list")


def main():
    """
    主函数 - 处理命令行参数和执行测试
    
    输入: 无
    输出: 无
    """
    parser = argparse.ArgumentParser(
        description="LangChain Expression Language (LCEL) 测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_all_tests.py                    # 运行所有测试
  python run_all_tests.py --tests basic     # 只运行基础测试
  python run_all_tests.py --tests all       # 运行所有测试
  python run_all_tests.py --quiet           # 静默模式
  python run_all_tests.py --list            # 列出所有测试
        """
    )
    
    parser.add_argument(
        '--tests',
        nargs='*',
        default=['all'],
        help='要运行的测试模块 (默认: all)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，只显示摘要'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用的测试模块'
    )
    
    args = parser.parse_args()
    
    runner = LCELTestRunner()
    
    if args.list:
        runner.list_available_tests()
        return
    
    # 确定要运行的测试
    if 'all' in args.tests:
        test_modules = None  # 运行所有测试
    else:
        test_modules = args.tests
    
    verbose = not args.quiet
    
    try:
        # 运行测试
        results = runner.run_all_tests(test_modules, verbose)
        
        # 根据成功率设置退出代码
        if results['overall_success_rate'] < 100:
            sys.exit(1)  # 有失败的测试
        else:
            sys.exit(0)  # 所有测试通过
            
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 测试运行器出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 