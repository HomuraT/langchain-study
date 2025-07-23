#!/usr/bin/env python3
"""
LangChain Callbacks 测试运行器

全自动化的回调系统测试运行器，支持：
- 批量测试执行
- 详细测试报告
- 命令行接口
- 错误分析

作者: LangChain Study Project
创建时间: 2025年
"""

import unittest
import sys
import time
import argparse
from typing import Dict, List, Optional
from io import StringIO


class CallbacksTestRunner:
    """回调测试运行器类"""
    
    def __init__(self) -> None:
        """
        初始化测试运行器
        
        输入: 无
        输出: 无
        """
        self.test_modules = {
            'handlers': 'unitests.test_callbacks.test_callback_handlers',
            'inheritance': 'unitests.test_callbacks.test_callback_inheritance'
        }
        
        self.test_descriptions = {
            'handlers': '回调处理器测试 (所有回调事件的触发时机和参数)',
            'inheritance': '回调继承测试 (运行时回调vs构造函数回调的传播机制)'
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
        print(f"🔔 运行测试模块: {module_name}")
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
        
        print(f"\n🔔 开始运行LangChain回调测试套件")
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
    
    def _print_module_stats(self, stats: Dict[str, any]) -> None:
        """
        打印单个模块的统计信息
        
        输入:
            stats: Dict[str, any] - 模块统计信息
        输出: 无
        """
        print(f"\n📊 模块 [{stats['module']}] 统计:")
        print(f"  ✅ 成功: {stats['success']}")
        print(f"  ❌ 失败: {stats['failures']}")
        print(f"  🚫 错误: {stats['errors']}")
        print(f"  ⏭️  跳过: {stats['skipped']}")
        print(f"  📈 成功率: {stats['success_rate']:.1f}%")
        print(f"  ⏱️  耗时: {stats['time']:.2f}秒")
        
        if stats['success_rate'] == 100.0:
            print(f"  🎉 模块 {stats['module']} 全部测试通过!")
        elif stats['success_rate'] >= 80.0:
            print(f"  ⚠️  模块 {stats['module']} 大部分测试通过")
        else:
            print(f"  🔥 模块 {stats['module']} 需要关注")
    
    def _calculate_overall_stats(self, all_results: List[Dict], total_time: float) -> Dict[str, any]:
        """
        计算总体统计信息
        
        输入:
            all_results: List[Dict] - 所有模块结果
            total_time: float - 总耗时
        输出:
            Dict[str, any] - 总体统计
        """
        total_tests = sum(r['total'] for r in all_results)
        total_success = sum(r['success'] for r in all_results)
        total_failures = sum(r['failures'] for r in all_results)
        total_errors = sum(r['errors'] for r in all_results)
        total_skipped = sum(r['skipped'] for r in all_results)
        
        return {
            'total_modules': len(all_results),
            'total_tests': total_tests,
            'total_success': total_success,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'overall_success_rate': (total_success / total_tests * 100) if total_tests > 0 else 0,
            'total_time': total_time,
            'modules_with_issues': len([r for r in all_results if r['failures'] > 0 or r['errors'] > 0])
        }
    
    def _print_overall_report(self, total_stats: Dict, all_results: List[Dict]) -> None:
        """
        打印总体测试报告
        
        输入:
            total_stats: Dict - 总体统计
            all_results: List[Dict] - 所有模块结果
        输出: 无
        """
        print(f"\n{'='*80}")
        print(f"🏆 LangChain回调测试套件总体报告")
        print(f"{'='*80}")
        
        print(f"📦 测试模块: {total_stats['total_modules']}")
        print(f"🧪 总测试数: {total_stats['total_tests']}")
        print(f"✅ 成功: {total_stats['total_success']}")
        print(f"❌ 失败: {total_stats['total_failures']}")
        print(f"🚫 错误: {total_stats['total_errors']}")
        print(f"⏭️  跳过: {total_stats['total_skipped']}")
        print(f"📈 总成功率: {total_stats['overall_success_rate']:.1f}%")
        print(f"⏱️  总耗时: {total_stats['total_time']:.2f}秒")
        print(f"⚠️  问题模块: {total_stats['modules_with_issues']}")
        
        # 模块详情
        print(f"\n📋 各模块表现:")
        for result in all_results:
            status = "🎉" if result['success_rate'] == 100.0 else "⚠️" if result['success_rate'] >= 80.0 else "🔥"
            print(f"  {status} {result['module']}: {result['success_rate']:.1f}% ({result['success']}/{result['total']})")
        
        # 总结
        if total_stats['overall_success_rate'] == 100.0:
            print(f"\n🎊 恭喜！所有回调测试都通过了！")
        elif total_stats['overall_success_rate'] >= 90.0:
            print(f"\n👍 很好！绝大多数测试都通过了")
        elif total_stats['overall_success_rate'] >= 70.0:
            print(f"\n👌 还不错，大部分测试通过了")
        else:
            print(f"\n🔧 需要修复一些问题")
        
        print(f"{'='*80}")
    
    def list_available_tests(self) -> None:
        """
        列出所有可用的测试模块
        
        输入: 无
        输出: 无
        """
        print("\n📝 可用的回调测试模块:")
        for module, description in self.test_descriptions.items():
            print(f"  - {module}: {description}")
        
        print(f"\n🔔 回调事件类型包括:")
        print(f"  - Chat Model事件: on_chat_model_start")
        print(f"  - LLM事件: on_llm_start, on_llm_new_token, on_llm_end, on_llm_error")
        print(f"  - Chain事件: on_chain_start, on_chain_end, on_chain_error")
        print(f"  - Tool事件: on_tool_start, on_tool_end, on_tool_error")
        print(f"  - Agent事件: on_agent_action, on_agent_finish")
        print(f"  - Retriever事件: on_retriever_start, on_retriever_end, on_retriever_error")
        print(f"  - 通用事件: on_text, on_retry")
        
        print(f"\n🎯 测试重点:")
        print(f"  - 每个回调事件的触发时机")
        print(f"  - 回调参数的完整性验证")
        print(f"  - 运行时回调vs构造函数回调的区别")
        print(f"  - 回调在chain中的传播机制")
        print(f"  - 嵌套组件的回调继承")
        print(f"  - 错误情况下的回调行为")


def main():
    """
    主函数 - 处理命令行参数和执行测试
    
    输入: 无
    输出: 无
    """
    parser = argparse.ArgumentParser(
        description="LangChain回调系统测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_all_tests.py                        # 运行所有测试
  python run_all_tests.py --tests handlers       # 只运行回调处理器测试
  python run_all_tests.py --tests inheritance    # 只运行回调继承测试
  python run_all_tests.py --quiet                # 静默模式
  python run_all_tests.py --list                 # 列出所有测试
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
    
    runner = CallbacksTestRunner()
    
    if args.list:
        runner.list_available_tests()
        return 0
    
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