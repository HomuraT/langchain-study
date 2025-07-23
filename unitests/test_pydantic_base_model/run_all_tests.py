#!/usr/bin/env python3
"""
Pydantic BaseModel测试运行器

运行所有Pydantic BaseModel构造方式测试，并提供详细的测试报告
"""

import sys
import unittest
import time
from typing import List, Dict, Any
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from unitests.test_pydantic_base_model.test_basemodel_construction import TestPydanticBaseModelConstruction
from unitests.test_pydantic_base_model.test_advanced_construction import TestAdvancedPydanticConstruction


class PydanticTestRunner:
    """Pydantic测试运行器"""
    
    def __init__(self):
        self.test_suites = {
            'basic': TestPydanticBaseModelConstruction,
            'advanced': TestAdvancedPydanticConstruction
        }
        self.results = {}
    
    def run_basic_tests(self) -> unittest.TestResult:
        """运行基本构造方式测试"""
        print("\n" + "="*60)
        print("🔧 运行基本Pydantic BaseModel构造方式测试")
        print("="*60)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestPydanticBaseModelConstruction)
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
    
    def run_advanced_tests(self) -> unittest.TestResult:
        """运行高级构造方式测试"""
        print("\n" + "="*60)
        print("🚀 运行高级Pydantic BaseModel构造方式测试")
        print("="*60)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestAdvancedPydanticConstruction)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        self.results['advanced'] = {
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
        print("🌟 开始运行完整的Pydantic BaseModel构造方式测试套件")
        print("测试包含：基本构造方式 + 高级构造方式")
        print("预计运行时间：30-60秒")
        
        overall_start = time.time()
        
        # 运行基本测试
        basic_result = self.run_basic_tests()
        
        # 运行高级测试
        advanced_result = self.run_advanced_tests()
        
        overall_end = time.time()
        
        # 汇总结果
        summary = self._generate_summary(overall_end - overall_start)
        self._print_summary(summary)
        
        return summary
    
    def run_specific_tests(self, test_names: List[str]) -> Dict[str, Any]:
        """运行指定的测试"""
        print(f"\n🎯 运行指定测试: {', '.join(test_names)}")
        
        overall_start = time.time()
        
        for test_name in test_names:
            if test_name == 'basic':
                self.run_basic_tests()
            elif test_name == 'advanced':
                self.run_advanced_tests()
            else:
                print(f"⚠️ 未知的测试类型: {test_name}")
        
        overall_end = time.time()
        summary = self._generate_summary(overall_end - overall_start)
        self._print_summary(summary)
        
        return summary
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """生成测试摘要"""
        summary = {
            'total_duration': total_duration,
            'total_tests': 0,
            'total_failures': 0,
            'total_errors': 0,
            'overall_success_rate': 0,
            'test_categories': {}
        }
        
        for category, result_data in self.results.items():
            summary['total_tests'] += result_data['tests_run']
            summary['total_failures'] += result_data['failures']
            summary['total_errors'] += result_data['errors']
            summary['test_categories'][category] = result_data
        
        if summary['total_tests'] > 0:
            successful_tests = summary['total_tests'] - summary['total_failures'] - summary['total_errors']
            summary['overall_success_rate'] = successful_tests / summary['total_tests'] * 100
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """打印测试摘要"""
        print("\n" + "="*60)
        print("📊 测试结果摘要")
        print("="*60)
        
        print(f"⏱️  总运行时间: {summary['total_duration']:.2f}秒")
        print(f"🧪 总测试数量: {summary['total_tests']}")
        print(f"✅ 成功率: {summary['overall_success_rate']:.1f}%")
        print(f"❌ 失败数量: {summary['total_failures']}")
        print(f"💥 错误数量: {summary['total_errors']}")
        
        print("\n📋 分类详情:")
        for category, data in summary['test_categories'].items():
            status_icon = "✅" if data['failures'] == 0 and data['errors'] == 0 else "❌"
            print(f"  {status_icon} {category.upper()}:")
            print(f"     - 测试数量: {data['tests_run']}")
            print(f"     - 运行时间: {data['duration']:.2f}秒")
            print(f"     - 成功率: {data['success_rate']:.1f}%")
            if data['failures'] > 0:
                print(f"     - 失败: {data['failures']}")
            if data['errors'] > 0:
                print(f"     - 错误: {data['errors']}")
        
        print("\n" + "="*60)
        
        if summary['total_failures'] == 0 and summary['total_errors'] == 0:
            print("🎉 所有测试通过！Pydantic BaseModel构造方式测试完成。")
        else:
            print("⚠️ 存在测试失败或错误，请检查上面的详细输出。")
        
        print("="*60)
    
    def list_available_tests(self) -> None:
        """列出可用的测试"""
        print("\n📝 可用的测试类别:")
        print("- basic: 基本Pydantic BaseModel构造方式测试")
        print("  * 基本创建方式")
        print("  * Field字段定义")
        print("  * 数据验证器")
        print("  * 嵌套模型")
        print("  * 泛型模型")
        print("  * 枚举类型")
        print("  * Union和Optional类型")
        print("  * 自定义类型")
        print("  * 别名和序列化")
        print("  * 配置类")
        print("  * 继承和混合")
        print("  * 工厂方法和动态创建")
        print("  * 条件字段")
        print("  * Settings模型")
        print("  * Dataclass风格")
        print("  * 递归模型")
        print("  * 高级验证和转换")
        print("  * 错误处理")
        
        print("\n- advanced: 高级Pydantic BaseModel构造方式测试")
        print("  * 性能优化构造")
        print("  * 元编程和动态模型创建")
        print("  * 装饰器模式")
        print("  * 中间件模式")
        print("  * 异步支持")
        print("  * 复杂验证逻辑")
        print("  * 数据库集成")
        print("  * 高级序列化")
        print("  * 版本控制模型")
        print("  * 性能对比")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pydantic BaseModel构造方式测试运行器")
    parser.add_argument(
        '--tests', 
        nargs='*', 
        choices=['basic', 'advanced', 'all'],
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
    
    args = parser.parse_args()
    
    runner = PydanticTestRunner()
    
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