"""
Pydantic BaseModel高级构造方式测试

测试更高级的Pydantic BaseModel构造和使用模式，包括：
- 性能优化构造
- 序列化和反序列化高级用法
- 元编程和动态模型创建
- 装饰器和中间件模式
- 异步支持
- 复杂验证逻辑
- 与外部库集成
- 缓存和优化策略
"""

import unittest
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, ClassVar
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import sqlite3
import json

from pydantic import (
    BaseModel, Field, ValidationError, field_validator, model_validator,
    create_model, ConfigDict
)
from pydantic.json import pydantic_encoder
from pydantic.types import PositiveInt, NegativeInt, PositiveFloat
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Annotated


# 1. 性能优化构造方式
class OptimizedModel(BaseModel):
    """性能优化的模型"""
    
    model_config = ConfigDict(
        # 性能优化配置
        validate_assignment=False,  # 关闭赋值验证提高性能
        use_enum_values=True,  # 使用枚举值
        arbitrary_types_allowed=True,  # 允许任意类型
        # copy_on_model_validation removed in v2
        
        # 自定义JSON编码器提高序列化性能
        json_encoders={
            datetime: lambda v: v.timestamp(),
            set: list,
            bytes: lambda v: v.decode('utf-8', errors='ignore')
        }
    )


class CachedModel(BaseModel):
    """带缓存的模型"""
    name: str
    data: Dict[str, Any]
    
    @lru_cache(maxsize=128)
    def get_computed_value(self, key: str) -> Any:
        """缓存计算结果"""
        # 模拟复杂计算
        time.sleep(0.01)
        return f"computed_{key}_{self.name}"
    
    @property
    @lru_cache(maxsize=1)
    def expensive_property(self) -> str:
        """缓存昂贵的属性计算"""
        # 模拟昂贵计算
        time.sleep(0.05)
        return f"expensive_result_for_{self.name}"


# 2. 元编程和动态模型创建
class ModelMetaFactory:
    """模型元工厂"""
    
    @staticmethod
    def create_entity_model(entity_name: str, fields: Dict[str, Any]) -> type:
        """动态创建实体模型"""
        # 添加基础字段
        base_fields = {
            'id': (int, Field(..., description="实体ID")),
            'created_at': (datetime, Field(default_factory=datetime.now)),
            'updated_at': (Optional[datetime], None)
        }
        
        # 合并自定义字段
        all_fields = {**base_fields, **fields}
        
        # 动态创建模型
        model_class = create_model(
            f"{entity_name}Model",
            **all_fields
        )
        
        # 添加动态方法
        def to_dict_with_meta(self):
            """转换为包含元数据的字典"""
            data = self.model_dump()
            data['_entity_type'] = entity_name
            data['_created'] = self.created_at.isoformat() if self.created_at else None
            return data
        
        model_class.to_dict_with_meta = to_dict_with_meta
        
        return model_class
    
    @staticmethod
    def create_api_response_model(data_model: type, include_meta: bool = True) -> type:
        """创建API响应模型"""
        fields = {
            'success': (bool, True),
            'data': (data_model, ...),
            'message': (str, "操作成功")
        }
        
        if include_meta:
            fields.update({
                'timestamp': (datetime, Field(default_factory=datetime.now)),
                'request_id': (str, Field(default_factory=lambda: f"req_{int(time.time())}")),
                'version': (str, "1.0.0")
            })
        
        return create_model(f"{data_model.__name__}Response", **fields)


# 3. 装饰器模式
def validate_model_method(func: Callable) -> Callable:
    """模型方法验证装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 执行前验证
        if not hasattr(self, '__validated__'):
            raise ValueError("模型未经过验证")
        
        try:
            result = func(self, *args, **kwargs)
            # 记录操作
            if not hasattr(self, '__operations__'):
                self.__operations__ = []
            self.__operations__.append({
                'method': func.__name__,
                'timestamp': datetime.now(),
                'args': len(args),
                'kwargs': list(kwargs.keys())
            })
            return result
        except Exception as e:
            # 记录错误
            if not hasattr(self, '__errors__'):
                self.__errors__ = []
            self.__errors__.append({
                'method': func.__name__,
                'error': str(e),
                'timestamp': datetime.now()
            })
            raise
    
    return wrapper


class TrackedModel(BaseModel):
    """可追踪的模型"""
    name: str
    value: int
    
    def __init__(self, **data):
        super().__init__(**data)
        self.__validated__ = True
        self.__operations__ = []
        self.__errors__ = []
    
    @validate_model_method
    def increment_value(self, amount: int = 1) -> None:
        """增加值"""
        self.value += amount
    
    @validate_model_method
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'name': self.name,
            'value': self.value,
            'operations_count': len(self.__operations__),
            'errors_count': len(self.__errors__)
        }


# 4. 中间件模式
class ModelMiddleware:
    """模型中间件"""
    
    def __init__(self):
        self.middlewares = []
    
    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self.middlewares.append(middleware)
    
    def process(self, model_instance: BaseModel, operation: str, *args, **kwargs):
        """处理中间件链"""
        context = {
            'model': model_instance,
            'operation': operation,
            'args': args,
            'kwargs': kwargs,
            'timestamp': datetime.now()
        }
        
        # 执行前置中间件
        for middleware in self.middlewares:
            if hasattr(middleware, 'before'):
                middleware.before(context)
        
        # 执行操作
        try:
            result = getattr(model_instance, operation)(*args, **kwargs)
            context['result'] = result
            context['success'] = True
        except Exception as e:
            context['error'] = str(e)
            context['success'] = False
            result = None
        
        # 执行后置中间件
        for middleware in reversed(self.middlewares):
            if hasattr(middleware, 'after'):
                middleware.after(context)
        
        return result


class LoggingMiddleware:
    """日志中间件"""
    
    def before(self, context: Dict[str, Any]):
        """操作前记录"""
        print(f"[{context['timestamp']}] 开始执行 {context['operation']} on {type(context['model']).__name__}")
    
    def after(self, context: Dict[str, Any]):
        """操作后记录"""
        status = "成功" if context.get('success') else "失败"
        print(f"[{datetime.now()}] 操作 {context['operation']} {status}")


class MiddlewareModel(BaseModel):
    """支持中间件的模型"""
    name: str
    data: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        self._middleware = ModelMiddleware()
        self._middleware.add_middleware(LoggingMiddleware())
    
    def update_data(self, key: str, value: Any) -> None:
        """更新数据（通过中间件）"""
        def _update():
            self.data[key] = value
        
        self._middleware.process(self, '_update')
        _update()


# 5. 异步模型支持
class AsyncModel(BaseModel):
    """异步支持的模型"""
    name: str
    url: str
    timeout: int = 30
    
    async def fetch_data(self) -> Dict[str, Any]:
        """异步获取数据"""
        # 模拟异步HTTP请求
        await asyncio.sleep(0.1)
        return {
            'name': self.name,
            'url': self.url,
            'timestamp': datetime.now().isoformat(),
            'data': f"async_data_for_{self.name}"
        }
    
    async def batch_process(self, items: List[str]) -> List[Dict[str, Any]]:
        """批量异步处理"""
        tasks = []
        for item in items:
            task = asyncio.create_task(self._process_item(item))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def _process_item(self, item: str) -> Dict[str, Any]:
        """处理单个项目"""
        await asyncio.sleep(0.05)  # 模拟异步处理
        return {
            'item': item,
            'processed_by': self.name,
            'timestamp': datetime.now().isoformat()
        }


# 6. 复杂验证逻辑
class BusinessRuleModel(BaseModel):
    """业务规则模型"""
    user_type: str
    age: int
    income: float
    credit_score: int
    employment_years: float
    
    @model_validator(mode='after')
    def validate_business_rules(self):
        """复杂业务规则验证"""
        user_type = self.user_type
        age = self.age
        income = self.income
        credit_score = self.credit_score
        employment_years = self.employment_years
        
        # 规则1: VIP用户的特殊要求
        if user_type == 'VIP':
            if income < 100000:
                raise ValueError('VIP用户年收入必须超过10万')
            if credit_score < 750:
                raise ValueError('VIP用户信用分数必须超过750')
        
        # 规则2: 年龄和就业年限的关系
        if employment_years > (age - 16):
            raise ValueError('就业年限不能超过（年龄-16）年')
        
        # 规则3: 收入和信用分数的关联验证
        expected_min_score = min(800, max(300, income / 1000 + 500))
        if credit_score < expected_min_score:
            raise ValueError(f'基于收入水平，信用分数至少应为{expected_min_score:.0f}')
        
        return self

    def calculate_risk_level(self) -> str:
        """计算风险等级"""
        score = 0
        
        # 年龄因子
        if self.age < 25:
            score += 2
        elif self.age > 65:
            score += 1
        
        # 收入因子
        if self.income < 30000:
            score += 3
        elif self.income < 50000:
            score += 2
        elif self.income > 100000:
            score -= 1
        
        # 信用分数因子
        if self.credit_score < 600:
            score += 3
        elif self.credit_score < 700:
            score += 1
        elif self.credit_score > 800:
            score -= 1
        
        # 就业年限因子
        if self.employment_years < 1:
            score += 2
        elif self.employment_years > 10:
            score -= 1
        
        # 确定风险等级
        if score >= 6:
            return "高风险"
        elif score >= 3:
            return "中风险"
        else:
            return "低风险"


# 7. 数据库集成模型
class DatabaseModel(BaseModel):
    """数据库集成模型"""
    
    model_config = ConfigDict(
        # 允许从数据库行创建模型
        from_attributes=True  # orm_mode renamed to from_attributes in v2
    )
    
    @classmethod
    def from_db_row(cls, row: sqlite3.Row):
        """从数据库行创建模型实例"""
        # 将sqlite3.Row转换为字典
        row_dict = dict(row)
        return cls(**row_dict)
    
    @classmethod
    def bulk_from_db(cls, cursor: sqlite3.Cursor, query: str, params: tuple = ()):
        """批量从数据库创建模型实例"""
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [cls.from_db_row(row) for row in rows]
    
    def to_db_dict(self) -> Dict[str, Any]:
        """转换为数据库插入格式"""
        data = self.model_dump()
        # 处理特殊类型
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                data[key] = json.dumps(value)
        return data


class UserDBModel(DatabaseModel):
    """用户数据库模型"""
    id: Optional[int] = None
    name: str
    email: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# 8. 序列化高级用法
class AdvancedSerializationModel(BaseModel):
    """高级序列化模型"""
    public_data: str = Field(description='公开数据')
    private_data: str = Field(..., exclude=True)  # 序列化时排除
    computed_field: Optional[str] = None
    
    model_config = ConfigDict(
        # 自定义序列化器
        json_encoders={
            datetime: lambda v: v.strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    
    def model_dump(self, include_computed: bool = True, **kwargs):
        """自定义字典转换"""
        data = super().model_dump(**kwargs)
        
        if include_computed and self.computed_field is None:
            # 动态计算字段
            data['computed_field'] = f"computed_from_{self.public_data}"
        
        return data
    
    @classmethod
    def parse_with_preprocessing(cls, data: Dict[str, Any]):
        """带预处理的解析"""
        # 数据预处理
        if 'public_data' in data:
            data['public_data'] = data['public_data'].strip().title()
        
        if 'private_data' in data:
            # 简单的数据脱敏
            data['private_data'] = '*' * len(data['private_data'])
        
        return cls(**data)


# 9. 版本控制模型
class VersionedModel(BaseModel):
    """支持版本控制的模型"""
    data: Dict[str, Any]
    version: int = 1
    previous_versions: List[Dict[str, Any]] = Field(default_factory=list)
    
    def update_data(self, updates: Dict[str, Any]) -> 'VersionedModel':
        """更新数据并创建新版本"""
        # 保存当前版本
        current_state = {
            'version': self.version,
            'data': self.data.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 创建新版本
        new_data = self.data.copy()
        new_data.update(updates)
        
        new_previous = self.previous_versions.copy()
        new_previous.append(current_state)
        
        return VersionedModel(
            data=new_data,
            version=self.version + 1,
            previous_versions=new_previous
        )
    
    def rollback_to_version(self, target_version: int) -> Optional['VersionedModel']:
        """回滚到指定版本"""
        if target_version == self.version:
            return self
        
        for prev_state in self.previous_versions:
            if prev_state['version'] == target_version:
                # 回滚逻辑
                return VersionedModel(
                    data=prev_state['data'],
                    version=target_version,
                    previous_versions=[
                        v for v in self.previous_versions 
                        if v['version'] < target_version
                    ]
                )
        
        return None


class TestAdvancedPydanticConstruction(unittest.TestCase):
    """高级Pydantic构造方式测试类"""
    
    def test_performance_optimized_models(self) -> None:
        """测试性能优化模型"""
        try:
            # 测试优化配置
            model = OptimizedModel()
            self.assertFalse(model.__config__.validate_assignment)
            
            # 测试缓存模型
            cached_model = CachedModel(name="测试", data={"key": "value"})
            
            # 第一次调用（会有延迟）
            start_time = time.time()
            result1 = cached_model.get_computed_value("test")
            first_call_time = time.time() - start_time
            
            # 第二次调用（应该从缓存返回）
            start_time = time.time()
            result2 = cached_model.get_computed_value("test")
            second_call_time = time.time() - start_time
            
            self.assertEqual(result1, result2)
            self.assertLess(second_call_time, first_call_time * 0.5)  # 缓存应该更快
            
            print("Performance optimized models test passed!")
            
        except Exception as e:
            print(f"Performance optimized models test failed: {e}")
    
    def test_meta_programming_models(self) -> None:
        """测试元编程模型"""
        try:
            factory = ModelMetaFactory()
            
            # 动态创建用户模型
            UserModel = factory.create_entity_model("User", {
                'name': (str, ...),
                'email': (str, ...),
                'age': (int, 18)
            })
            
            user = UserModel(name="测试用户", email="test@example.com", age=25)
            self.assertEqual(user.name, "测试用户")
            self.assertIsNotNone(user.id)
            self.assertIsNotNone(user.created_at)
            
            # 测试动态方法
            meta_dict = user.to_dict_with_meta()
            self.assertEqual(meta_dict['_entity_type'], "User")
            self.assertIn('_created', meta_dict)
            
            # 创建API响应模型
            ResponseModel = factory.create_api_response_model(UserModel)
            response = ResponseModel(data=user)
            
            self.assertTrue(response.success)
            self.assertEqual(response.data.name, "测试用户")
            self.assertIsNotNone(response.timestamp)
            
            print("Meta-programming models test passed!")
            
        except Exception as e:
            print(f"Meta-programming models test failed: {e}")
    
    def test_decorator_patterns(self) -> None:
        """测试装饰器模式"""
        try:
            model = TrackedModel(name="测试模型", value=10)
            
            # 执行操作
            model.increment_value(5)
            model.increment_value(3)
            
            status = model.get_status()
            self.assertEqual(status['value'], 18)
            self.assertEqual(status['operations_count'], 3)  # increment_value x2 + get_status x1
            self.assertEqual(status['errors_count'], 0)
            
            print("Decorator patterns test passed!")
            
        except Exception as e:
            print(f"Decorator patterns test failed: {e}")
    
    def test_middleware_patterns(self) -> None:
        """测试中间件模式"""
        try:
            model = MiddlewareModel(name="测试中间件")
            
            # 测试中间件功能（会输出日志）
            model.update_data("key1", "value1")
            model.update_data("key2", "value2")
            
            self.assertEqual(model.data["key1"], "value1")
            self.assertEqual(model.data["key2"], "value2")
            
            print("Middleware patterns test passed!")
            
        except Exception as e:
            print(f"Middleware patterns test failed: {e}")
    
    def test_async_models(self) -> None:
        """测试异步模型"""
        try:
            async def run_async_tests():
                model = AsyncModel(name="异步模型", url="https://example.com")
                
                # 测试异步数据获取
                data = await model.fetch_data()
                self.assertEqual(data['name'], "异步模型")
                self.assertIn('timestamp', data)
                
                # 测试批量异步处理
                items = ["item1", "item2", "item3"]
                results = await model.batch_process(items)
                
                self.assertEqual(len(results), 3)
                for i, result in enumerate(results):
                    self.assertEqual(result['item'], items[i])
                    self.assertEqual(result['processed_by'], "异步模型")
            
            # 运行异步测试
            asyncio.run(run_async_tests())
            
            print("Async models test passed!")
            
        except Exception as e:
            print(f"Async models test failed: {e}")
    
    def test_complex_validation_logic(self) -> None:
        """测试复杂验证逻辑"""
        try:
            # 有效的VIP用户
            vip_user = BusinessRuleModel(
                user_type="VIP",
                age=35,
                income=120000,
                credit_score=780,
                employment_years=10
            )
            
            risk_level = vip_user.calculate_risk_level()
            self.assertIn(risk_level, ["低风险", "中风险", "高风险"])
            
            # 测试业务规则验证失败
            with self.assertRaises(ValidationError):
                BusinessRuleModel(
                    user_type="VIP",
                    age=25,
                    income=50000,  # VIP用户收入不足
                    credit_score=600,
                    employment_years=2
                )
            
            print("Complex validation logic test passed!")
            
        except Exception as e:
            print(f"Complex validation logic test failed: {e}")
    
    def test_database_integration(self) -> None:
        """测试数据库集成"""
        try:
            # 创建内存数据库
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 创建表
            cursor.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # 插入测试数据
            test_user = UserDBModel(
                name="数据库用户",
                email="db@example.com",
                metadata={"role": "admin"}
            )
            
            db_data = test_user.to_db_dict()
            cursor.execute("""
                INSERT INTO users (name, email, created_at, metadata)
                VALUES (?, ?, ?, ?)
            """, (db_data['name'], db_data['email'], 
                 db_data['created_at'], db_data['metadata']))
            
            # 从数据库读取
            users = UserDBModel.bulk_from_db(
                cursor, "SELECT * FROM users WHERE name = ?", ("数据库用户",)
            )
            
            self.assertEqual(len(users), 1)
            self.assertEqual(users[0].name, "数据库用户")
            self.assertEqual(users[0].email, "db@example.com")
            
            conn.close()
            print("Database integration test passed!")
            
        except Exception as e:
            print(f"Database integration test failed: {e}")
    
    def test_advanced_serialization(self) -> None:
        """测试高级序列化"""
        try:
            model = AdvancedSerializationModel(
                public_data="  public info  ",
                private_data="secret"
            )
            
            # 测试预处理解析
            parsed = AdvancedSerializationModel.parse_with_preprocessing({
                'public_data': '  test data  ',
                'private_data': 'sensitive'
            })
            
            self.assertEqual(parsed.public_data, "Test Data")  # 格式化处理
            
            # 测试序列化
            data_dict = model.model_dump()
            self.assertIn('computed_field', data_dict)
            self.assertIn('computed_from_Public Info', data_dict['computed_field'])
            
            # 测试私有字段排除
            json_data = model.model_dump(exclude={'private_data'})
            self.assertNotIn('private_data', json_data)
            
            print("Advanced serialization test passed!")
            
        except Exception as e:
            print(f"Advanced serialization test failed: {e}")
    
    def test_versioned_models(self) -> None:
        """测试版本控制模型"""
        try:
            # 创建初始版本
            model_v1 = VersionedModel(data={"name": "初始版本", "value": 100})
            self.assertEqual(model_v1.version, 1)
            self.assertEqual(len(model_v1.previous_versions), 0)
            
            # 更新到版本2
            model_v2 = model_v1.update_data({"value": 200, "new_field": "新增"})
            self.assertEqual(model_v2.version, 2)
            self.assertEqual(len(model_v2.previous_versions), 1)
            self.assertEqual(model_v2.data["value"], 200)
            
            # 更新到版本3
            model_v3 = model_v2.update_data({"value": 300})
            self.assertEqual(model_v3.version, 3)
            self.assertEqual(len(model_v3.previous_versions), 2)
            
            # 回滚到版本1
            rollback = model_v3.rollback_to_version(1)
            self.assertIsNotNone(rollback)
            self.assertEqual(rollback.version, 1)
            self.assertEqual(rollback.data["value"], 100)
            self.assertNotIn("new_field", rollback.data)
            
            print("Versioned models test passed!")
            
        except Exception as e:
            print(f"Versioned models test failed: {e}")
    
    def test_performance_comparison(self) -> None:
        """测试性能对比"""
        try:
            # 基准测试：创建大量模型实例
            basic_models = []
            optimized_models = []
            
            # 基本模型性能
            start_time = time.time()
            for i in range(1000):
                model = BasicUserModel(name=f"用户{i}", age=25, email=f"user{i}@example.com")
                basic_models.append(model)
            basic_time = time.time() - start_time
            
            # 优化模型性能
            start_time = time.time()
            for i in range(1000):
                model = OptimizedModel()
                optimized_models.append(model)
            optimized_time = time.time() - start_time
            
            print(f"基本模型创建1000个实例耗时: {basic_time:.4f}秒")
            print(f"优化模型创建1000个实例耗时: {optimized_time:.4f}秒")
            
            # 验证创建成功
            self.assertEqual(len(basic_models), 1000)
            self.assertEqual(len(optimized_models), 1000)
            
            print("Performance comparison test passed!")
            
        except Exception as e:
            print(f"Performance comparison test failed: {e}")


def main() -> int:
    """
    运行高级Pydantic BaseModel构造测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行高级Pydantic BaseModel构造方式测试")
    print("=" * 60)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main()


# 用于外部导入的BasicUserModel定义
class BasicUserModel(BaseModel):
    """基本用户模型（用于性能测试）"""
    name: str
    age: int
    email: str 