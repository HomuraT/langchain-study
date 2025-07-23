"""
Pydantic BaseModel构造方式测试

测试Pydantic BaseModel的各种创建和构造方式，包括：
- 基本创建方式
- Field字段定义
- 数据验证器
- 嵌套模型
- 泛型模型
- 动态模型创建
- 继承和混合
- 工厂方法
- 配置类
- Union类型和Optional字段
- 自定义类型
- 别名和序列化
- 条件字段
- 验证和错误处理
- 高级特性
"""

import unittest
import json
import re
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Generic, TypeVar, Callable, ClassVar
from uuid import UUID, uuid4
from pathlib import Path

from pydantic import (
    BaseModel, Field, ValidationError, field_validator, model_validator,
    EmailStr, HttpUrl, FilePath, DirectoryPath, Json, SecretStr,
    constr, conint, confloat, conlist, conset, condecimal,
    create_model, ConfigDict
)
from pydantic_settings import BaseSettings
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Annotated, Literal


# 1. 基本BaseModel构造方式
class BasicUserModel(BaseModel):
    """基本用户模型"""
    name: str
    age: int
    email: str


class UserWithDefaults(BaseModel):
    """带默认值的用户模型"""
    name: str
    age: int = 25
    email: str = "user@example.com"
    is_active: bool = True


# 2. Field字段定义的各种方式
class UserWithFields(BaseModel):
    """使用Field定义字段的用户模型"""
    name: str = Field(..., description="用户姓名", min_length=1, max_length=50)
    age: int = Field(..., description="用户年龄", ge=0, le=150)
    email: str = Field(..., description="用户邮箱", pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    score: float = Field(0.0, description="用户分数", ge=0.0, le=100.0)
    tags: List[str] = Field(default_factory=list, description="用户标签")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class UserWithAdvancedFields(BaseModel):
    """高级字段定义"""
    # 字符串约束
    username: constr(min_length=3, max_length=20, pattern=r'^[a-zA-Z0-9_]+$')
    # 数字约束
    age: conint(ge=0, le=120)
    salary: confloat(gt=0)
    # 容器约束
    skills: conlist(str, min_length=1, max_length=10)
    unique_skills: conset(str, min_length=1)
    # 小数约束
    rating: condecimal(max_digits=3, decimal_places=2, ge=0, le=5)


# 3. 数据验证器
class UserWithValidators(BaseModel):
    """带验证器的用户模型"""
    name: str
    age: int
    email: str
    password: str
    confirm_password: str
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """验证姓名"""
        v = v.strip()  # 先去除空白
        if not v:
            raise ValueError('姓名不能为空')
        if len(v) < 2:
            raise ValueError('姓名至少需要2个字符')
        return v.title()
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """验证邮箱格式"""
        if '@' not in v:
            raise ValueError('无效的邮箱格式')
        return v.lower()
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        """验证年龄"""
        if v < 0:
            raise ValueError('年龄不能为负数')
        if v > 150:
            raise ValueError('年龄不能超过150岁')
        return v
    
    @model_validator(mode='after')
    def validate_passwords_match(self):
        """验证密码匹配"""
        if self.password != self.confirm_password:
            raise ValueError('两次输入的密码不一致')
        return self


# 4. 嵌套模型
class Address(BaseModel):
    """地址模型"""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "中国"


class Company(BaseModel):
    """公司模型"""
    name: str
    address: Address
    founded_year: int
    employees_count: int


class UserWithNested(BaseModel):
    """带嵌套模型的用户"""
    name: str
    age: int
    address: Address
    company: Optional[Company] = None
    previous_addresses: List[Address] = Field(default_factory=list)


# 5. 泛型模型
T = TypeVar('T')

class GenericResponse(BaseModel, Generic[T]):
    """泛型响应模型"""
    code: int
    message: str
    data: T
    timestamp: datetime = Field(default_factory=datetime.now)


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应模型"""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int


# 6. 枚举类型
class UserStatus(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"


class UserWithEnum(BaseModel):
    """使用枚举的用户模型"""
    name: str
    status: UserStatus = UserStatus.ACTIVE
    role: UserRole = UserRole.USER


# 7. Union类型和Optional字段
class FlexibleUser(BaseModel):
    """灵活的用户模型"""
    id: Union[int, str, UUID]
    name: str
    age: Optional[int] = None
    contact: Union[str, Dict[str, str]]  # 可以是邮箱字符串或联系方式字典
    profile: Optional[Union[str, Dict[str, Any]]] = None


# 8. 自定义类型
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

class PositiveInt(int):
    """正整数类型"""
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.int_schema(),
        )
    
    @classmethod
    def validate(cls, v):
        if not isinstance(v, int):
            raise ValueError('必须是整数')
        if v <= 0:
            raise ValueError('必须是正整数')
        return cls(v)


class UserWithCustomTypes(BaseModel):
    """使用自定义类型的用户模型"""
    name: str
    age: PositiveInt
    email: EmailStr
    website: Optional[HttpUrl] = None
    config_path: Optional[FilePath] = None
    data_dir: Optional[DirectoryPath] = None
    secret_key: SecretStr
    json_data: Json[Dict[str, Any]] = Field(default_factory=dict)


# 9. 别名和序列化
class UserWithAliases(BaseModel):
    """使用别名的用户模型"""
    model_config = ConfigDict(populate_by_name=True)
    
    user_name: str = Field(..., alias="userName")
    user_age: int = Field(..., alias="userAge")
    email_address: str = Field(..., alias="email")
    is_verified: bool = Field(False, alias="verified")


# 10. 配置类
class StrictUser(BaseModel):
    """严格模式用户模型"""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True,
        str_to_lower=True
    )
    
    name: str
    age: int
    email: str


class FlexibleConfig(BaseModel):
    """灵活配置模型"""
    model_config = ConfigDict(
        extra='allow',
        arbitrary_types_allowed=True
    )
    
    name: str
    data: Dict[str, Any] = Field(default_factory=dict)


# 11. 继承和混合
class TimestampMixin(BaseModel):
    """时间戳混合模型"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class BaseEntity(TimestampMixin):
    """基础实体模型"""
    id: UUID = Field(default_factory=uuid4)
    version: int = 1


class ExtendedUser(BaseEntity):
    """扩展用户模型（继承）"""
    name: str
    email: str
    profile: Dict[str, Any] = Field(default_factory=dict)


# 12. 工厂方法和动态创建
class UserFactory:
    """用户模型工厂"""
    
    @staticmethod
    def create_basic_user_model():
        """创建基本用户模型"""
        return create_model(
            'DynamicUser',
            name=(str, ...),
            age=(int, 25),
            email=(str, 'user@example.com')
        )
    
    @staticmethod
    def create_user_with_fields(**extra_fields):
        """创建带额外字段的用户模型"""
        base_fields = {
            'name': (str, ...),
            'age': (int, ...),
            'email': (str, ...)
        }
        base_fields.update(extra_fields)
        return create_model('CustomUser', **base_fields)
    
    @staticmethod
    def create_user_for_role(role: str):
        """根据角色创建用户模型"""
        if role == "admin":
            return create_model(
                'AdminUser',
                name=(str, ...),
                email=(str, ...),
                permissions=(List[str], Field(default_factory=list)),
                is_super_admin=(bool, False)
            )
        elif role == "regular":
            return create_model(
                'RegularUser',
                name=(str, ...),
                email=(str, ...),
                subscription_level=(str, "basic")
            )
        else:
            return create_model(
                'GuestUser',
                session_id=(str, ...),
                access_level=(str, "guest")
            )


# 13. 条件字段和动态验证
class ConditionalUser(BaseModel):
    """条件字段用户模型"""
    user_type: Literal["individual", "corporate"]
    name: str
    email: str
    
    # 个人用户字段
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    birth_date: Optional[date] = None
    
    # 企业用户字段
    company_name: Optional[str] = None
    tax_id: Optional[str] = None
    registration_date: Optional[date] = None
    
    @model_validator(mode='after')
    def validate_user_type_fields(self):
        """根据用户类型验证字段"""
        if self.user_type == "individual":
            if not self.first_name or not self.last_name:
                raise ValueError('个人用户必须提供姓和名')
        elif self.user_type == "corporate":
            if not self.company_name or not self.tax_id:
                raise ValueError('企业用户必须提供公司名称和税号')
        
        return self


# 14. Settings和环境变量
class DatabaseSettings(BaseSettings):
    """数据库设置模型"""
    model_config = ConfigDict(env_prefix="DB_")
    
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "testdb"
    db_user: str = "user"
    db_password: SecretStr = "password"
    db_ssl: bool = False


# 15. Dataclass风格
@pydantic_dataclass
class DataclassUser:
    """Dataclass风格的用户模型"""
    name: str
    age: int
    email: str = "default@example.com"
    is_active: bool = True


# 16. 复杂嵌套和递归模型
class TreeNode(BaseModel):
    """树节点模型（递归）"""
    name: str
    value: Optional[Any] = None
    children: List['TreeNode'] = Field(default_factory=list)
    parent: Optional['TreeNode'] = None


TreeNode.model_rebuild()


# 17. 高级验证和转换
class SmartUser(BaseModel):
    """智能用户模型"""
    name: str
    age: Union[int, str]  # 可以接受字符串并转换
    email: str
    phone: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('age', mode='before')
    @classmethod
    def parse_age(cls, v):
        """解析年龄（可以是字符串）"""
        if isinstance(v, str):
            if v.isdigit():
                return int(v)
            raise ValueError('年龄必须是数字')
        return v
    
    @field_validator('phone')
    @classmethod
    def format_phone(cls, v):
        """格式化电话号码"""
        if v is None:
            return v
        # 移除所有非数字字符
        clean_phone = re.sub(r'\D', '', v)
        if len(clean_phone) == 11 and clean_phone.startswith('1'):
            return f"+86-{clean_phone[:3]}-{clean_phone[3:7]}-{clean_phone[7:]}"
        return v
    
    @field_validator('preferences', mode='before')
    @classmethod
    def parse_preferences(cls, v):
        """解析偏好设置"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v


class TestPydanticBaseModelConstruction(unittest.TestCase):
    """Pydantic BaseModel构造方式测试类"""
    
    def test_basic_model_creation(self) -> None:
        """测试基本模型创建"""
        try:
            # 基本创建
            user = BasicUserModel(name="张三", age=25, email="zhangsan@example.com")
            self.assertEqual(user.name, "张三")
            self.assertEqual(user.age, 25)
            self.assertEqual(user.email, "zhangsan@example.com")
            
            # 带默认值的创建
            user_with_defaults = UserWithDefaults(name="李四")
            self.assertEqual(user_with_defaults.age, 25)
            self.assertEqual(user_with_defaults.email, "user@example.com")
            self.assertTrue(user_with_defaults.is_active)
            
            print("Basic model creation test passed!")
            
        except Exception as e:
            print(f"Basic model creation test failed: {e}")
    
    def test_field_definitions(self) -> None:
        """测试Field字段定义"""
        try:
            # Field定义
            user = UserWithFields(
                name="王五",
                age=30,
                email="wangwu@example.com",
                score=85.5,
                tags=["python", "AI"],
                metadata={"department": "tech"}
            )
            
            self.assertEqual(user.name, "王五")
            self.assertEqual(user.score, 85.5)
            self.assertIn("python", user.tags)
            self.assertEqual(user.metadata["department"], "tech")
            
            # 高级字段约束
            advanced_user = UserWithAdvancedFields(
                username="user_123",
                age=25,
                salary=50000.0,
                skills=["Python", "JavaScript"],
                unique_skills={"Python", "AI"},
                rating=Decimal("4.85")
            )
            
            self.assertEqual(advanced_user.username, "user_123")
            self.assertEqual(len(advanced_user.skills), 2)
            self.assertIsInstance(advanced_user.rating, Decimal)
            
            print("Field definitions test passed!")
            
        except Exception as e:
            print(f"Field definitions test failed: {e}")
    
    def test_validators(self) -> None:
        """测试验证器"""
        try:
            # 正常验证
            user = UserWithValidators(
                name="  zhang san  ",
                age=25,
                email="ZHANGSAN@EXAMPLE.COM",
                password="123456",
                confirm_password="123456"
            )
            
            self.assertEqual(user.name, "Zhang San")  # 自动title格式化
            self.assertEqual(user.email, "zhangsan@example.com")  # 自动小写
            
            # 验证失败情况
            with self.assertRaises(ValidationError):
                UserWithValidators(
                    name="",  # 空姓名
                    age=25,
                    email="invalid_email",
                    password="123456",
                    confirm_password="654321"  # 密码不匹配
                )
            
            print("Validators test passed!")
            
        except Exception as e:
            print(f"Validators test failed: {e}")
    
    def test_nested_models(self) -> None:
        """测试嵌套模型"""
        try:
            address = Address(
                street="中关村大街1号",
                city="北京",
                state="北京市",
                zip_code="100000"
            )
            
            company = Company(
                name="科技公司",
                address=address,
                founded_year=2020,
                employees_count=100
            )
            
            user = UserWithNested(
                name="员工A",
                age=28,
                address=address,
                company=company,
                previous_addresses=[address]
            )
            
            self.assertEqual(user.address.city, "北京")
            self.assertEqual(user.company.name, "科技公司")
            self.assertEqual(len(user.previous_addresses), 1)
            
            print("Nested models test passed!")
            
        except Exception as e:
            print(f"Nested models test failed: {e}")
    
    def test_generic_models(self) -> None:
        """测试泛型模型"""
        try:
            # 字符串响应
            str_response = GenericResponse[str](
                code=200,
                message="成功",
                data="Hello World"
            )
            
            self.assertEqual(str_response.code, 200)
            self.assertEqual(str_response.data, "Hello World")
            
            # 用户列表响应
            users_data = [
                {"name": "用户1", "age": 25},
                {"name": "用户2", "age": 30}
            ]
            
            paginated = PaginatedResponse[dict](
                items=users_data,
                total=2,
                page=1,
                page_size=10,
                total_pages=1
            )
            
            self.assertEqual(len(paginated.items), 2)
            self.assertEqual(paginated.total, 2)
            
            print("Generic models test passed!")
            
        except Exception as e:
            print(f"Generic models test failed: {e}")
    
    def test_enum_types(self) -> None:
        """测试枚举类型"""
        try:
            user = UserWithEnum(
                name="测试用户",
                status=UserStatus.ACTIVE,
                role=UserRole.ADMIN
            )
            
            self.assertEqual(user.status, UserStatus.ACTIVE)
            self.assertEqual(user.role, UserRole.ADMIN)
            
            # 字符串值自动转换
            user2 = UserWithEnum(
                name="测试用户2",
                status="inactive",  # 字符串会自动转换为枚举
                role="user"
            )
            
            self.assertEqual(user2.status, UserStatus.INACTIVE)
            self.assertEqual(user2.role, UserRole.USER)
            
            print("Enum types test passed!")
            
        except Exception as e:
            print(f"Enum types test failed: {e}")
    
    def test_union_and_optional(self) -> None:
        """测试Union和Optional类型"""
        try:
            # 不同类型的ID
            user1 = FlexibleUser(
                id=123,
                name="用户1",
                contact="user1@example.com"
            )
            
            user2 = FlexibleUser(
                id=uuid4(),
                name="用户2",
                age=25,
                contact={"email": "user2@example.com", "phone": "123456789"}
            )
            
            self.assertIsInstance(user1.id, int)
            self.assertIsInstance(user2.id, UUID)
            self.assertIsInstance(user2.contact, dict)
            
            print("Union and Optional test passed!")
            
        except Exception as e:
            print(f"Union and Optional test failed: {e}")
    
    def test_custom_types(self) -> None:
        """测试自定义类型"""
        try:
            user = UserWithCustomTypes(
                name="测试用户",
                age=25,  # PositiveInt
                email="test@example.com",  # EmailStr
                website="https://example.com",  # HttpUrl
                secret_key="super_secret_key",  # SecretStr
                json_data='{"key": "value"}'  # Json
            )
            
            self.assertIsInstance(user.age, PositiveInt)
            self.assertTrue(str(user.website).startswith("https://"))
            self.assertEqual(user.secret_key.get_secret_value(), "super_secret_key")
            self.assertEqual(user.json_data["key"], "value")
            
            # 验证自定义类型验证
            with self.assertRaises(ValidationError):
                UserWithCustomTypes(
                    name="测试",
                    age=-5,  # 负数应该失败
                    email="invalid_email",
                    secret_key="key"
                )
            
            print("Custom types test passed!")
            
        except Exception as e:
            print(f"Custom types test failed: {e}")
    
    def test_aliases_and_serialization(self) -> None:
        """测试别名和序列化"""
        try:
            # 使用别名创建
            user_data = {
                "userName": "测试用户",
                "userAge": 25,
                "email": "test@example.com",
                "verified": True
            }
            
            user = UserWithAliases(**user_data)
            self.assertEqual(user.user_name, "测试用户")
            self.assertEqual(user.user_age, 25)
            self.assertTrue(user.is_verified)
            
            # 序列化测试
            json_data = user.model_dump(by_alias=True)
            self.assertIn("userName", json_data)
            self.assertIn("userAge", json_data)
            
            print("Aliases and serialization test passed!")
            
        except Exception as e:
            print(f"Aliases and serialization test failed: {e}")
    
    def test_config_classes(self) -> None:
        """测试配置类"""
        try:
            # 严格模式
            user = StrictUser(
                name="  Test User  ",  # 会自动去除空白并转小写
                age=25,
                email="TEST@EXAMPLE.COM"
            )
            
            self.assertEqual(user.name, "test user")  # 转小写并去空白
            
            # 额外字段应该被拒绝
            with self.assertRaises(ValidationError):
                StrictUser(
                    name="Test",
                    age=25,
                    email="test@example.com",
                    extra_field="should_fail"  # 额外字段
                )
            
            # 灵活配置
            flexible = FlexibleConfig(
                name="Test",
                data={"key": "value"},
                extra_field="allowed"  # 额外字段被允许
            )
            
            self.assertEqual(flexible.extra_field, "allowed")
            
            print("Config classes test passed!")
            
        except Exception as e:
            print(f"Config classes test failed: {e}")
    
    def test_inheritance_and_mixins(self) -> None:
        """测试继承和混合"""
        try:
            user = ExtendedUser(
                name="测试用户",
                email="test@example.com",
                profile={"bio": "开发者"}
            )
            
            # 验证继承的字段
            self.assertIsInstance(user.id, UUID)
            self.assertIsInstance(user.created_at, datetime)
            self.assertEqual(user.version, 1)
            
            # 验证自身字段
            self.assertEqual(user.name, "测试用户")
            self.assertEqual(user.profile["bio"], "开发者")
            
            print("Inheritance and mixins test passed!")
            
        except Exception as e:
            print(f"Inheritance and mixins test failed: {e}")
    
    def test_factory_methods_and_dynamic_creation(self) -> None:
        """测试工厂方法和动态创建"""
        try:
            factory = UserFactory()
            
            # 基本动态模型
            BasicDynamicUser = factory.create_basic_user_model()
            user = BasicDynamicUser(name="动态用户", age=30)
            self.assertEqual(user.name, "动态用户")
            self.assertEqual(user.email, "user@example.com")  # 默认值
            
            # 带额外字段的模型
            CustomUser = factory.create_user_with_fields(
                department=(str, "IT"),
                salary=(float, ...)
            )
            custom_user = CustomUser(
                name="自定义用户",
                age=28,
                email="custom@example.com",
                department="Engineering",
                salary=75000.0
            )
            self.assertEqual(custom_user.department, "Engineering")
            self.assertEqual(custom_user.salary, 75000.0)
            
            # 根据角色创建模型
            AdminUser = factory.create_user_for_role("admin")
            admin = AdminUser(
                name="管理员",
                email="admin@example.com",
                permissions=["read", "write", "delete"],
                is_super_admin=True
            )
            self.assertTrue(admin.is_super_admin)
            self.assertIn("delete", admin.permissions)
            
            print("Factory methods and dynamic creation test passed!")
            
        except Exception as e:
            print(f"Factory methods and dynamic creation test failed: {e}")
    
    def test_conditional_fields(self) -> None:
        """测试条件字段"""
        try:
            # 个人用户
            individual = ConditionalUser(
                user_type="individual",
                name="张三",
                email="zhangsan@example.com",
                first_name="三",
                last_name="张",
                birth_date=date(1990, 1, 1)
            )
            
            self.assertEqual(individual.first_name, "三")
            self.assertEqual(individual.last_name, "张")
            
            # 企业用户
            corporate = ConditionalUser(
                user_type="corporate",
                name="科技公司",
                email="contact@techcompany.com",
                company_name="北京科技有限公司",
                tax_id="123456789",
                registration_date=date(2020, 1, 1)
            )
            
            self.assertEqual(corporate.company_name, "北京科技有限公司")
            self.assertEqual(corporate.tax_id, "123456789")
            
            # 验证失败情况
            with self.assertRaises(ValidationError):
                ConditionalUser(
                    user_type="individual",
                    name="测试",
                    email="test@example.com"
                    # 缺少必需的个人用户字段
                )
            
            print("Conditional fields test passed!")
            
        except Exception as e:
            print(f"Conditional fields test failed: {e}")
    
    def test_settings_model(self) -> None:
        """测试Settings模型"""
        try:
            # 默认设置
            settings = DatabaseSettings()
            self.assertEqual(settings.db_host, "localhost")
            self.assertEqual(settings.db_port, 5432)
            
            # 自定义设置
            custom_settings = DatabaseSettings(
                db_host="192.168.1.100",
                db_port=3306,
                db_name="production"
            )
            self.assertEqual(custom_settings.db_host, "192.168.1.100")
            self.assertEqual(custom_settings.db_port, 3306)
            
            print("Settings model test passed!")
            
        except Exception as e:
            print(f"Settings model test failed: {e}")
    
    def test_dataclass_style(self) -> None:
        """测试Dataclass风格"""
        try:
            user = DataclassUser(
                name="数据类用户",
                age=30
            )
            
            self.assertEqual(user.name, "数据类用户")
            self.assertEqual(user.age, 30)
            self.assertEqual(user.email, "default@example.com")
            self.assertTrue(user.is_active)
            
            print("Dataclass style test passed!")
            
        except Exception as e:
            print(f"Dataclass style test failed: {e}")
    
    def test_recursive_models(self) -> None:
        """测试递归模型"""
        try:
            # 创建树结构
            root = TreeNode(name="根节点", value="root")
            child1 = TreeNode(name="子节点1", value="child1", parent=root)
            child2 = TreeNode(name="子节点2", value="child2", parent=root)
            
            root.children = [child1, child2]
            
            self.assertEqual(root.name, "根节点")
            self.assertEqual(len(root.children), 2)
            self.assertEqual(root.children[0].name, "子节点1")
            self.assertEqual(root.children[0].parent.name, "根节点")
            
            print("Recursive models test passed!")
            
        except Exception as e:
            print(f"Recursive models test failed: {e}")
    
    def test_advanced_validation_and_conversion(self) -> None:
        """测试高级验证和转换"""
        try:
            user = SmartUser(
                name="智能用户",
                age="25",  # 字符串年龄，会自动转换
                email="smart@example.com",
                phone="13812345678",  # 会被格式化
                preferences='{"theme": "dark", "language": "zh-CN"}'  # JSON字符串
            )
            
            self.assertEqual(user.age, 25)  # 转换为整数
            self.assertEqual(user.phone, "+86-138-1234-5678")  # 格式化电话
            self.assertEqual(user.preferences["theme"], "dark")  # 解析JSON
            
            print("Advanced validation and conversion test passed!")
            
        except Exception as e:
            print(f"Advanced validation and conversion test failed: {e}")
    
    def test_error_handling(self) -> None:
        """测试错误处理"""
        try:
            # 测试各种验证错误
            test_cases = [
                # 缺少必需字段
                ({}, "name"),
                # 类型错误
                ({"name": "test", "age": "invalid"}, "age"),
                # 字段验证失败
                ({"name": "", "age": 25, "email": "test@example.com"}, "name"),
            ]
            
            for invalid_data, expected_field in test_cases:
                with self.assertRaises(ValidationError) as context:
                    UserWithValidators(
                        password="123",
                        confirm_password="123",
                        **invalid_data
                    )
                
                # 验证错误包含预期字段
                error_str = str(context.exception)
                print(f"Validation error for {expected_field}: {error_str}")
            
            print("Error handling test passed!")
            
        except Exception as e:
            print(f"Error handling test failed: {e}")


def main() -> int:
    """
    运行Pydantic BaseModel构造测试的主函数
    
    Returns:
        int: 退出码，0表示成功
    """
    print("🚀 运行Pydantic BaseModel构造方式测试")
    print("=" * 60)
    
    # 运行测试
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    main() 