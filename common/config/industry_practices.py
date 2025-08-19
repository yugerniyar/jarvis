"""
行业真实案例：配置管理的不同做法

展示不同规模公司和项目的配置管理实践
"""

# === 案例1: 小公司/个人项目 ===


def simple_project_config():
    """
    场景：个人项目，5-10个Python文件
    做法：直接在代码中硬编码或简单字典
    """

    # 直接硬编码（最简单）
    LOGGING_CONFIG = {
        'level': 'INFO',
        'file': 'app.log',
        'format': '%(asctime)s - %(levelname)s - %(message)s'
    }

    # 或者简单的配置文件
    import json
    with open('config.json') as f:
        config = json.load(f)

    print("小项目做法：简单直接，快速开发")


# === 案例2: 中型互联网公司 ===
def medium_company_config():
    """
    场景：50-200人团队，微服务架构
    做法：配置中心 + 环境变量 + 基础验证
    """

    import os
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class ServiceConfig:
        """简单的配置类，不用Pydantic"""
        log_level: str = "INFO"
        log_file: str = "service.log"
        db_url: Optional[str] = None
        redis_url: Optional[str] = None

        @classmethod
        def from_env(cls):
            return cls(
                log_level=os.getenv('LOG_LEVEL', 'INFO'),
                log_file=os.getenv('LOG_FILE', 'service.log'),
                db_url=os.getenv('DATABASE_URL'),
                redis_url=os.getenv('REDIS_URL')
            )

    config = ServiceConfig.from_env()
    print("中型公司做法：环境变量 + 简单配置类")


# === 案例3: 大型科技公司 ===
def large_company_config():
    """
    场景：1000+人团队，复杂系统
    做法：配置中心 + 强类型 + 动态更新
    """

    # 类似我们创建的配置系统
    from typing import Any, Dict

    from pydantic import BaseSettings, Field

    class ProductionConfig(BaseSettings):
        """生产级配置系统"""

        # 服务配置
        service_name: str = Field(..., env='SERVICE_NAME')
        service_version: str = Field(default="1.0.0", env='SERVICE_VERSION')

        # 日志配置
        log_level: str = Field(default="INFO", env='LOG_LEVEL')
        log_format: str = Field(default="json", env='LOG_FORMAT')

        # 数据库配置
        database_config: Dict[str, Any] = Field(default_factory=dict)

        # 配置文件路径
        config_file: str = Field(default="", env='CONFIG_FILE')

        class Config:
            env_file = '.env'
            case_sensitive = False

    # 支持配置热更新
    config = ProductionConfig()
    print("大型公司做法：严格类型检查 + 环境变量 + 配置中心")


# === 案例4: 开源项目 ===
def open_source_config():
    """
    场景：开源项目，用户多样化
    做法：多种配置方式支持 + 详细文档
    """

    # 支持多种配置加载方式
    class FlexibleConfig:
        def __init__(self):
            self.config = {}

        def load_from_file(self, file_path: str):
            """支持JSON、YAML、TOML等"""
            if file_path.endswith('.json'):
                import json
                with open(file_path) as f:
                    self.config.update(json.load(f))
            elif file_path.endswith('.yaml'):
                import yaml
                with open(file_path) as f:
                    self.config.update(yaml.safe_load(f))

        def load_from_env(self):
            """从环境变量加载"""
            import os
            for key, value in os.environ.items():
                if key.startswith('MYAPP_'):
                    config_key = key[6:].lower()  # 去掉前缀
                    self.config[config_key] = value

        def load_from_dict(self, config_dict: dict):
            """从字典加载"""
            self.config.update(config_dict)

    print("开源项目做法：灵活支持多种配置方式")


# === 我们项目的定位 ===
def our_project_analysis():
    """
    分析我们当前项目应该使用什么配置方式
    """

    project_characteristics = {
        "项目规模": "中小型（预计20-50个模块）",
        "团队规模": "小团队（1-5人）",
        "部署环境": "多环境（开发/测试/生产）",
        "技术特点": "AI/机器学习项目，配置复杂",
        "未来规划": "可能扩展为微服务",
        "用户类型": "开发者和研究者"
    }

    recommendations = {
        "当前阶段": {
            "推荐方案": "YAML + Python配置类（我们现在的做法）",
            "原因": [
                "YAML文件便于人工编辑和版本控制",
                "Python类提供类型安全和验证",
                "支持环境特定配置",
                "为未来扩展预留空间"
            ]
        },

        "简化方案": {
            "推荐方案": "纯YAML + 简单字典",
            "原因": [
                "如果觉得Python配置类太复杂",
                "项目规模确定不会扩大",
                "团队对配置管理要求不高"
            ]
        },

        "未来扩展": {
            "推荐方案": "配置中心 + 动态更新",
            "原因": [
                "微服务架构需要统一配置管理",
                "生产环境需要不重启更新配置",
                "多服务实例需要配置同步"
            ]
        }
    }

    print("=== 我们项目配置方案分析 ===")
    print("\n项目特征:")
    for key, value in project_characteristics.items():
        print(f"  {key}: {value}")

    print("\n配置方案建议:")
    for stage, details in recommendations.items():
        print(f"\n{stage}:")
        print(f"  方案: {details['推荐方案']}")
        print("  原因:")
        for reason in details['原因']:
            print(f"    - {reason}")


if __name__ == "__main__":
    print("=== 行业配置管理实践对比 ===\n")

    print("1. 小型项目实践")
    simple_project_config()

    print("\n2. 中型公司实践")
    medium_company_config()

    print("\n3. 大型公司实践")
    large_company_config()

    print("\n4. 开源项目实践")
    open_source_config()

    print("\n5. 我们项目分析")
    our_project_analysis()
