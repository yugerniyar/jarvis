"""
配置系统使用示例

展示在实际项目中如何选择和使用不同的配置方式
"""

from .logging import LoggingConfig
from .base import ConfigManager
import logging.config

# 方式1: 纯YAML配置（推荐用于简单项目）
import yaml


def setup_logging_with_yaml():
    """使用YAML文件配置日志系统"""

    # 直接加载YAML配置
    with open('common/config/yaml/logging.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 提取环境特定配置
    env = "development"  # 可以从环境变量获取
    env_config = config.get('environments', {}).get(env, {})

    # 合并基础配置和环境配置
    base_logging = config['logging']
    env_logging = env_config.get('logging', {})

    # 简单的字典合并
    final_config = {**base_logging, **env_logging}

    print("YAML配置方式:")
    print(f"日志级别: {final_config['levels']['root']}")
    print(f"日志目录: {final_config['directories']['base_dir']}")


# 方式2: Python配置类（推荐用于复杂项目）


def setup_logging_with_class():
    """使用Python配置类配置日志系统"""

    # 创建配置管理器
    config_manager = ConfigManager()

    # 加载YAML配置到配置类
    logging_config = config_manager.load_config(
        LoggingConfig,
        'common/config/yaml/logging.yaml'
    )

    # 配置类提供了验证和方法
    dict_config = logging_config.get_dict_config()

    print("Python类配置方式:")
    print(f"配置验证: 通过Pydantic自动验证")
    print(f"字典配置: {dict_config['version']}")
    print(f"处理器配置: {logging_config.get_handler_config('console')}")


# 方式3: 混合方式（大型项目推荐）
def setup_logging_hybrid():
    """混合使用YAML和Python类"""

    # 1. YAML存储基础配置数据
    # 2. Python类提供验证和业务逻辑
    # 3. 环境变量覆盖特定参数

    import os

    config_manager = ConfigManager()

    # 根据环境变量选择配置文件
    env = os.getenv('ENVIRONMENT', 'development')
    config_file = f'common/config/yaml/logging_{env}.yaml'

    try:
        logging_config = config_manager.load_config(LoggingConfig, config_file)
    except FileNotFoundError:
        # 回退到默认配置
        logging_config = LoggingConfig()

    # 环境变量覆盖
    if os.getenv('LOG_LEVEL'):
        logging_config.default_level = os.getenv('LOG_LEVEL')

    if os.getenv('LOG_DIR'):
        logging_config.log_dir = os.getenv('LOG_DIR')

    print("混合配置方式:")
    print(f"环境: {env}")
    print(f"日志级别: {logging_config.default_level}")
    print(f"日志目录: {logging_config.log_dir}")


# 实际项目中的选择建议
def configuration_recommendations():
    """配置方式选择建议"""

    scenarios = {
        "小型项目 (< 10个模块)": {
            "推荐": "纯YAML配置",
            "原因": "简单直观，配置量少，维护成本低",
            "示例": "个人项目、原型开发、简单脚本"
        },

        "中型项目 (10-50个模块)": {
            "推荐": "Python配置类",
            "原因": "需要配置验证，有一定复杂度，类型安全",
            "示例": "企业应用、API服务、数据处理管道"
        },

        "大型项目 (50+个模块)": {
            "推荐": "混合方式",
            "原因": "多环境部署，复杂配置逻辑，团队协作",
            "示例": "微服务架构、机器学习平台、分布式系统"
        },

        "开源项目": {
            "推荐": "Python配置类 + 文档",
            "原因": "用户友好，配置验证，减少支持成本",
            "示例": "框架、库、工具包"
        }
    }

    for scenario, details in scenarios.items():
        print(f"\n{scenario}:")
        print(f"  推荐方案: {details['推荐']}")
        print(f"  选择原因: {details['原因']}")
        print(f"  适用场景: {details['示例']}")


if __name__ == "__main__":
    print("=== 配置系统使用示例 ===\n")

    print("1. YAML配置方式")
    setup_logging_with_yaml()

    print("\n2. Python类配置方式")
    setup_logging_with_class()

    print("\n3. 混合配置方式")
    setup_logging_hybrid()

    print("\n4. 配置方式选择建议")
    configuration_recommendations()
