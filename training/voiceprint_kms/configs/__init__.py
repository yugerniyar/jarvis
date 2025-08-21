# config/__init__.py

"""
VoicePrint + KWS 配置管理系统 

这个包提供了完整的配置管理功能
- 配置模式定义(Schema)
- YAML文件加载
- 配置验证
- 环境变量覆盖
"""

__version__ = "0.1.0"
__author__ = "VoicePrint KWS yuger"


# 包级别的常量
DEFAULT_CONFIG_DIR = "configs"
SUPPORTED_FORMATS = [".yaml", ".yml", ".json"]

try:
    # 基础组件
    from .schema import (
        MainConfig,
        ModelConfig,
        EcapaTdnnConfig,
        ConformerConfig,
        TrainingConfig,
        DataConfig,
        ExperimentConfig
    )

    # 功能组件
    from .loader import ConfigLoader
    from .validator import ConfigValidator
    from .manager import ConfigManager

    # 工具函数
    from .defaults import get_default_config

except ImportError as e:
    import warnings
    warnings.warn(f"配置包导入失败: {e}. 请检查依赖是否安装齐全.")
    # TODO: 降级处理之后再考虑
