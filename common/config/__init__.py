"""
配置管理模块

提供统一的配置管理接口，支持：
- 环境变量配置
- YAML/JSON配置文件
- 动态配置更新
- 微服务配置分离
"""

from .audio import AudioConfig
from .base import BaseConfig, ConfigManager
from .inference import InferenceConfig
from .training import TrainingConfig
from .wake_word import WakeWordConfig

__all__ = [
    "BaseConfig",
    "ConfigManager",
    "TrainingConfig",
    "InferenceConfig",
    "WakeWordConfig",
    "AudioConfig"
]
