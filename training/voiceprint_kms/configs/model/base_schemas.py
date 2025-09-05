"""
BaseModelConfig: 基础模型配置类

作者: yuger
创建时间: 2025-09-05
"""

from abc import ABC, abstractmethod  # 抽象基类支持
from dataclasses import dataclass  # 简化配置类的定义
from typing import List  # 类型注解


@dataclass
class BaseModelConfig(ABC):
    """基础模型配置类 - 所有模型配置的父类"""

    # 基本属性
    name: str            # 模型名称
    input_dim: int = 80
    output_dim: int = 512

    # 通用参数
    dropout: float = 0.2  # Dropout率

    # 构造函数验证配置参数是否合理
    @abstractmethod
    def validate(self) -> bool:
        """验证配置参数是否合理 - 子类必须实现"""
        pass

    def get_model_info(self) -> dict:
        """获取模型基本信息"""
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
        }
