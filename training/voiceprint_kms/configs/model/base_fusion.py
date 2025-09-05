"""
多模态融合模型配置基类定义

BaseFusionConfig: 所有多模态融合配置的父类

作者: yuger
创建时间: 2025-09-06
"""

from abc import ABC, abstractmethod  # 抽象基类支持
from dataclasses import dataclass  # 简化配置类的定义
from typing import Dict, Literal  # 类型注解


@dataclass
class BaseFusionConfig(ABC):
    """多模态融合配置基类 - 所有融合配置的父类"""

    # 基本属性
    name: str = "start_niyar"          # 模型名称

    # 融合策略配置
    fusion_method: Literal[
        "attention",        # 注意力融合
        "concat",          # 简单拼接
        "weighted",        # 加权融合
    ] = "attention"
    fusion_output_dim: int = 256  # 融合后输出维度,影响后续网络设计 (如分类器输入维度) 默认256

    # 注意力融合参数 (仅在 fusion_method="attention" 时使用)
    attention_heads: int = 8         # 注意力头数
    attention_dropout: float = 0.1    # 注意力 dropout 概率

    # 架构通用参数
    fusion_layers: int = 2          # 融合层数
    use_layer_norm: bool = True     # 是否使用归一化，提升训练稳定性
    final_dropout: float = 0.1    # 最终 dropout 概率
    activation: Literal[            # 激活函数类型
        "relu",                     # ReLU: max(0,x), 计算简单快速, 梯度不消失, 但负值完全丢失信息
        # GELU: x*Φ(x), 平滑非线性, 性能优于ReLU, 但计算开销较大 适合Transformer架构
        "gelu",
        # Swish: x*sigmoid(x), 平滑非线性, 在某些任务上优于ReLU, 计算复杂度中等 适合深层网络需要自适应特性
        "swish"
    ] = "relu"                      # 默认ReLU: 平衡性能和效率的经典选择

    # 正则化参数
    residual_connection: bool = True  # 是否使用残差连接，缓解梯度消失

    @abstractmethod
    def get_input_modalities(self) -> Dict[str, int]:
        """
        获取输入模态及其维度 - 子类必须实现

        Returns:
            Dict[str, int]: 模态名称到输入维度的映射
            例如: {"voice": 192, "text": 768}
        """
        pass

    @abstractmethod
    def get_modality_weights(self) -> Dict[str, float]:
        """
        获取各模态的权重 - 子类必须实现
        用于加权融合等方法

        Returns:
            Dict[str, float]: 模态名称到权重的映射
            例如: {"voice": 0.6, "text": 0.4}
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        验证配置参数是否合理 - 子类必须实现

        Returns:
            bool: 如果配置合理返回True，否则抛出异常
        """
        pass

    def validate_base_params(self) -> bool:
        """验证基础融合参数是否合理 - 通用验证逻辑"""
        # 检查输入维度合理性
        if self.fusion_output_dim <= 0:
            raise ValueError("fusion_output_dim 必须大于0")
        # 检查融合层数
        if self.fusion_layers <= 0:
            raise ValueError("fusion_layers 必须大于0")
        # 检查dropout范围
        if not (0 <= self.final_dropout <= 1):
            raise ValueError("final_dropout 必须在 [0,1] 之间")
        # 检查注意力参数(仅当使用注意力融合时)
        if self.fusion_method == "attention":
            if self.attention_heads <= 0:
                raise ValueError("attention_heads 必须大于0")
            if not (0 <= self.attention_dropout <= 1):
                raise ValueError("attention_dropout 必须在 [0,1] 之间")

        return True

    def get_fusion_info(self) -> dict:
        """获取融合模型基本信息"""
        return {
            "name": self.name,                  # 模型名称
            "fusion_method": self.fusion_method,    # 融合方法
            "fusion_output_dim": self.fusion_output_dim,    # 融合后输出维度
            "input_modalities": self.get_input_modalities(),  # 输入模态及其维度
            "modality_weights": self.get_modality_weights(),  # 各模态权重
            "architecture": {                   # 架构相关参数
                "fusion_layers": self.fusion_layers,
                "use_layer_norm": self.use_layer_norm,
                "residual_connection": self.residual_connection,
                "activation": self.activation,
                "final_dropout": self.final_dropout,
            }
        }
