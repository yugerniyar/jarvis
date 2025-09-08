"""
骨干网络结构模式
ECAPA-TDNN  声纹识别模型配置
Conformer   关键词检测模型配置

作者: yuger
创建时间: 2025-09-06
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal

from .base_schemas import BaseModelConfig

"""
ECAPA-TDNN声纹识别模型配置类

作者: yuger
创建时间: 2025-09-06
第一次修改时间: 2025-09-08
"""


@dataclass
class EcapaTdnnConfig(BaseModelConfig):
    """ECAPA-TDNN声纹识别模型配置"""

    # 覆盖基类默认值
    name: str = "niyar_ecapa_tdnn"
    output_dim: int = 192  # 声纹嵌入维度
    activation: Literal['relu', 'gelu', 'swish'] = "swish"

    # ECAPA-TDNN特有参数
    channels: List[int] = field(default_factory=lambda: [
                                512, 512, 512, 512, 1536])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])

    # 池化和输出参数
    pooling_type: Literal["statistical", "temporal",
                          "self-attention"] = "statistical"  # 池化类型
    embedding_dim: int = 192    # 最终声纹嵌入维度

    # 验证方法

    def validate(self) -> bool:
        # 先验证基类参数
        self.validate_base_params()

        # 验证ECAPA特有参数
        if len(self.channels) != len(self.kernel_sizes) != len(self.dilations):
            raise ValueError("Channels, kernel sizes, dilations 长度必须相同")

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于0")

        # 检查维度一致性
        if self.output_dim != self.embedding_dim:
            raise ValueError("output_dim 应该等于 embedding_dim")

        # 检查池化类型
        valid_pooling = ["statistical", "temporal", "self-attention"]
        if self.pooling_type not in valid_pooling:
            raise ValueError(f"pooling_type 必须是 {valid_pooling} 之一")

        return True

    def get_model_info(self) -> dict:
        """获取ECAPA-TDNN完整配置信息"""
        base_info = super().get_model_info()  # 获取基类信息

        # 添加ECAPA特有信息
        ecapa_info = {
            "ecapa_specific": {
                "network_structure": {
                    "channels": self.channels,
                    "kernel_sizes": self.kernel_sizes,
                    "dilations": self.dilations,
                    "num_blocks": len(self.channels)
                },
                "pooling_config": {
                    "pooling_type": self.pooling_type,
                    "embedding_dim": self.embedding_dim
                }
            }
        }

        # 合并信息
        return {**base_info, **ecapa_info}


"""
Conformer关键词检测模型配置类
用于多模态系统中的关键词检测组件
作者: yuger
创建时间: 2025-09-06
第一次修改时间: 2025-09-08
"""


@dataclass
class ConformerConfig(BaseModelConfig):
    """
    Conformer关键词检测模型配置 

    用于多模态系统中的关键词检测组件
    """

    # 覆盖基类默认值
    name: str = "niyar_conformer"
    num_layers: int = 6               # Conformer层数
    output_dim: int = 512            # 关键词特征维度
    activation: Literal['relu', 'gelu', 'swish'] = "gelu"   # Conformer常用GELU
    layer_norm: bool = True         # Conformer通常使用层归一化

    # Conformer特有参数
    encoder_dim: int = 512                  # 编码器维度

    # 注意力机制参数
    num_heads: int = 8                     # 注意力头数
    attention_dropout: float = 0.1         # 注意力Dropout率

    # 前馈网络参数
    ff_dim: int = 2048                    # 前馈网络维度
    ff_dropout: float = 0.1                # 前馈网络Dropout率
    ff_activation: Literal["relu", "gelu", "swish"] = "relu"

    # 卷积模块参数
    conv_kernel_size: int = 31             # 卷积核大小
    conv_dropout: float = 0.1              # 卷积Dropout率

    # 关键词检测参数
    num_keywords: int = 1                 # 关键词数量 后续训练会进行扩展
    keyword_threshold: float = 0.5         # 关键词检测阈值

    def validate(self) -> bool:
        """验证Conformer配置参数"""
        # 先验证基类参数
        self.validate_base_params()

        # 验证Conformer特有参数
        if not (1 <= self.num_layers <= 24):
            raise ValueError("num_layers 必须在 1-24 之间")
        if not (1 <= self.encoder_dim <= 2048):
            raise ValueError("encoder_dim 必须在 1-2048 之间")
        if not (1 <= self.num_heads <= 16):
            raise ValueError("num_heads 必须在 1-16 之间")
        if not (1 <= self.ff_dim <= 8192):
            raise ValueError("ff_dim 必须在 1-8192 之间")
        if not (1 <= self.conv_kernel_size <= 63):
            raise ValueError("conv_kernel_size 必须在 1-63 之间")

        # 检查维度一致性
        if self.output_dim != self.encoder_dim:
            raise ValueError("output_dim 应该等于 encoder_dim")

        # 检查注意力头数与编码器维度的关系
        if self.encoder_dim % self.num_heads != 0:
            raise ValueError("encoder_dim 必须能被 num_heads 整除")

        # 检查dropout范围
        if not (0 <= self.attention_dropout <= 1):
            raise ValueError("attention_dropout 必须在 [0,1] 范围内")
        if not (0 <= self.ff_dropout <= 1):
            raise ValueError("ff_dropout 必须在 [0,1] 范围内")
        if not (0 <= self.conv_dropout <= 1):
            raise ValueError("conv_dropout 必须在 [0,1] 范围内")
        if not (0 <= self.keyword_threshold <= 1):
            raise ValueError("keyword_threshold 必须在 [0,1] 范围内")

        return True

    def get_model_info(self) -> dict:
        """获取Conformer完整配置信息"""
        base_info = super().get_model_info()  # 获取基类信息

        # 添加Conformer特有信息
        conformer_info = {
            "conformer_specific": {
                "encoder_config": {
                    "num_layers": self.num_layers,
                    "encoder_dim": self.encoder_dim,
                    "num_heads": self.num_heads,
                    "head_dim": self.encoder_dim // self.num_heads
                },
                "attention_config": {
                    "attention_dropout": self.attention_dropout
                },
                "feedforward_config": {
                    "ff_dim": self.ff_dim,
                    "ff_dropout": self.ff_dropout,
                    "ff_activation": self.ff_activation
                },
                "convolution_config": {
                    "conv_kernel_size": self.conv_kernel_size,
                    "conv_dropout": self.conv_dropout
                },
                "keyword_detection": {
                    "num_keywords": self.num_keywords,
                    "keyword_threshold": self.keyword_threshold
                }
            }
        }

        # 合并信息
        return {**base_info, **conformer_info}
