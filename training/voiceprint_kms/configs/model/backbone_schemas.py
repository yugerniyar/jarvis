"""
骨干网络结构模式
ECAPA-TDNN  声纹识别模型配置
Conformer   关键词检测模型配置

作者: yuger
创建时间: 2025-09-06
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List

from .base_schemas import BaseModelConfig


@dataclass
class EcapaTdnnConfig(BaseModelConfig):
    """ECAPA-TDNN声纹识别模型配置"""

    # 模型名称
    name: str = "niyar_ecapa_tdnn"

    # 网络结构参数
    channels: List[int] = field(default_factory=lambda: [
                                512, 512, 512, 512, 1536])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])

    # 池化和输出参数
    pooling_type: str = "statistical"  # 池化类型：statistical, temporal, self-attention
    embedding_dim: int = 192    # 最终声纹嵌入维度

    # 验证方法
    def validate(self) -> bool:
        # 检查列表长度是否一致
        if len(self.channels) != len(self.kernel_sizes) != len(self.dilations):
            raise ValueError("Channels, kernel sizes, and dilations 长度必须相同")

        # 检查dropout范围
        if not (0 <= self.dropout <= 1):
            raise ValueError("Dropout 必须在 [0,1] 之间")

        if self.embedding_dim <= 0:
            raise ValueError("Embedding dim 必须大于0")

        return True


@dataclass
class ConformerConfig(BaseModelConfig):
    """
    Conformer关键词检测模型配置 

    用于多模态系统中的关键词检测组件
    """

    # 模型名称
    name: str = "niyar_conformer"

    # 基础架构参数
    num_layers: int = 6                    # Conformer层数
    encoder_dim: int = 512                  # 编码器维度

    # 注意力机制参数
    num_heads: int = 8                     # 注意力头数
    attention_dropout: float = 0.1         # 注意力Dropout率

    # 前馈网络参数
    ff_dim: int = 2048                    # 前馈网络维度
    ff_dropout: float = 0.1                # 前馈网络Dropout率
    ff_activation: str = "relu"            # 激活函数

    # 卷积模块参数
    conv_kernel_size: int = 31             # 卷积核大小
    conv_dropout: float = 0.1              # 卷积Dropout率

    # 关键词检测参数
    num_keywords: int = 1                 # 关键词数量 后续训练会进行扩展
    keyword_threshold: float = 0.5         # 关键词检测阈值

    def validate(self) -> bool:
        """验证配置参数是否合理"""
        if not (0 < self.num_layers <= 24):
            raise ValueError("num_layers 必须在 1 到 24 之间")
        if not (0 < self.encoder_dim <= 2048):
            raise ValueError("encoder_dim 必须在 1 到 2048 之间")
        if not (0 < self.num_heads <= 16):
            raise ValueError("num_heads 必须在 1 到 16 之间")
        if not (0 < self.ff_dim <= 8192):
            raise ValueError("ff_dim 必须在 1 到 8192 之间")
        if not (0 < self.conv_kernel_size <= 31):
            raise ValueError("conv_kernel_size 必须在 1 到 31 之间")
        if not (0 <= self.keyword_threshold <= 1):
            raise ValueError("keyword_threshold 必须在 [0,1] 之间")

        return True
