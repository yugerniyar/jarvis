"""
BaseModelConfig: 基础模型配置类

作者: yuger
创建时间: 2025-09-05
"""

from abc import ABC, abstractmethod  # 抽象基类支持
from dataclasses import dataclass  # 简化配置类的定义
from typing import List, Literal  # 类型注解

"""
基础模型配置类 - 所有模型配置的父类

作者: yuger
创建时间: 2025-09-05
第一次修改时间: 2025-09-08
"""


@dataclass
class BaseModelConfig(ABC):
    """基础模型配置类 - 所有模型配置的父类"""

    # 基本属性
    name: str            # 模型名称
    input_dim: int = 80
    output_dim: int = 512
    hidden_dim: int = 256              # 隐藏层维度，很多模型都需要

    # 网络结构
    num_layers: int = 2        # 基础层数，子类可以覆盖
    dropout: float = 0.2  # Dropout率
    droppath_rate: float = 0.0         # DropPath率（现代模型常用）

    # 激活和偏置
    activation: Literal[            # 激活函数类型
        "relu",                      # ReLU: max(0,x), 计算简单快速,
        # GELU: x*Φ(x), 平滑非线性, 性能优于ReLU, 但计算开销较大 适合Transformer架构
        "gelu",
        # Swish: x*sigmoid(x), 平滑非线性, 在某些任务上优于ReLU, 计算复杂度中等
        "swish"
    ] = "relu"                      # 默认ReLU: 平衡性能和效率的经典选择
    use_bias: bool = True          # 是否使用偏置

    # 归一化

    batch_norm: bool = False      # 是否使用批归一化
    layer_norm: bool = False      # 是否使用层归一化
    norm_eps: float = 1e-5        # 归一化的epsilon值，防止除零

    # 权重初始化
    init_method: Literal[
        "xavier_uniform",  # Xavier均匀分布初始化，适用于Sigmoid/Tanh激活
        "xavier_normal",   # Xavier正态分布初始化
        "kaiming_uniform",  # Kaiming均匀分布初始化，适用于ReLU激活
        "kaiming_normal"   # Kaiming正态分布初始化
    ] = "xavier_uniform"  # 默认Xavier均匀分布，适用范围广
    init_gain: float = 1.0          # 初始化增益，影响权重初始值的尺度

    # 正则化
    weight_decay: float = 1e-5         # L2正则化（训练必需）
    label_smoothing: float = 0.0        # 标签平滑参数，防止过拟合

    # 计算配置
    use_amp: bool = False              # 混合精度训练（节省显存）
    gradient_checkpointing: bool = False  # 梯度检查点（大模型必需）
    compile_model: bool = False        # 是否编译模型（提升推理速度）

    # 版本和兼容性
    model_version: str = "1.0"            # 模型版本
    framework: Literal[
        "pytorch",      # PyTorch框架
        "tensorflow"    # TensorFlow框架
    ] = "pytorch"  # 默认使用PyTorch框架

    # 推理配置
    eval_mode: bool = False            # 是否为评估模式
    output_format: Literal[
        "tensor",    # 返回张量格式
        "numpy",     # 返回NumPy数组格式
        "list"       # 返回列表格式
    ] = "tensor"      # 默认返回张量格式
    return_intermediate: bool = False  # 是否返回中间层输出

    # 构造函数验证配置参数是否合理

    def validate_base_params(self) -> bool:
        """验证基类参数的合理性"""
        # 基本维度检查
        if self.input_dim <= 0:
            raise ValueError("input_dim 必须大于0")
        if self.output_dim <= 0:
            raise ValueError("output_dim 必须大于0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim 必须大于0")
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须大于0")

        # Dropout范围检查
        if not (0 <= self.dropout <= 1):
            raise ValueError("dropout 必须在 [0,1] 范围内")
        if not (0 <= self.droppath_rate <= 1):
            raise ValueError("droppath_rate 必须在 [0,1] 范围内")
        if not (0 <= self.label_smoothing <= 1):
            raise ValueError("label_smoothing 必须在 [0,1] 范围内")

        # 正则化参数检查
        if self.weight_decay < 0:
            raise ValueError("weight_decay 必须非负")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps 必须大于0")
        if self.init_gain <= 0:
            raise ValueError("init_gain 必须大于0")

        return True

    @abstractmethod
    def validate(self) -> bool:
        """验证完整配置 - 子类必须实现"""
        pass

    def get_model_info(self) -> dict:
        """获取模型完整信息"""
        return {
            "basic_info": {         # 基础信息块
                "name": self.name,
                "version": self.model_version,
                "framework": self.framework
            },
            "architecture": {       # 架构相关参数
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
            "activation_and_bias": {    # 激活函数和偏置
                "activation": self.activation,
                "use_bias": self.use_bias
            },
            "normalization": {          # 归一化配置
                "batch_norm": self.batch_norm,
                "layer_norm": self.layer_norm,
                "norm_eps": self.norm_eps
            },
            "regularization": {     # 正则化配置
                "dropout": self.dropout,
                "droppath_rate": self.droppath_rate,
                "weight_decay": self.weight_decay,
                "label_smoothing": self.label_smoothing
            },
            "initialization": {         # 权重初始化配置
                "init_method": self.init_method,
                "init_gain": self.init_gain
            },
            "computation": {            # 计算相关配置
                "use_amp": self.use_amp,
                "gradient_checkpointing": self.gradient_checkpointing,
                "compile_model": self.compile_model
            },
            "inference": {              # 推理相关配置
                "eval_mode": self.eval_mode,
                "output_format": self.output_format,
                "return_intermediate": self.return_intermediate
            }
        }
