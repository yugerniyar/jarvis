"""
训练相关配置

统一管理训练过程中的所有配置参数
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from .base import BaseConfig


class OptimizerConfig(BaseConfig):
    """优化器配置"""

    type: str = Field(default="adam", description="优化器类型")
    lr: float = Field(default=0.001, description="学习率")
    weight_decay: float = Field(default=1e-4, description="权重衰减")
    momentum: float = Field(default=0.9, description="动量系数(SGD)")
    betas: tuple = Field(default=(0.9, 0.999), description="Adam beta参数")
    eps: float = Field(default=1e-8, description="数值稳定性参数")

    @classmethod
    def get_config_key(cls) -> str:
        return "optimizer"


class SchedulerConfig(BaseConfig):
    """学习率调度器配置"""

    type: str = Field(default="step", description="调度器类型")
    step_size: int = Field(default=30, description="StepLR步长")
    gamma: float = Field(default=0.1, description="学习率衰减系数")
    T_max: int = Field(default=100, description="CosineAnnealingLR最大周期")
    patience: int = Field(default=10, description="ReduceLROnPlateau耐心值")
    factor: float = Field(default=0.5, description="ReduceLROnPlateau衰减因子")

    @classmethod
    def get_config_key(cls) -> str:
        return "scheduler"


class ModelConfig(BaseConfig):
    """模型配置"""

    type: str = Field(default="cnn", description="模型类型")
    num_classes: int = Field(default=2, description="分类数量")
    input_channels: int = Field(default=1, description="输入通道数")
    dropout_rate: float = Field(default=0.3, description="Dropout率")
    hidden_size: int = Field(default=256, description="隐藏层大小")
    num_layers: int = Field(default=2, description="层数")
    bidirectional: bool = Field(default=True, description="是否双向(RNN)")
    d_model: int = Field(default=256, description="Transformer模型维度")
    nhead: int = Field(default=8, description="注意力头数")
    num_encoder_layers: int = Field(default=6, description="编码器层数")

    @classmethod
    def get_config_key(cls) -> str:
        return "model"

    @validator('dropout_rate')
    def validate_dropout_rate(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('dropout_rate必须在0-1之间')
        return v


class DataConfig(BaseConfig):
    """数据配置"""

    train_dir: str = Field(..., description="训练数据目录")
    val_dir: str = Field(..., description="验证数据目录")
    test_dir: Optional[str] = Field(default=None, description="测试数据目录")
    dataset_type: str = Field(default="mfcc", description="数据集类型")
    batch_size: int = Field(default=32, description="批大小")
    num_workers: int = Field(default=4, description="数据加载器工作进程数")
    shuffle: bool = Field(default=True, description="是否打乱数据")
    pin_memory: bool = Field(default=True, description="是否使用固定内存")

    # 数据集参数
    wake_word: str = Field(default="xiaoming", description="唤醒词")
    sample_rate: int = Field(default=16000, description="采样率")
    duration: float = Field(default=1.0, description="音频长度(秒)")
    n_mfcc: int = Field(default=13, description="MFCC特征数")

    @classmethod
    def get_config_key(cls) -> str:
        return "data"

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError('batch_size必须大于0')
        return v


class AugmentationConfig(BaseConfig):
    """数据增强配置"""

    enabled: bool = Field(default=False, description="是否启用数据增强")
    noise_prob: float = Field(default=0.3, description="添加噪音概率")
    noise_factor: float = Field(default=0.1, description="噪音强度")
    time_shift_prob: float = Field(default=0.3, description="时间偏移概率")
    time_shift_max: float = Field(default=0.1, description="最大时间偏移(秒)")
    pitch_shift_prob: float = Field(default=0.2, description="音调变换概率")
    pitch_shift_semitones: int = Field(default=2, description="音调变换半音数")
    speed_change_prob: float = Field(default=0.2, description="速度变换概率")
    speed_factor_range: tuple = Field(default=(0.8, 1.2), description="速度变换范围")

    @classmethod
    def get_config_key(cls) -> str:
        return "augmentation"


class TrainingConfig(BaseConfig):
    """训练主配置"""

    # 基本训练参数
    epochs: int = Field(default=100, description="训练轮数")
    early_stopping: bool = Field(default=True, description="是否早停")
    early_stopping_patience: int = Field(default=15, description="早停耐心值")
    save_best_only: bool = Field(default=True, description="是否只保存最佳模型")

    # 损失函数
    criterion: str = Field(default="cross_entropy", description="损失函数类型")
    label_smoothing: float = Field(default=0.0, description="标签平滑")

    # 输出目录
    output_dir: str = Field(default="./outputs", description="输出目录")
    checkpoint_dir: str = Field(
        default="./outputs/checkpoints", description="检查点目录")
    log_dir: str = Field(default="./outputs/logs", description="日志目录")

    # 日志和保存频率
    log_interval: int = Field(default=100, description="日志记录间隔")
    save_interval: int = Field(default=10, description="模型保存间隔(epochs)")
    eval_interval: int = Field(default=1, description="验证间隔(epochs)")

    # 设备和分布式
    device: str = Field(default="auto", description="训练设备")
    mixed_precision: bool = Field(default=False, description="是否使用混合精度")
    gradient_clip_norm: Optional[float] = Field(
        default=None, description="梯度裁剪阈值")

    # 恢复训练
    resume_from_checkpoint: Optional[str] = Field(
        default=None, description="恢复训练的检查点路径")

    # 子配置
    model: ModelConfig = Field(default_factory=ModelConfig, description="模型配置")
    data: DataConfig = Field(default_factory=DataConfig, description="数据配置")
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="优化器配置")
    scheduler: Optional[SchedulerConfig] = Field(
        default=None, description="调度器配置")
    augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig, description="数据增强配置")

    @classmethod
    def get_config_key(cls) -> str:
        return "training"

    @validator('epochs')
    def validate_epochs(cls, v):
        if v <= 0:
            raise ValueError('epochs必须大于0')
        return v

    @validator('early_stopping_patience')
    def validate_patience(cls, v):
        if v <= 0:
            raise ValueError('early_stopping_patience必须大于0')
        return v

    def get_optimizer_config(self) -> OptimizerConfig:
        """获取优化器配置"""
        return self.optimizer

    def get_model_config(self) -> ModelConfig:
        """获取模型配置"""
        return self.model

    def get_data_config(self) -> DataConfig:
        """获取数据配置"""
        return self.data
