"""
音频处理相关配置

统一管理音频处理的各种参数
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, validator

from .base import BaseConfig


class AudioIOConfig(BaseConfig):
    """音频输入输出配置"""

    # 基本音频参数
    sample_rate: int = Field(default=16000, description="采样率")
    channels: int = Field(default=1, description="声道数")
    bit_depth: int = Field(default=16, description="位深度")

    # 设备配置
    input_device: Optional[str] = Field(default=None, description="输入设备")
    output_device: Optional[str] = Field(default=None, description="输出设备")

    # 缓冲配置
    buffer_size: int = Field(default=1024, description="音频缓冲区大小")
    latency: str = Field(default="low", description="延迟模式")

    @classmethod
    def get_config_key(cls) -> str:
        return "audio_io"

    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f'采样率必须是 {valid_rates} 中的一个')
        return v

    @validator('channels')
    def validate_channels(cls, v):
        if v not in [1, 2]:
            raise ValueError('声道数必须是1(单声道)或2(立体声)')
        return v


class PreprocessingConfig(BaseConfig):
    """音频预处理配置"""

    # 标准化
    normalize: bool = Field(default=True, description="是否归一化")
    normalize_method: str = Field(default="peak", description="归一化方法")
    target_lufs: float = Field(default=-23.0, description="目标响度(LUFS)")

    # 去噪
    noise_reduction: bool = Field(default=False, description="是否降噪")
    noise_reduction_strength: float = Field(default=0.5, description="降噪强度")

    # 滤波
    high_pass_filter: bool = Field(default=True, description="是否高通滤波")
    high_pass_cutoff: float = Field(default=80.0, description="高通截止频率")
    low_pass_filter: bool = Field(default=False, description="是否低通滤波")
    low_pass_cutoff: float = Field(default=8000.0, description="低通截止频率")

    # 静音处理
    remove_silence: bool = Field(default=True, description="是否移除静音")
    silence_threshold: float = Field(default=0.01, description="静音阈值")
    min_silence_duration: float = Field(default=0.5, description="最小静音持续时间")

    # 音频增强
    enable_enhancement: bool = Field(default=False, description="是否启用音频增强")
    enhancement_method: str = Field(
        default="spectral_subtraction", description="增强方法")

    @classmethod
    def get_config_key(cls) -> str:
        return "preprocessing"


class FeatureExtractionConfig(BaseConfig):
    """特征提取配置"""

    # MFCC特征
    n_mfcc: int = Field(default=13, description="MFCC系数数量")
    n_fft: int = Field(default=2048, description="FFT窗口大小")
    hop_length: int = Field(default=512, description="帧移")
    win_length: Optional[int] = Field(default=None, description="窗口长度")
    window: str = Field(default="hann", description="窗口函数")

    # Mel频谱图
    n_mels: int = Field(default=80, description="Mel滤波器数量")
    fmin: float = Field(default=0.0, description="最低频率")
    fmax: Optional[float] = Field(default=None, description="最高频率")

    # 特征后处理
    use_delta: bool = Field(default=False, description="是否使用一阶差分")
    use_delta_delta: bool = Field(default=False, description="是否使用二阶差分")
    use_energy: bool = Field(default=True, description="是否包含能量特征")

    # 谱图特征
    stft_config: Dict[str, Any] = Field(
        default={
            "n_fft": 2048,
            "hop_length": 512,
            "window": "hann"
        },
        description="STFT配置"
    )

    @classmethod
    def get_config_key(cls) -> str:
        return "feature_extraction"

    @validator('n_mfcc')
    def validate_n_mfcc(cls, v):
        if v <= 0 or v > 40:
            raise ValueError('MFCC系数数量应在1-40之间')
        return v


class PostprocessingConfig(BaseConfig):
    """音频后处理配置"""

    # 音量控制
    auto_gain_control: bool = Field(default=True, description="是否自动增益控制")
    target_volume: float = Field(default=0.7, description="目标音量")

    # 动态范围压缩
    compression: bool = Field(default=False, description="是否压缩")
    compression_ratio: float = Field(default=4.0, description="压缩比")
    threshold_db: float = Field(default=-20.0, description="压缩阈值(dB)")

    # 均衡器
    enable_eq: bool = Field(default=False, description="是否启用均衡器")
    eq_bands: List[Tuple[float, float]] = Field(
        default=[(100, 0), (1000, 0), (10000, 0)],
        description="均衡器频段[(频率, 增益)]"
    )

    # 限幅器
    limiter: bool = Field(default=True, description="是否启用限幅器")
    limiter_threshold: float = Field(default=0.95, description="限幅阈值")

    @classmethod
    def get_config_key(cls) -> str:
        return "postprocessing"


class RealTimeConfig(BaseConfig):
    """实时音频处理配置"""

    # 实时参数
    chunk_size: int = Field(default=1024, description="音频块大小")
    overlap: int = Field(default=512, description="重叠采样数")
    max_latency_ms: int = Field(default=100, description="最大延迟(毫秒)")

    # 缓冲管理
    input_buffer_size: int = Field(default=4096, description="输入缓冲区大小")
    output_buffer_size: int = Field(default=4096, description="输出缓冲区大小")

    # 流处理
    stream_chunk_duration: float = Field(default=0.1, description="流块持续时间(秒)")
    processing_queue_size: int = Field(default=10, description="处理队列大小")

    @classmethod
    def get_config_key(cls) -> str:
        return "realtime"


class QualityConfig(BaseConfig):
    """音频质量配置"""

    # 质量检测
    enable_quality_check: bool = Field(default=True, description="是否启用质量检测")
    min_snr_db: float = Field(default=10.0, description="最小信噪比(dB)")
    max_distortion: float = Field(default=0.1, description="最大失真")

    # 质量增强
    enable_enhancement: bool = Field(default=False, description="是否启用质量增强")
    enhancement_algorithm: str = Field(default="wiener", description="增强算法")

    # 监控指标
    monitor_clipping: bool = Field(default=True, description="是否监控削波")
    monitor_noise_level: bool = Field(default=True, description="是否监控噪音水平")
    monitor_frequency_response: bool = Field(
        default=False, description="是否监控频响")

    @classmethod
    def get_config_key(cls) -> str:
        return "quality"


class AudioConfig(BaseConfig):
    """音频处理主配置"""

    # 子配置模块
    io: AudioIOConfig = Field(
        default_factory=AudioIOConfig, description="音频IO配置")
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig, description="预处理配置")
    feature_extraction: FeatureExtractionConfig = Field(
        default_factory=FeatureExtractionConfig, description="特征提取配置")
    postprocessing: PostprocessingConfig = Field(
        default_factory=PostprocessingConfig, description="后处理配置")
    realtime: RealTimeConfig = Field(
        default_factory=RealTimeConfig, description="实时处理配置")
    quality: QualityConfig = Field(
        default_factory=QualityConfig, description="质量配置")

    # 全局音频设置
    enable_debug_output: bool = Field(default=False, description="是否启用调试输出")
    temp_dir: str = Field(default="./temp/audio", description="临时文件目录")
    max_audio_length: float = Field(default=30.0, description="最大音频长度(秒)")

    @classmethod
    def get_config_key(cls) -> str:
        return "audio"

    def get_sample_rate(self) -> int:
        """获取采样率"""
        return self.io.sample_rate

    def get_channels(self) -> int:
        """获取声道数"""
        return self.io.channels

    def get_feature_config(self) -> FeatureExtractionConfig:
        """获取特征提取配置"""
        return self.feature_extraction
