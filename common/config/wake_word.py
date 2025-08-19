"""
语音唤醒专用配置

专门针对语音唤醒功能的详细配置
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator

from .base import BaseConfig


class VADConfig(BaseConfig):
    """语音活动检测(VAD)配置"""

    enabled: bool = Field(default=True, description="是否启用VAD")
    mode: int = Field(default=2, description="VAD模式(0-3)")
    frame_duration_ms: int = Field(default=30, description="帧长度(毫秒)")

    # 阈值配置
    energy_threshold: float = Field(default=0.01, description="能量阈值")
    zero_crossing_threshold: float = Field(default=0.1, description="过零率阈值")

    @classmethod
    def get_config_key(cls) -> str:
        return "vad"


class WakeWordDetectionConfig(BaseConfig):
    """唤醒词检测配置"""

    # 支持的唤醒词
    wake_words: List[str] = Field(default=["xiaoming"], description="支持的唤醒词列表")
    primary_wake_word: str = Field(default="xiaoming", description="主要唤醒词")

    # 检测阈值
    detection_threshold: float = Field(default=0.5, description="检测阈值")
    confidence_threshold: float = Field(default=0.8, description="置信度阈值")

    # 多级检测
    enable_two_stage: bool = Field(default=False, description="是否启用两阶段检测")
    first_stage_threshold: float = Field(default=0.3, description="第一阶段阈值")
    second_stage_threshold: float = Field(default=0.8, description="第二阶段阈值")

    # 时间窗口
    detection_window_ms: int = Field(default=1000, description="检测窗口(毫秒)")
    sliding_window_ms: int = Field(default=100, description="滑动窗口(毫秒)")

    @classmethod
    def get_config_key(cls) -> str:
        return "wake_word_detection"

    @validator('wake_words')
    def validate_wake_words(cls, v):
        if not v:
            raise ValueError('至少需要一个唤醒词')
        return v

    @validator('primary_wake_word')
    def validate_primary_wake_word(cls, v, values):
        if 'wake_words' in values and v not in values['wake_words']:
            raise ValueError('主要唤醒词必须在唤醒词列表中')
        return v


class AudioBufferConfig(BaseConfig):
    """音频缓冲配置"""

    buffer_size_ms: int = Field(default=2000, description="缓冲区大小(毫秒)")
    chunk_size_ms: int = Field(default=100, description="音频块大小(毫秒)")
    overlap_ms: int = Field(default=50, description="重叠大小(毫秒)")

    # 缓冲策略
    buffer_strategy: str = Field(default="circular", description="缓冲策略")
    max_buffer_size: int = Field(
        default=10 * 1024 * 1024, description="最大缓冲大小(字节)")

    @classmethod
    def get_config_key(cls) -> str:
        return "audio_buffer"


class SpeakerVerificationConfig(BaseConfig):
    """说话人验证配置"""

    enabled: bool = Field(default=True, description="是否启用说话人验证")
    model_path: str = Field(default="", description="说话人识别模型路径")

    # 验证阈值
    verification_threshold: float = Field(default=0.7, description="验证阈值")
    enrollment_threshold: float = Field(default=0.8, description="注册阈值")

    # 说话人管理
    max_speakers: int = Field(default=10, description="最大说话人数")
    speaker_embedding_dim: int = Field(default=192, description="说话人嵌入维度")

    # 自适应学习
    enable_adaptation: bool = Field(default=True, description="是否启用自适应学习")
    adaptation_rate: float = Field(default=0.1, description="自适应学习率")
    min_samples_for_adaptation: int = Field(default=5, description="自适应最小样本数")

    @classmethod
    def get_config_key(cls) -> str:
        return "speaker_verification"


class FalseWakeupSuppressionConfig(BaseConfig):
    """误唤醒抑制配置"""

    enabled: bool = Field(default=True, description="是否启用误唤醒抑制")

    # 抑制策略
    context_window_ms: int = Field(default=3000, description="上下文窗口(毫秒)")
    min_silence_before_ms: int = Field(default=500, description="唤醒前最小静音(毫秒)")
    min_silence_after_ms: int = Field(default=200, description="唤醒后最小静音(毫秒)")

    # 频率限制
    min_interval_between_wakeups_ms: int = Field(
        default=2000, description="唤醒间最小间隔(毫秒)")
    max_wakeups_per_minute: int = Field(default=10, description="每分钟最大唤醒次数")

    # 自学习
    enable_negative_learning: bool = Field(
        default=True, description="是否启用负样本学习")
    false_positive_penalty: float = Field(default=0.1, description="误唤醒惩罚")

    @classmethod
    def get_config_key(cls) -> str:
        return "false_wakeup_suppression"


class PerformanceConfig(BaseConfig):
    """性能配置"""

    # 实时性要求
    max_latency_ms: int = Field(default=500, description="最大延迟(毫秒)")
    target_cpu_usage: float = Field(default=0.5, description="目标CPU使用率")

    # 并发处理
    max_concurrent_streams: int = Field(default=4, description="最大并发流数")
    thread_pool_size: int = Field(default=4, description="线程池大小")

    # 资源限制
    max_memory_usage_mb: int = Field(default=512, description="最大内存使用(MB)")
    enable_gpu: bool = Field(default=False, description="是否启用GPU")
    gpu_memory_fraction: float = Field(default=0.3, description="GPU内存占比")

    @classmethod
    def get_config_key(cls) -> str:
        return "performance"


class LoggingConfig(BaseConfig):
    """日志配置"""

    enabled: bool = Field(default=True, description="是否启用日志")
    level: str = Field(default="INFO", description="日志级别")

    # 日志输出
    log_to_file: bool = Field(default=True, description="是否输出到文件")
    log_to_console: bool = Field(default=True, description="是否输出到控制台")
    log_file_path: str = Field(
        default="./logs/wake_word.log", description="日志文件路径")

    # 日志轮转
    max_file_size_mb: int = Field(default=10, description="日志文件最大大小(MB)")
    backup_count: int = Field(default=5, description="备份文件数量")

    # 特殊日志
    log_audio_events: bool = Field(default=True, description="是否记录音频事件")
    log_detection_details: bool = Field(default=False, description="是否记录检测细节")
    log_performance_metrics: bool = Field(default=True, description="是否记录性能指标")

    @classmethod
    def get_config_key(cls) -> str:
        return "logging"


class WakeWordConfig(BaseConfig):
    """语音唤醒主配置"""

    # 核心功能配置
    vad: VADConfig = Field(default_factory=VADConfig, description="语音活动检测配置")
    detection: WakeWordDetectionConfig = Field(
        default_factory=WakeWordDetectionConfig, description="唤醒词检测配置")
    audio_buffer: AudioBufferConfig = Field(
        default_factory=AudioBufferConfig, description="音频缓冲配置")
    speaker_verification: SpeakerVerificationConfig = Field(
        default_factory=SpeakerVerificationConfig, description="说话人验证配置")
    false_wakeup_suppression: FalseWakeupSuppressionConfig = Field(
        default_factory=FalseWakeupSuppressionConfig, description="误唤醒抑制配置")

    # 系统配置
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="性能配置")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="日志配置")

    # 全局开关
    enabled: bool = Field(default=True, description="是否启用语音唤醒")
    debug_mode: bool = Field(default=False, description="是否启用调试模式")

    # 模型文件路径
    model_file: str = Field(default="", description="模型文件路径")
    config_file: Optional[str] = Field(default=None, description="额外配置文件路径")

    @classmethod
    def get_config_key(cls) -> str:
        return "wake_word"

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self.enabled

    def get_detection_config(self) -> WakeWordDetectionConfig:
        """获取检测配置"""
        return self.detection

    def get_speaker_config(self) -> SpeakerVerificationConfig:
        """获取说话人配置"""
        return self.speaker_verification
