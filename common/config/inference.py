"""
推理相关配置

管理推理服务的配置参数
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from .base import BaseConfig


class ModelInferenceConfig(BaseConfig):
    """模型推理配置"""

    model_path: str = Field(..., description="模型文件路径")
    model_type: str = Field(default="cnn", description="模型类型")
    device: str = Field(default="auto", description="推理设备")
    batch_size: int = Field(default=1, description="推理批大小")
    max_batch_size: int = Field(default=32, description="最大批大小")

    # 性能优化
    use_onnx: bool = Field(default=False, description="是否使用ONNX")
    use_tensorrt: bool = Field(default=False, description="是否使用TensorRT")
    use_quantization: bool = Field(default=False, description="是否使用量化")
    num_threads: int = Field(default=4, description="推理线程数")

    # 缓存配置
    enable_cache: bool = Field(default=True, description="是否启用缓存")
    cache_size: int = Field(default=1000, description="缓存大小")
    cache_ttl: int = Field(default=3600, description="缓存TTL(秒)")

    @classmethod
    def get_config_key(cls) -> str:
        return "model_inference"


class APIConfig(BaseConfig):
    """API服务配置"""

    host: str = Field(default="0.0.0.0", description="服务主机")
    port: int = Field(default=8000, description="服务端口")
    workers: int = Field(default=1, description="工作进程数")

    # API限制
    max_request_size: int = Field(
        default=10 * 1024 * 1024, description="最大请求大小(字节)")
    rate_limit: Optional[str] = Field(
        default="100/minute", description="请求频率限制")
    timeout: int = Field(default=30, description="请求超时时间(秒)")

    # 安全配置
    enable_cors: bool = Field(default=True, description="是否启用CORS")
    cors_origins: List[str] = Field(default=["*"], description="CORS允许的源")
    api_key_required: bool = Field(default=False, description="是否需要API密钥")
    api_keys: List[str] = Field(default=[], description="有效的API密钥列表")

    @classmethod
    def get_config_key(cls) -> str:
        return "api"

    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('端口号必须在1-65535之间')
        return v


class AudioProcessingConfig(BaseConfig):
    """音频处理配置"""

    sample_rate: int = Field(default=16000, description="采样率")
    channels: int = Field(default=1, description="音频通道数")
    bit_depth: int = Field(default=16, description="位深度")

    # 音频预处理
    normalize: bool = Field(default=True, description="是否归一化")
    remove_silence: bool = Field(default=True, description="是否移除静音")
    noise_reduction: bool = Field(default=False, description="是否降噪")

    # 特征提取
    n_mfcc: int = Field(default=13, description="MFCC特征数")
    n_fft: int = Field(default=2048, description="FFT窗口大小")
    hop_length: int = Field(default=512, description="帧移")
    win_length: Optional[int] = Field(default=None, description="窗口长度")

    @classmethod
    def get_config_key(cls) -> str:
        return "audio_processing"


class WakeWordInferenceConfig(BaseConfig):
    """语音唤醒推理配置"""

    model_path: str = Field(..., description="唤醒词模型路径")
    threshold: float = Field(default=0.5, description="唤醒阈值")
    confidence_threshold: float = Field(default=0.8, description="置信度阈值")

    # 实时检测
    chunk_duration: float = Field(default=1.0, description="音频块长度(秒)")
    overlap_duration: float = Field(default=0.2, description="重叠长度(秒)")
    max_detection_time: float = Field(default=5.0, description="最大检测时间(秒)")

    # 后处理
    enable_smoothing: bool = Field(default=True, description="是否启用平滑")
    smoothing_window: int = Field(default=5, description="平滑窗口大小")
    min_trigger_duration: float = Field(default=0.3, description="最小触发持续时间")

    @classmethod
    def get_config_key(cls) -> str:
        return "wake_word_inference"


class ASRInferenceConfig(BaseConfig):
    """语音识别推理配置"""

    model_name: str = Field(default="whisper", description="ASR模型名称")
    model_size: str = Field(default="base", description="模型大小")
    language: str = Field(default="zh", description="识别语言")

    # Whisper配置
    beam_size: int = Field(default=5, description="束搜索大小")
    best_of: int = Field(default=5, description="最佳候选数")
    temperature: float = Field(default=0.0, description="采样温度")

    # 实时识别
    chunk_length: int = Field(default=30, description="音频块长度(秒)")
    stream_chunk_s: float = Field(default=1.0, description="流式块长度(秒)")

    @classmethod
    def get_config_key(cls) -> str:
        return "asr_inference"


class TTSInferenceConfig(BaseConfig):
    """TTS推理配置"""

    model_name: str = Field(default="coqui", description="TTS模型名称")
    voice_name: str = Field(default="default", description="语音名称")
    language: str = Field(default="zh", description="合成语言")

    # 合成参数
    speed: float = Field(default=1.0, description="语速")
    pitch: float = Field(default=1.0, description="音调")
    volume: float = Field(default=1.0, description="音量")

    # 输出格式
    output_format: str = Field(default="wav", description="输出音频格式")
    sample_rate: int = Field(default=22050, description="输出采样率")

    @classmethod
    def get_config_key(cls) -> str:
        return "tts_inference"


class VoiceConversionConfig(BaseConfig):
    """声线转换配置"""

    model_path: str = Field(..., description="声线转换模型路径")
    target_speaker: str = Field(default="default", description="目标说话人")

    # 转换参数
    conversion_strength: float = Field(default=1.0, description="转换强度")
    preserve_prosody: bool = Field(default=True, description="是否保持韵律")

    @classmethod
    def get_config_key(cls) -> str:
        return "voice_conversion"


class InferenceConfig(BaseConfig):
    """推理主配置"""

    # 服务配置
    api: APIConfig = Field(default_factory=APIConfig, description="API配置")
    audio_processing: AudioProcessingConfig = Field(
        default_factory=AudioProcessingConfig, description="音频处理配置")

    # 模型配置
    wake_word: WakeWordInferenceConfig = Field(
        default_factory=lambda: WakeWordInferenceConfig(model_path=""), description="语音唤醒配置")
    asr: ASRInferenceConfig = Field(
        default_factory=ASRInferenceConfig, description="语音识别配置")
    tts: TTSInferenceConfig = Field(
        default_factory=TTSInferenceConfig, description="语音合成配置")
    voice_conversion: VoiceConversionConfig = Field(
        default_factory=lambda: VoiceConversionConfig(model_path=""), description="声线转换配置")

    # 全局设置
    enable_wake_word: bool = Field(default=True, description="是否启用语音唤醒")
    enable_asr: bool = Field(default=True, description="是否启用语音识别")
    enable_tts: bool = Field(default=True, description="是否启用语音合成")
    enable_voice_conversion: bool = Field(
        default=False, description="是否启用声线转换")

    # 性能监控
    enable_metrics: bool = Field(default=True, description="是否启用性能监控")
    metrics_port: int = Field(default=9090, description="监控端口")

    @classmethod
    def get_config_key(cls) -> str:
        return "inference"
