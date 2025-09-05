"""
具体融合模型配置类定义

包含声纹识别和关键词检测的多模态融合模型配置
- VoiceprintKeywordFusionConfig: 声纹识别和关键词检测的融合配置

作者: yuger
创建时间: 2025-09-06
"""

from dataclasses import dataclass  # 数据类装饰器,简化配置类定义
from typing import Dict  # 类型注解,用于定义字典类型

from .base_fusion import BaseFusionConfig  # 导入融合基类


@dataclass
class VoiceprintKeywordFusionConfig(BaseFusionConfig):
    """声纹识别+关键词检测融合配置

    融合ECAPA-TDNN的声纹特征和Conformer的关键词特征
    用于多模态身份验证和唤醒词检测
    """

    # 基本属性 (覆盖基类默认值)
    name: str = "voiceprint_keyword_niyar"  # 具体的模型名称标识

    # 输入模态维度配置 (对应骨干网络的输出维度)
    voice_feature_dim: int = 192        # 声纹特征维度，来自ECAPA-TDNN的embedding_dim
    keyword_feature_dim: int = 512      # 关键词特征维度，来自Conformer的encoder_dim

    # 加权融合参数 (当fusion_method="weighted"时使用)
    voice_weight: float = 0.6           # 声纹特征权重，声纹对身份识别更重要
    keyword_weight: float = 0.4         # 关键词特征权重，用于唤醒检测辅助

    # 多任务输出配置
    num_speakers: int = 1000            # 支持的说话人数量，影响声纹分类器维度
    num_keywords: int = 1               # 支持的关键词数量，当前阶段设为1
    # 任务权重配置 (多任务学习中各任务的损失权重)
    speaker_task_weight: float = 1.0    # 声纹识别任务损失权重
    keyword_task_weight: float = 1.0    # 关键词检测任务损失权重

    # 特征预处理配置 (统一不同模态的特征维度)
    use_feature_projection: bool = True     # 是否对输入特征进行维度投影
    voice_projection_dim: int = 256         # 声纹特征投影后维度
    keyword_projection_dim: int = 256       # 关键词特征投影后维度

    # 抽象方法实现
    def get_input_modalities(self) -> Dict[str, int]:
        """获取输入模态及其维度"""
        return {
            "voice": self.voice_feature_dim,    # 声纹特征维度
            "keyword": self.keyword_feature_dim,  # 关键词特征维度
        }

    def get_modality_weights(self) -> Dict[str, float]:
        """获取各模态的权重配置"""
        return {
            "voice": self.voice_weight,         # 声纹特征权重
            "keyword": self.keyword_weight,     # 关键词特征权重
        }

    # 参数验证方法
    def validate(self) -> bool:
        """验证融合配置参数完整性和合理性"""
        # 基础参数验证
        self.validate_base_params()

        # 输入维度检查
        if self.voice_feature_dim <= 0:
            raise ValueError("voice_feature_dim 必须大于0")
        if self.keyword_feature_dim <= 0:
            raise ValueError("keyword_feature_dim 必须大于0")

        # 加权融合参数验证
        if self.fusion_method == "weighted":
            if not (0 <= self.voice_weight <= 1):
                raise ValueError("voice_weight 必须在 [0,1] 之间")
            if not (0 <= self.keyword_weight <= 1):
                raise ValueError("keyword_weight 必须在 [0,1] 之间")
            weight_sum = self.voice_weight + self.keyword_weight
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"voice_weight 和 keyword_weight 之和必须为1,当前为{weight_sum}")
        # 多任务输出检查
        if self.num_speakers <= 0:
            raise ValueError("num_speakers 必须大于0")
        if self.num_keywords <= 0:
            raise ValueError("num_keywords 必须大于0")
        if self.speaker_task_weight < 0:
            raise ValueError("speaker_task_weight 必须非负")
        if self.keyword_task_weight < 0:
            raise ValueError("keyword_task_weight 必须非负")
        # 特征投影配置验证
        if self.use_feature_projection:
            if self.voice_projection_dim <= 0:
                raise ValueError("voice_projection_dim 必须大于0")
            if self.keyword_projection_dim <= 0:
                raise ValueError("keyword_projection_dim 必须大于0")

        return True

    # 信息获取方法
    def get_fusion_info(self) -> dict:
        """获取多任务配置信息"""
        return {
            "tasks": {
                "speaker_recognition": {
                    "num_classes": self.num_speakers,
                    "loss_weight": self.speaker_task_weight,
                },
                "keyword_detection": {
                    "num_classes": self.num_keywords,
                    "loss_weight": self.keyword_task_weight,
                },
            },
            "feature_projection": {
                "enabled": self.use_feature_projection,
                "voice_projection_dim": self.voice_projection_dim if self.use_feature_projection else None,
                "keyword_projection_dim": self.keyword_projection_dim if self.use_feature_projection else None,
            }
        }

    def get_complete_info(self) -> dict:
        """获取完整的融合模型配置信息"""
        base_info = super().get_fusion_info()
        task_info = self.get_fusion_info()
        return {**base_info, "specific_config": task_info}
