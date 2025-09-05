"""
模型导入包

统一导入所有模型配置类,提供简洁的外部接口
"""

# 各种骨干网络配置
# 从不同的文件导入配置类
from .backbone_schemas import ConformerConfig, EcapaTdnnConfig
# 融合模型基类配置
from .base_fusion import BaseFusionConfig
# 模型基础配置
from .base_schemas import BaseModelConfig
# 融合模型具体实现配置
from .fusion_schemas import VoiceprintKeywordFusionConfig

# 控制对外可见得类
__all__ = [
    "BaseModelConfig",
    "EcapaTdnnConfig",
    "ConformerConfig",
    "BaseFusionConfig",
    "VoiceprintKeywordFusionConfig",
]
