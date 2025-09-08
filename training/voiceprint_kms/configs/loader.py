"""
配置加载器类
实现从YAML文件加载配置并转换为Python类实例的功能

作者: yuger
创建时间: 2025-09-09
"""

import os  # 操作系统接口,用于路径处理
from pathlib import Path  # 路径处理,用于跨平台路径操作
from typing import Any, Dict, Optional, Type, Union  # 类型注解

import yaml  # YAML文件解析库

# 导入各配置类
from .model import (BaseFusionConfig, BaseModelConfig, ConformerConfig,
                    EcapaTdnnConfig, VoiceprintKeywordFusionConfig)


class ConfigLoader:
    """配置加载器 - 负责从YAML文件加载配置并转换为配置类实例"""

    # 模型配置类映射表
    MODEL_CONFIG_MAPPING = {
        "ecapa_tdnn": EcapaTdnnConfig,
        "conformer": ConformerConfig,
    }

    # 融合配置类映射表
    FUSION_CONFIG_MAPPING = {
        "voiceprint_keyword_fusion": VoiceprintKeywordFusionConfig,
    }

    def __init__(self, config_root: Optional[str] = None):
        """
        初始化配置加载器
        Args:
            config_root (Optional[str]): 配置文件根目录路径,默认为None表示当前目录
        """
        if config_root is None:
            # 自动推断配置根目录
            current_dir = Path(__file__).parent
            self.config_root = current_dir.parent  # 假设配置文件在上级目录
        else:
            self.config_root = Path(config_root)

    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """
        加载YAML文件并返回字典

        Args:
            file_path (str): YAML文件路径
        Returns:
            Dict[str, Any]: 解析后的配置字典

        Raises:
            FileNotFoundError: 如果文件不存在
            yaml.YAMLError: 如果YAML解析失败
        """
        full_path = self.config_root / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {full_path}")
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            return content
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML文件格式错误{full_path}: {e}")

    def _create_config_instance(self, config_class: Type, config_data: Dict[str, Any]) -> Any:
        """
        根据配置数据创建配置类实例

        Args:
            config_class (Type): 配置类
            config_data (Dict[str,Any]): 配置数据

        Returns:
            Any: 配置类实例
        """
        try:
            # 数值类型转换 - 处理YAML中的科学计数法字符串
            converted_data = self._convert_numeric_strings(config_data)
            # 创建配置实例
            instance = config_class(**converted_data)
            # 验证配置
            instance.validate()
            return instance
        except TypeError as e:
            raise ValueError(f"创建配置实例失败 {config_class.__name__}: {e}")

    def _convert_numeric_strings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换YAML中的数值字符串为对应的数值类型

        Args:
            data: 原始配置数据

        Returns:
            转换后的配置数据
        """
        converted = {}

        # 需要转换为float的字段列表
        float_fields = {
            'weight_decay', 'dropout', 'droppath_rate', 'label_smoothing',
            'norm_eps', 'init_gain', 'attention_dropout', 'ff_dropout',
            'conv_dropout', 'keyword_threshold', 'final_dropout',
            'voice_weight', 'keyword_weight', 'speaker_task_weight',
            'keyword_task_weight'
        }

        for key, value in data.items():
            if key in float_fields and isinstance(value, str):
                try:
                    # 尝试转换科学计数法字符串为浮点数
                    converted[key] = float(value)
                except ValueError:
                    # 如果转换失败，保持原值
                    converted[key] = value
            else:
                converted[key] = value

        return converted

    def load_model_config(self, config_key: str, yaml_file: str, model_type: Optional[str] = None) -> Union[EcapaTdnnConfig, ConformerConfig]:
        """
        加载模型配置

        Args:
            config_key: YAML文件中的配置键名(如:"ecapa_v0_1")
            yaml_file: YAML文件名 (如:"ecapa_tdnn.yaml")
            model_type: 模型类型,如果不指定会根据yaml_file推断

        Returns:
            Union[EcapaTdnnConfig,ConformerConfig]: 模型配置实例

        Example:
            loader = ConfigLoader()
            config = loader.load_model_config("ecapa_v0_1","ecapa_tdnn.yaml")
        """
        # 加载yaml文件
        yaml_content = self._load_yaml_file(yaml_file)

        # 检查配置键是否存在
        if config_key not in yaml_content:
            raise KeyError(f"配置键 '{config_key}' 在文件 '{yaml_file}' 中未找到")

        config_data = yaml_content[config_key]

        # 自动推断模型类型
        if model_type is None:
            if "ecapa" in yaml_file.lower():
                model_type = "ecapa_tdnn"
            elif "conformer" in yaml_file.lower():
                model_type = "conformer"
            else:
                raise ValueError(f"无法从文件名推断模型类型: {yaml_file}")

        # 获取对应的配置类
        if model_type not in self.MODEL_CONFIG_MAPPING:
            raise ValueError(f"不支持的模型类型: {model_type}")

        config_class = self.MODEL_CONFIG_MAPPING[model_type]

        return self._create_config_instance(config_class, config_data)

    def load_fusion_config(self,
                           config_key: str,
                           yaml_file: str = "model/fusion.yaml",
                           fusion_type: str = "voiceprint_keyword_fusion") -> VoiceprintKeywordFusionConfig:
        """
        加载融合配置

        Args:
            config_key: YAML文件中的配置键名(如: "voiceprint_keyword_fusion_attention_v0_1")
            yaml_file: YAML文件名,默认为"model/fusion.yaml"
            fusion_type: 融合类型,默认为"voiceprint_keyword_fusion"

        Returns:
            VoiceprintKeywordFusionConfig: 融合配置实例

        Example:
            loader = ConfigLoader()
            config = loader.load_fusion_config("voiceprint_keyword_fusion_attention_v0_1")
        """
        # 加载YAML文件
        yaml_content = self._load_yaml_file(yaml_file)

        # 检查配置键是否存在
        if config_key not in yaml_content:
            raise KeyError(f"配置键 '{config_key}' 在文件 '{yaml_file}' 中未找到")

        config_data = yaml_content[config_key]

        # 获取对应的配置类
        if fusion_type not in self.FUSION_CONFIG_MAPPING:
            raise ValueError(f"不支持的融合类型: {fusion_type}")

        config_class = self.FUSION_CONFIG_MAPPING[fusion_type]

        return self._create_config_instance(config_class, config_data)

    def list_available_model_configs(self, yaml_file: str) -> list:
        """
        列出YAML文件中所有可用的配置

        Args:
            yaml_file (str): YAML文件名

        Returns:
            配置键名列表 (List[str])
        """
        yaml_content = self._load_yaml_file(yaml_file)

        return list(yaml_content.keys())


class ConfigManager:
    """统一配置管理器 - 提供更高级的配置管理功能"""

    def __init__(self, config_root: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_root (Optional[str]): 配置文件根目录路径,默认为None表示当前目录
        """
        self.loader = ConfigLoader(config_root)

    def get_ecapa_config(self, version: str) -> EcapaTdnnConfig:
        """
        获取ECAPA-TDNN配置

        Args:
            version (str): 配置版本

        Returns:
            EcapaTdnnConfig: ECAPA-TDNN配置实例
        """
        config_key = f"ecapa_{version}"
        return self.loader.load_model_config(config_key, "model/ecapa_tdnn.yaml")

    def get_conformer_config(self, version: str) -> ConformerConfig:
        """
        获取Conformer配置

        Args:
            version (str): 配置版本

        Returns:
            ConformerConfig: Conformer配置实例
        """
        config_key = f"conformer_{version}"
        return self.loader.load_model_config(config_key, "model/conformer.yaml")

    def get_fusion_config(self, version: str) -> VoiceprintKeywordFusionConfig:
        """
        获取融合配置

        Args:
            version (str): 配置版本

        Returns:
            VoiceprintKeywordFusionConfig: 融合配置实例
        """
        config_key = f"voiceprint_keyword_fusion_{version}"
        return self.loader.load_fusion_config(config_key, "model/fusion.yaml")

    def get_complete_model_config(self,
                                  fusion_method: str = "attention",
                                  version: str = 'v0_1') -> Dict[str, Any]:
        """
        获取完整模型配置(包含所有组件)

        Args:
            fusion_method (str, optional): 融合方法. Defaults to "attention".
            version (str, optional): 版本. Defaults to 'v0_1'.

        Returns:
            Dict[str, Any]: 包含所有配置字典
        """
        return {
            "ecapa_tdnn": self.get_ecapa_config(version),
            "conformer": self.get_conformer_config(version),
            "fusion": self.get_fusion_config(f"{fusion_method}_{version}"),
        }
