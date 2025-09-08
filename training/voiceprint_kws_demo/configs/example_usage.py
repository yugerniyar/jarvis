"""
配置家在其使用示例与测试

演示如何使用ConfigLoader和ConfigManager类来加载和管理配置

作者: yuger
创建时间: 2025-09-09
"""

import os  # 操作系统接口,用于路径处理

from training.voiceprint_kms.configs.loader import ConfigLoader, ConfigManager


def basic_usage_example():
    """
    基础使用示例

    作者: yuger
    创建时间: 2025-09-09
    """
    print("=== 基础配置加载示例 ===")

    # 加载ECAPA-TDNN模型配置
    loader = ConfigLoader(config_root=os.path.join(
        "training", "voiceprint_kms", "configs"))
    ecapa_config = loader.load_model_config(
        "ecapa_v0_1", "model/ecapa_tdnn.yaml")
    print("ECAPA-TDNN配置:", ecapa_config)
    print("ECAPA-TDNN输入特征维度:", ecapa_config.input_dim)

    # 加载Conformer模型配置
    conformer_config = loader.load_model_config(
        "conformer_v0_1", "model/conformer.yaml")
    print("Conformer配置:", conformer_config)
    print("Conformer输入特征维度:", conformer_config.input_dim)

    # 加载融合模型配置
    fusion_config = loader.load_fusion_config(
        "voiceprint_keyword_fusion_attention_v0_1", "model/fusion.yaml")
    print("融合模型配置:", fusion_config)
    print("融合模型输入模态:", fusion_config.get_input_modalities())


def advanced_usage_example():
    """
    高级使用示例

    作者: yuger
    创建时间: 2025-09-09
    """
    print("\n=== 高级配置管理示例 ===")

    # 使用ConfigManager统一管理配置
    manager = ConfigManager(config_root=os.path.join(
        "training", "voiceprint_kms", "configs"))

    # 快速获取各种配置
    ecapa_config = manager.get_ecapa_config("v0_1")
    conformer_config = manager.get_conformer_config("v0_1")
    fusion_config = manager.get_fusion_config("attention_v0_1")

    print(f"快速加载完成:")
    print(f"  ECAPA: {ecapa_config.name}")
    print(f"  Conformer: {conformer_config.name}")
    print(f"  Fusion: {fusion_config.name}")

    # 展示所有可用配置键
    complete_config = manager.get_complete_model_config(
        fusion_method="attention")
    print("完整模型配置键:", list(complete_config.keys()))


def config_validation_example():
    """配置验证示例"""
    print("\n=== 配置验证示例 ===")

    loader = ConfigLoader(config_root=os.path.join(
        "training", "voiceprint_kms", "configs"))

    try:
        # 加载并验证配置
        config = loader.load_model_config(
            "ecapa_v0_1", "model/ecapa_tdnn.yaml")
        print(f"配置验证通过: {config.name}")

        # 显示配置信息
        info = config.get_model_info()
        print(f"基础信息: {info['basic_info']}")

    except Exception as e:
        print(f"配置验证失败: {e}")


def list_configs_example():
    """列出可用配置示例"""
    print("\n=== 可用配置列表 ===")

    loader = ConfigLoader(config_root=os.path.join(
        "training", "voiceprint_kms", "configs"))

    # 列出各文件中的可用配置
    ecapa_configs = loader.list_available_model_configs(
        "model/ecapa_tdnn.yaml")
    conformer_configs = loader.list_available_model_configs(
        "model/conformer.yaml")
    fusion_configs = loader.list_available_model_configs("model/fusion.yaml")

    print(f"ECAPA配置: {ecapa_configs}")
    print(f"Conformer配置: {conformer_configs}")
    print(f"融合配置: {fusion_configs}")


if __name__ == "__main__":
    # 运行所有示例
    basic_usage_example()
    advanced_usage_example()
    config_validation_example()
    list_configs_example()
