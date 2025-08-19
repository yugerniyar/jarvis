"""
配置系统使用示例

演示如何使用配置管理系统
"""

import os
from pathlib import Path

from common.config import (AudioConfig, InferenceConfig, TrainingConfig,
                           WakeWordConfig, get_config, get_config_manager)


def demo_basic_usage():
    """基本使用示例"""
    print("=== 基本配置使用示例 ===")

    # 方式1: 使用全局配置管理器
    config_manager = get_config_manager()

    # 获取训练配置
    training_config = config_manager.get_config(TrainingConfig)
    print(f"训练轮数: {training_config.epochs}")
    print(f"批大小: {training_config.data.batch_size}")
    print(f"学习率: {training_config.optimizer.lr}")

    # 方式2: 使用快捷函数
    wake_word_config = get_config(WakeWordConfig)
    print(f"唤醒词: {wake_word_config.detection.primary_wake_word}")
    print(f"检测阈值: {wake_word_config.detection.detection_threshold}")


def demo_file_loading():
    """从文件加载配置示例"""
    print("\n=== 从文件加载配置示例 ===")

    config_file = Path(__file__).parent / "development.yaml"

    # 直接从文件加载
    training_config = TrainingConfig.from_file(config_file)
    print(f"从文件加载 - 训练轮数: {training_config.epochs}")

    # 音频配置
    audio_config = AudioConfig.from_file(config_file)
    print(f"采样率: {audio_config.get_sample_rate()}")
    print(f"MFCC特征数: {audio_config.feature_extraction.n_mfcc}")


def demo_env_override():
    """环境变量覆盖示例"""
    print("\n=== 环境变量覆盖示例 ===")

    # 设置环境变量
    os.environ["TRAINING_EPOCHS"] = "200"
    os.environ["TRAINING_BATCH_SIZE"] = "64"
    os.environ["WAKE_WORD_DETECTION_THRESHOLD"] = "0.7"

    # 重新获取配置（会应用环境变量覆盖）
    training_config = get_config(TrainingConfig)
    print(f"环境变量覆盖后 - 训练轮数: {training_config.epochs}")

    # 清理环境变量
    for key in ["TRAINING_EPOCHS", "TRAINING_BATCH_SIZE", "WAKE_WORD_DETECTION_THRESHOLD"]:
        os.environ.pop(key, None)


def demo_config_export():
    """配置导出示例"""
    print("\n=== 配置导出示例 ===")

    # 获取配置
    training_config = get_config(TrainingConfig)

    # 导出为YAML
    yaml_content = training_config.to_yaml()
    print("YAML格式配置:")
    print(yaml_content[:200] + "...")

    # 导出为JSON
    json_content = training_config.to_json()
    print("\nJSON格式配置:")
    print(json_content[:200] + "...")


def demo_microservice_config():
    """微服务配置分离示例"""
    print("\n=== 微服务配置分离示例 ===")

    # 推理服务配置
    inference_config = get_config(InferenceConfig)
    print(f"API端口: {inference_config.api.port}")
    print(f"启用唤醒词: {inference_config.enable_wake_word}")
    print(f"启用ASR: {inference_config.enable_asr}")

    # 语音唤醒服务独立配置
    wake_word_config = get_config(WakeWordConfig)
    print(f"VAD模式: {wake_word_config.vad.mode}")
    print(f"缓冲区大小: {wake_word_config.audio_buffer.buffer_size_ms}ms")


def demo_dynamic_config():
    """动态配置更新示例"""
    print("\n=== 动态配置更新示例 ===")

    config_manager = get_config_manager()

    # 获取当前配置
    training_config = config_manager.get_config(TrainingConfig)
    print(f"当前学习率: {training_config.optimizer.lr}")

    # 动态更新配置
    training_config.optimizer.lr = 0.0005
    config_manager.set_config("training", training_config)

    # 重新获取配置
    updated_config = config_manager.get_config(TrainingConfig)
    print(f"更新后学习率: {updated_config.optimizer.lr}")


def demo_validation():
    """配置验证示例"""
    print("\n=== 配置验证示例 ===")

    try:
        # 尝试创建无效配置
        invalid_config = TrainingConfig(
            epochs=-1,  # 无效值
            data={
                "train_dir": "/path/to/train",
                "val_dir": "/path/to/val",
                "batch_size": 0  # 无效值
            }
        )
    except Exception as e:
        print(f"配置验证失败: {e}")

    try:
        # 正确的配置
        valid_config = TrainingConfig(
            epochs=50,
            data={
                "train_dir": "/path/to/train",
                "val_dir": "/path/to/val",
                "batch_size": 32
            }
        )
        print("配置验证成功!")
    except Exception as e:
        print(f"意外错误: {e}")


if __name__ == "__main__":
    # 运行所有示例
    demo_basic_usage()
    demo_file_loading()
    demo_env_override()
    demo_config_export()
    demo_microservice_config()
    demo_dynamic_config()
    demo_validation()

    print("\n=== 配置系统演示完成 ===")
