"""
基础配置类

提供配置管理的基础功能：
- 配置加载和验证
- 环境变量支持
- 配置热更新
- 微服务配置分离
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field


class BaseConfig(BaseModel, ABC):
    """
    配置基类

    所有配置类都应该继承此类，提供：
    - 数据验证
    - 环境变量支持
    - 序列化/反序列化
    """

    class Config:
        # 允许环境变量覆盖配置
        env_file = ".env"
        env_file_encoding = "utf-8"
        # 验证赋值
        validate_assignment = True
        # 允许任意类型（用于扩展）
        arbitrary_types_allowed = True

    @classmethod
    @abstractmethod
    def get_config_key(cls) -> str:
        """返回配置在文件中的键名"""
        pass

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "BaseConfig":
        """从配置文件加载"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 根据文件扩展名选择解析器
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

        # 获取对应的配置部分
        config_key = cls.get_config_key()
        config_data = data.get(config_key, {})

        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()

    def to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """转换为YAML格式"""
        yaml_str = yaml.dump(
            {self.get_config_key(): self.dict()},
            default_flow_style=False,
            allow_unicode=True
        )

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)

        return yaml_str

    def to_json(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """转换为JSON格式"""
        json_str = json.dumps(
            {self.get_config_key(): self.dict()},
            ensure_ascii=False,
            indent=2
        )

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str


class ConfigManager:
    """
    配置管理器

    统一管理所有配置，支持：
    - 多个配置源合并
    - 配置热更新
    - 环境变量覆盖
    - 微服务配置分离
    """

    def __init__(self, config_dir: Union[str, Path] = None):
        """
        初始化配置管理器

        Args:
            config_dir: 配置文件目录，默认为项目根目录的config
        """
        if config_dir is None:
            # 默认配置目录
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self.configs: Dict[str, BaseConfig] = {}
        self._load_default_configs()

    def _load_default_configs(self):
        """加载默认配置文件"""
        # 查找配置文件
        config_files = []
        for pattern in ["*.yaml", "*.yml", "*.json"]:
            config_files.extend(self.config_dir.glob(pattern))

        # 优先级：environment specific > default
        env = os.getenv("ENVIRONMENT", "development")

        # 按优先级排序配置文件
        priority_files = []
        for config_file in config_files:
            if env in config_file.stem:
                priority_files.insert(0, config_file)  # 环境特定配置优先
            else:
                priority_files.append(config_file)

        # 加载配置文件
        for config_file in priority_files:
            try:
                self._load_config_file(config_file)
            except Exception as e:
                print(f"警告: 加载配置文件失败 {config_file}: {e}")

    def _load_config_file(self, config_path: Path):
        """加载单个配置文件"""
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            return

        # 存储原始配置数据（用于后续创建特定配置对象）
        for key, value in data.items():
            if key not in self.configs:
                self.configs[key] = value

    def get_config(self, config_class: type, config_key: str = None) -> BaseConfig:
        """
        获取特定类型的配置

        Args:
            config_class: 配置类
            config_key: 配置键名，如果不提供则使用类的默认键名

        Returns:
            配置对象
        """
        if config_key is None:
            config_key = config_class.get_config_key()

        config_data = self.configs.get(config_key, {})

        # 应用环境变量覆盖
        config_data = self._apply_env_overrides(config_data, config_key)

        return config_class(**config_data)

    def _apply_env_overrides(self, config_data: Dict, config_key: str) -> Dict:
        """应用环境变量覆盖"""
        # 查找以配置键名开头的环境变量
        prefix = f"{config_key.upper()}_"

        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # 转换环境变量键名为配置键名
                config_field = env_key[len(prefix):].lower()

                # 尝试转换数据类型
                try:
                    # 尝试解析为JSON（支持复杂数据类型）
                    config_data[config_field] = json.loads(env_value)
                except (json.JSONDecodeError, ValueError):
                    # 作为字符串处理
                    config_data[config_field] = env_value

        return config_data

    def set_config(self, config_key: str, config_obj: BaseConfig):
        """设置配置"""
        self.configs[config_key] = config_obj

    def reload_config(self, config_path: Union[str, Path]):
        """重新加载配置文件"""
        self._load_config_file(Path(config_path))

    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.configs.copy()

    def export_config(self, output_path: Union[str, Path], format: str = "yaml"):
        """导出当前配置到文件"""
        output_path = Path(output_path)

        if format.lower() == "yaml":
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.configs, f, default_flow_style=False,
                          allow_unicode=True)
        elif format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.configs, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")


# 全局配置管理器实例
_config_manager = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(config_class: type) -> BaseConfig:
    """快捷方式：获取配置"""
    return get_config_manager().get_config(config_class)
