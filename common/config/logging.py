"""
日志配置类

专门管理日志系统的配置参数
"""

from typing import Any, Dict, Optional

from pydantic import Field, validator

from .base import BaseConfig


class LogHandlerConfig(BaseConfig):
    """单个日志处理器配置"""

    type: str = Field(..., description="处理器类型：file/console/rotating")
    level: str = Field(default="INFO", description="日志级别")
    format: Optional[str] = Field(default=None, description="日志格式")
    filename: Optional[str] = Field(default=None, description="文件名（file类型需要）")
    max_bytes: Optional[int] = Field(
        default=10*1024*1024, description="最大文件大小")
    backup_count: Optional[int] = Field(default=5, description="备份文件数量")

    @classmethod
    def get_config_key(cls) -> str:
        return "handler"


class LoggerConfig(BaseConfig):
    """单个日志器配置"""

    name: str = Field(..., description="日志器名称")
    level: str = Field(default="INFO", description="日志级别")
    handlers: list = Field(default=[], description="使用的处理器列表")
    propagate: bool = Field(default=True, description="是否向父级传播")

    @classmethod
    def get_config_key(cls) -> str:
        return "logger"


class LoggingConfig(BaseConfig):
    """日志系统主配置"""

    # 全局设置
    version: int = Field(default=1, description="配置格式版本")
    disable_existing_loggers: bool = Field(
        default=False, description="是否禁用现有日志器")

    # 基础配置
    log_dir: str = Field(default="./logs", description="日志目录")
    default_level: str = Field(default="INFO", description="默认日志级别")
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="时间格式")

    # 格式化器配置
    formatters: Dict[str, Dict[str, Any]] = Field(
        default={
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
            }
        },
        description="格式化器配置"
    )

    # 处理器配置
    handlers: Dict[str, Dict[str, Any]] = Field(
        default={
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file_all": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": "logs/all.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/error.log",
                "maxBytes": 10485760,
                "backupCount": 5,
                "encoding": "utf8"
            }
        },
        description="处理器配置"
    )

    # 日志器配置
    loggers: Dict[str, Dict[str, Any]] = Field(
        default={
            "training": {
                "level": "DEBUG",
                "handlers": ["console", "file_all"],
                "propagate": False
            },
            "inference": {
                "level": "INFO",
                "handlers": ["console", "file_all"],
                "propagate": False
            },
            "common": {
                "level": "INFO",
                "handlers": ["console", "file_all"],
                "propagate": False
            }
        },
        description="日志器配置"
    )

    # 根日志器配置
    root: Dict[str, Any] = Field(
        default={
            "level": "INFO",
            "handlers": ["console", "file_all", "file_error"]
        },
        description="根日志器配置"
    )

    # 环境变量覆盖
    use_env_vars: bool = Field(default=True, description="是否使用环境变量覆盖")
    env_prefix: str = Field(default="LOG_", description="环境变量前缀")

    @classmethod
    def get_config_key(cls) -> str:
        return "logging"

    @validator('default_level', 'root')
    def validate_log_levels(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if isinstance(v, str):
            if v.upper() not in valid_levels:
                raise ValueError(f'日志级别必须是 {valid_levels} 中的一个')
        elif isinstance(v, dict) and 'level' in v:
            if v['level'].upper() not in valid_levels:
                raise ValueError(f'日志级别必须是 {valid_levels} 中的一个')
        return v

    def get_dict_config(self) -> Dict[str, Any]:
        """返回logging.config.dictConfig()可用的配置字典"""
        return {
            "version": self.version,
            "disable_existing_loggers": self.disable_existing_loggers,
            "formatters": self.formatters,
            "handlers": self.handlers,
            "loggers": self.loggers,
            "root": self.root
        }

    def get_handler_config(self, handler_name: str) -> Optional[Dict[str, Any]]:
        """获取指定处理器的配置"""
        return self.handlers.get(handler_name)

    def get_logger_config(self, logger_name: str) -> Optional[Dict[str, Any]]:
        """获取指定日志器的配置"""
        return self.loggers.get(logger_name)
