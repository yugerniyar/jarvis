"""
日志记录模块
设计要点 ：
- 入口调用 configur_logging(...) 进行一次性配置 (训练/服务入口)
- 之后各模块通过 get_logger(__name__) 获取logger
- 支持文件滚动 控制台输出 环境变量覆盖 可选json输出
提供统一的日志记录接口，支持：
- 日志级别控制
- 日志格式化
- 日志输出到文件
"""

import logging  # Python内置的 标准日志框架 提供 Logger Handler Formatter 用于打印和管理日志
import os  # 与操作系统交互 做路径拼接 创建目录 读取环境变量
from logging.handlers import \
    RotatingFileHandler  # 一个日志 Handler 按文件大小滚动 主要目的是为了切分 放置一个日志文件无限大
from typing import Optional  # 类型注解 用来表示参数类型

# 配置默认值 便于后续修改或者通过环境变量直接覆盖掉
# 当前目录下的默认日志目录 os.getcwd() 返回当前进程工作目录 一般为绝对路径
_DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")
_DEFAULT_ALL_LOG = "all.log"    # 保存所有级别日志的文件名
_DEFAULT_ERROR_LOG = "error.log"    # 保存错误级别日志的文件名
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024   # 单个日志文件最大字节数
_DEFAULT_BACKUP_COUNT = 5   # 日志文件备份数量
_DEFAULT_CONSOLE_ENABLED = True  # 默认启用控制台输出
_DEFAULT_JSON_ENABLED = False  # 默认使用文本格式


def _make_formatter() -> logging.Formatter:
    """
    创建日志格式化器
    格式: 时间 | 级别 | 模块名:行号 | 消息内容
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def configure_logging(
    log_dir: Optional[str] = None,
    console: bool = _DEFAULT_CONSOLE_ENABLED,
    log_level: str = "INFO"
) -> None:
    """
    配置全局日志系统 (在程序入口调用一次)
    参数说明:
    log_dir: 日志文件目录
    console: 是否启用控制台输出
    log_level: 日志级别 (可选值: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    # 第一步: 确定日志目录(支持环境变量覆盖)
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", _DEFAULT_LOG_DIR)

    # 第二部: 创建日志目录（如果不存在的话)
    os.makedirs(log_dir, exist_ok=True)

    # 第三步: 根获取跟日志器并设置级别
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 第四步: 防止重复配置
    if root_logger.handlers:
        return

    # 第五步: 创建格式化器
    formatter = _make_formatter()

    # 第六步: 创建all.log文件handler
    all_log_path = os.path.join(log_dir, _DEFAULT_ALL_LOG)
    all_handler = RotatingFileHandler(
        all_log_path,
        maxBytes=_DEFAULT_MAX_BYTES,
        backupCount=_DEFAULT_BACKUP_COUNT,
        encoding="utf-8"
    )
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(formatter)
    root_logger.addHandler(all_handler)

    # 第七步: 创建error.log 文件 handler
    error_log_path = os.path.join(log_dir, _DEFAULT_ERROR_LOG)
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=_DEFAULT_MAX_BYTES,
        backupCount=_DEFAULT_BACKUP_COUNT,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # 第八步: 創建控制檯handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def get_logger(name: str = __name__) -> logging.Logger:
    """
    获取日志器(在每个模块中使用)
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    configure_logging()
    logger = get_logger(__name__)

    logger.debug("这是一个调试日志")
    logger.info("这是一个信息日志")
    logger.warning("这是一个警告日志")
    logger.error("这是一个错误日志")
    logger.critical("这是一个严重日志")
    print("日志配置完成，查看logs目录下的日志文件。")
