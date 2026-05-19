# -*- coding:utf-8 -*-
"""
统一日志配置模块
提供一致的日志格式和配置
"""
import os
import logging
import sys
from pathlib import Path
from typing import Optional


# 默认日志格式
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '%(message)s'
DEBUG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'


def setup_logger(
    name: str,
    level: str = None,
    log_file: Optional[str] = None,
    format_style: str = 'default'
) -> logging.Logger:
    """
    设置统一的日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file: 日志文件路径（可选）
        format_style: 格式风格（default, simple, debug）

    Returns:
        配置好的日志记录器
    """
    # 获取或创建日志记录器
    logger = logging.getLogger(name)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 设置日志级别
    if level is None:
        level = os.environ.get('TRAINER_LOG_LEVEL', 'INFO').upper()

    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logger.setLevel(level_map.get(level, logging.INFO))

    # 选择格式
    format_map = {
        'default': DEFAULT_FORMAT,
        'simple': SIMPLE_FORMAT,
        'debug': DEBUG_FORMAT
    }
    log_format = format_map.get(format_style, DEFAULT_FORMAT)
    formatter = logging.Formatter(log_format)

    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 防止日志向上传播
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器（如果不存在则创建）

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# 预定义的日志记录器
def get_api_logger() -> logging.Logger:
    """获取 API 日志记录器"""
    return get_logger('trainer.api')


def get_training_logger() -> logging.Logger:
    """获取训练日志记录器"""
    return get_logger('trainer.training')


def get_database_logger() -> logging.Logger:
    """获取数据库日志记录器"""
    return get_logger('trainer.database')


def get_storage_logger() -> logging.Logger:
    """获取存储日志记录器"""
    return get_logger('trainer.storage')
