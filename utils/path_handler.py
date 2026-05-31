#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径处理工具：统一管理本地数据根目录下的路径拼接与解析"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from config import TrainerConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


LEGACY_STORAGE_PREFIXES = ('minio://', 'minio:')
LEGACY_STORAGE_ROOT_MARKERS = (
    'algorithm/trainer/',
    'training/algorithm/trainer/',
    'data/Sucai1/algorithm/trainer/',
    'data/Sucai1/training/algorithm/trainer/',
)


def _normalize_legacy_storage_path(path: str):
    if path is None:
        return None

    path_str = str(path).strip()
    if not path_str:
        return path

    if path_str.startswith(LEGACY_STORAGE_PREFIXES):
        relative_path = path_str
        for prefix in LEGACY_STORAGE_PREFIXES:
            if relative_path.startswith(prefix):
                relative_path = relative_path[len(prefix):]
                break

        relative_path = relative_path.lstrip('/')
        if not relative_path:
            return PathHandler.get_data_root()

        for marker in LEGACY_STORAGE_ROOT_MARKERS:
            if relative_path.startswith(marker):
                relative_path = relative_path[len(marker):]
                break

        data_root = Path(PathHandler.get_data_root())
        return str(data_root / relative_path)

    if path_str.startswith('/'):
        return path

    for marker in LEGACY_STORAGE_ROOT_MARKERS:
        if path_str.startswith(marker):
            relative_path = path_str[len(marker):]
            data_root = Path(PathHandler.get_data_root())
            return str(data_root / relative_path)

    return path


class PathHandler:
    """路径处理器，基于本地数据根目录管理所有数据路径"""

    _config_loaded = False
    _data_root = '/data/Sucai1/training/algorithm/trainer'

    @staticmethod
    def _load_config() -> None:
        if PathHandler._config_loaded:
            return

        data_root = os.getenv('TRAINER_DATA_ROOT') or '/data/Sucai1/training/algorithm/trainer'

        if CONFIG_AVAILABLE:
            try:
                trainer_config = TrainerConfig()
                data_root = trainer_config.get_data_root() or data_root
            except Exception as e:
                logger.warning(f"加载数据根目录配置失败: {e}")

        env_root = os.getenv('TRAINER_DATA_ROOT')
        if env_root:
            data_root = env_root

        PathHandler._data_root = data_root.rstrip('/')
        PathHandler._config_loaded = True
        logger.info(f"本地数据根目录: {PathHandler._data_root}")

    @staticmethod
    def get_data_root() -> str:
        PathHandler._load_config()
        return PathHandler._data_root

    @staticmethod
    def build_data_path(*parts: str) -> str:
        """基于数据根目录拼接完整路径"""
        PathHandler._load_config()
        normalized = [str(part).strip().strip('/') for part in parts if str(part).strip()]
        return str(Path(PathHandler._data_root) / '/'.join(normalized))

    @staticmethod
    def resolve_local_path(*parts: str, require_exists: bool = False) -> str:
        """基于数据根目录返回本地路径，可选检查存在性"""
        if len(parts) == 1:
            normalized = ensure_local_path(parts[0])
            if isinstance(normalized, str) and normalized and (
                normalized.startswith('/') or normalized.startswith(PathHandler.get_data_root())
            ):
                if require_exists and not Path(normalized).exists():
                    raise FileNotFoundError(f"路径不存在: {normalized}")
                return normalized

        local_path = PathHandler.build_data_path(*parts)
        if require_exists and not Path(local_path).exists():
            raise FileNotFoundError(f"路径不存在: {local_path}")
        return local_path

    @staticmethod
    def get_local_path(path: str) -> str:
        """返回本地路径（兼容旧对象存储 URI）"""
        return ensure_local_path(path)

    @staticmethod
    def is_legacy_remote_uri(path: str) -> bool:
        """识别旧对象存储 URI 前缀"""
        return bool(path) and str(path).strip().startswith(LEGACY_STORAGE_PREFIXES)


def ensure_local_path(path: str) -> str:
    """确保获得本地路径"""
    return _normalize_legacy_storage_path(path)
