#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径处理工具

MinIO/RustFS 的磁盘后端目录不是业务文件系统，不能直接读写。这里统一把
minio://bucket/object 映射到训练服务本地工作区，并通过 S3 API 下载/上传。
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from minio.error import S3Error
except ImportError:
    S3Error = Exception

try:
    from config import TrainerConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from utils.minio_client import get_minio_manager
    MINIO_AVAILABLE = True
except ImportError:
    try:
        from minio_client import get_minio_manager
        MINIO_AVAILABLE = True
    except ImportError:
        MINIO_AVAILABLE = False
        get_minio_manager = None
        logger.warning("MinIO客户端未安装或导入失败")


class PathHandler:
    """路径处理器，支持本地路径和 MinIO/RustFS 对象路径"""

    _storage_config_loaded = False
    _bucket = 'algorithm'
    _base_path = 'trainer'
    _cache_root = '/data/trainer_work'

    @staticmethod
    def _load_storage_config() -> None:
        """加载对象存储和本地工作区配置"""
        if PathHandler._storage_config_loaded:
            return

        bucket = 'algorithm'
        base_path = 'trainer'
        cache_root = os.getenv('TRAINER_WORK_ROOT') or os.getenv('TRAINER_CACHE_ROOT') or '/data/trainer_work'

        if CONFIG_AVAILABLE:
            try:
                trainer_config = TrainerConfig()
                minio_config = trainer_config.get_minio_config()
                storage_config = trainer_config.get_storage_config()
                bucket = str(minio_config.get('bucket', bucket) or bucket).strip()
                base_path = str(minio_config.get('base_path', base_path) or base_path).strip('/')
                cache_root = (
                    os.getenv('TRAINER_WORK_ROOT')
                    or os.getenv('TRAINER_CACHE_ROOT')
                    or str(storage_config.get('object_storage_cache_root', '') or '').strip()
                    or cache_root
                )
            except Exception as e:
                logger.warning(f"加载对象存储配置失败: {str(e)}")

        env_bucket = os.getenv('TRAINER_STORAGE_BUCKET')
        env_base_path = os.getenv('TRAINER_STORAGE_BASE_PATH')
        if env_bucket:
            bucket = env_bucket.strip()
        if env_base_path:
            base_path = env_base_path.strip().strip('/')

        PathHandler._bucket = bucket or 'algorithm'
        PathHandler._base_path = base_path or 'trainer'
        PathHandler._cache_root = cache_root or '/data/trainer_work'
        PathHandler._storage_config_loaded = True

        logger.info(f"对象存储本地工作区: {PathHandler._cache_root}")

    @staticmethod
    def get_storage_root() -> str:
        """返回对象存储本地工作区根目录"""
        PathHandler._load_storage_config()
        return PathHandler._cache_root

    @staticmethod
    def get_storage_bucket() -> str:
        """返回对象存储默认 bucket"""
        PathHandler._load_storage_config()
        return PathHandler._bucket

    @staticmethod
    def get_storage_base_path() -> str:
        """返回对象存储默认 base_path"""
        PathHandler._load_storage_config()
        return PathHandler._base_path

    @staticmethod
    def build_minio_uri(bucket: str, object_path: str = '') -> str:
        """构造标准 minio:// URI"""
        bucket = str(bucket or '').strip().strip('/')
        object_path = str(object_path or '').strip().replace(os.sep, '/').lstrip('/')
        return f"minio://{bucket}/{object_path}" if object_path else f"minio://{bucket}"

    @staticmethod
    def build_storage_uri(*parts: str) -> str:
        """基于默认 bucket/base_path 构造对象存储 URI"""
        PathHandler._load_storage_config()
        normalized_parts = [PathHandler.get_storage_base_path()]
        normalized_parts.extend(str(part).strip().strip('/') for part in parts if str(part).strip())
        object_path = '/'.join(part for part in normalized_parts if part)
        return PathHandler.build_minio_uri(PathHandler.get_storage_bucket(), object_path)

    @staticmethod
    def normalize_minio_path(path: str) -> str:
        """兼容 minio:/xxx 这类单斜杠写法，统一规范成 minio://xxx"""
        path = str(path or '')
        if path.startswith('minio:/') and not path.startswith('minio://'):
            return 'minio://' + path[len('minio:/'):]
        if path.startswith('s3:/') and not path.startswith('s3://'):
            return 's3://' + path[len('s3:/'):]
        return path

    @staticmethod
    def is_minio_path(path: str) -> bool:
        """判断是否为 MinIO/S3 路径"""
        normalized = PathHandler.normalize_minio_path(path)
        return normalized.startswith('minio://') or normalized.startswith('s3://')

    @staticmethod
    def parse_minio_path(path: str) -> Tuple[str, str]:
        """
        解析 MinIO 路径。

        Args:
            path: minio://algorithm/trainer/original_dataset/123

        Returns:
            (bucket, object_path)
        """
        path = PathHandler.normalize_minio_path(path)
        if path.startswith('minio://'):
            path = path[8:]
        elif path.startswith('s3://'):
            path = path[5:]

        parts = path.split('/', 1)
        if len(parts) == 2:
            return parts[0], parts[1].strip('/')
        return parts[0], ''

    @staticmethod
    def _cache_path(bucket: str, object_path: str = '') -> Path:
        PathHandler._load_storage_config()
        local_path = Path(PathHandler._cache_root) / bucket
        if object_path:
            local_path = local_path / object_path
        return local_path

    @staticmethod
    def resolve_direct_local_path(path: str, require_exists: bool = True) -> Optional[str]:
        """
        历史兼容函数：不再返回 RustFS 后端目录，只返回本地工作区路径。
        """
        if not PathHandler.is_minio_path(path):
            candidate = Path(path)
            if not require_exists or candidate.exists():
                return str(candidate)
            return None

        bucket, object_path = PathHandler.parse_minio_path(path)
        local_path = PathHandler._cache_path(bucket, object_path)
        if require_exists and not local_path.exists():
            return None
        return str(local_path)

    @staticmethod
    def resolve_storage_local_path(*parts: str, require_exists: bool = False) -> str:
        """基于默认 bucket/base_path 返回本地工作区路径"""
        storage_uri = PathHandler.build_storage_uri(*parts)
        local_path = PathHandler.resolve_direct_local_path(storage_uri, require_exists=require_exists)
        if not local_path:
            raise FileNotFoundError(f"本地工作区路径不存在: {storage_uri}")
        return local_path

    @staticmethod
    def convert_local_to_minio_path(local_path: str, require_within_root: bool = True) -> Optional[str]:
        """将本地工作区路径反向转换为 minio:// URI"""
        PathHandler._load_storage_config()
        local_path_obj = Path(local_path).resolve()
        cache_root = Path(PathHandler._cache_root).resolve()

        try:
            relative_path = local_path_obj.relative_to(cache_root)
        except ValueError:
            if require_within_root:
                return None
            return None

        parts = relative_path.parts
        if not parts:
            return None

        bucket = parts[0]
        object_path = '/'.join(parts[1:])
        return PathHandler.build_minio_uri(bucket, object_path)

    @staticmethod
    def _get_manager():
        if not MINIO_AVAILABLE or get_minio_manager is None:
            raise RuntimeError("MinIO功能不可用，请安装 minio 包并检查配置")

        manager = get_minio_manager()
        if not manager or not manager.client:
            raise RuntimeError("MinIO客户端未初始化")
        return manager

    @staticmethod
    def _object_exists(manager, bucket: str, object_path: str) -> bool:
        if not object_path:
            return False
        try:
            manager.client.stat_object(bucket, object_path)
            return True
        except S3Error:
            return False
        except Exception:
            return False

    @staticmethod
    def _list_objects(manager, bucket: str, prefix: str):
        objects = manager.client.list_objects(bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects if obj.object_name and not obj.object_name.endswith('/')]

    @staticmethod
    def download_from_minio(minio_path: str, local_dir: Optional[str] = None, skip_if_exists: bool = True) -> str:
        """
        从 MinIO/RustFS 通过 S3 API 下载文件或目录到本地工作区。
        """
        manager = PathHandler._get_manager()
        bucket, object_path = PathHandler.parse_minio_path(minio_path)
        if not object_path:
            raise ValueError(f"MinIO路径缺少对象路径: {minio_path}")

        if local_dir is None:
            local_path = PathHandler._cache_path(bucket, object_path)
        else:
            local_path = Path(local_dir) / Path(object_path.rstrip('/')).name

        if PathHandler._object_exists(manager, bucket, object_path):
            if skip_if_exists and local_path.is_file() and local_path.stat().st_size > 0:
                logger.info(f"本地文件已存在，跳过下载: {local_path}")
                return str(local_path)

            local_path.parent.mkdir(parents=True, exist_ok=True)
            manager.client.fget_object(bucket, object_path, str(local_path))
            logger.info(f"对象文件下载成功: {bucket}/{object_path} -> {local_path}")
            return str(local_path)

        prefix = object_path.rstrip('/') + '/'
        object_names = PathHandler._list_objects(manager, bucket, prefix)
        if not object_names:
            raise FileNotFoundError(f"对象或对象前缀不存在: {minio_path}")

        if skip_if_exists and local_path.is_dir() and any(item.is_file() for item in local_path.rglob('*')):
            logger.info(f"本地目录已存在且有内容，跳过下载: {local_path}")
            return str(local_path)

        local_path.mkdir(parents=True, exist_ok=True)
        download_count = 0
        skip_count = 0
        for object_name in object_names:
            relative_name = object_name[len(prefix):].lstrip('/')
            if not relative_name:
                continue

            local_file = local_path / relative_name
            if skip_if_exists and local_file.is_file() and local_file.stat().st_size > 0:
                skip_count += 1
                continue

            local_file.parent.mkdir(parents=True, exist_ok=True)
            manager.client.fget_object(bucket, object_name, str(local_file))
            download_count += 1

        logger.info(
            f"对象目录下载完成: {minio_path} -> {local_path}, "
            f"下载{download_count}个, 跳过{skip_count}个"
        )
        return str(local_path)

    @staticmethod
    def get_local_path(path: str, download_if_needed: bool = True) -> str:
        """
        获取本地路径。MinIO路径会通过 S3 API 下载到本地工作区。
        """
        if not path:
            return path

        if PathHandler.is_minio_path(path):
            if download_if_needed:
                return PathHandler.download_from_minio(path, skip_if_exists=True)

            bucket, object_path = PathHandler.parse_minio_path(path)
            return str(PathHandler._cache_path(bucket, object_path))

        return path

    @staticmethod
    def upload_to_minio_if_needed(local_path: str, minio_path: Optional[str] = None) -> Optional[str]:
        """
        如果给定 minio_path，则通过 S3 API 上传本地文件或目录。
        """
        if not minio_path or not PathHandler.is_minio_path(minio_path):
            inferred = PathHandler.convert_local_to_minio_path(local_path, require_within_root=True)
            if not inferred:
                return None
            minio_path = inferred

        manager = PathHandler._get_manager()
        bucket, object_path = PathHandler.parse_minio_path(minio_path)
        local_path_obj = Path(local_path)

        if not local_path_obj.exists():
            raise FileNotFoundError(f"待上传路径不存在: {local_path_obj}")

        if local_path_obj.is_file():
            target_object = object_path
            if target_object.endswith('/'):
                target_object = f"{target_object}{local_path_obj.name}"
            with open(local_path_obj, 'rb') as file_data:
                manager.client.put_object(
                    bucket,
                    target_object,
                    file_data,
                    length=local_path_obj.stat().st_size
                )
            logger.info(f"对象文件上传成功: {local_path_obj} -> {bucket}/{target_object}")
            return PathHandler.build_minio_uri(bucket, target_object)

        upload_count = 0
        target_prefix = object_path.rstrip('/')
        for file_path in local_path_obj.rglob('*'):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(local_path_obj).as_posix()
            target_object = f"{target_prefix}/{relative_path}" if target_prefix else relative_path
            with open(file_path, 'rb') as file_data:
                manager.client.put_object(
                    bucket,
                    target_object,
                    file_data,
                    length=file_path.stat().st_size
                )
            upload_count += 1

        logger.info(f"对象目录上传完成: {local_path_obj} -> {bucket}/{target_prefix}, count={upload_count}")
        return PathHandler.build_minio_uri(bucket, target_prefix)


def ensure_local_path(path: str) -> str:
    """
    确保获得本地路径。

    MinIO/RustFS路径会通过 S3 API 下载到本地工作区。下载失败时直接抛错，
    避免后续代码误把 minio:// 或 RustFS 后端目录当成本地文件处理。
    """
    return PathHandler.get_local_path(path, download_if_needed=True)
