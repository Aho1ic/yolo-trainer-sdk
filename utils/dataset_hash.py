#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""目录哈希工具：用于检测 original_dataset/{dataset_id} 目录是否发生变化"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from .path_handler import PathHandler, ensure_local_path

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024
ORIGINAL_DATASET_HASH_SCHEMA_VERSION = 'original-images-json-v2'


def _hash_file_content(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compute_directory_hash(directory: str) -> Optional[str]:
    """计算目录的稳定哈希。

    遍历目录下所有文件(递归),按相对路径排序后,将
    "相对路径\\t文件大小\\t文件内容sha256" 拼接成一份清单,
    最终对清单做一次 sha256,得到 64 位十六进制摘要。

    任意文件新增、删除、重命名、内容修改(包括 .json 内部字符变动)
    都会产生不同的哈希。

    Args:
        directory: 目录的本地绝对路径或可被 ensure_local_path 解析的路径。

    Returns:
        64 位十六进制哈希字符串。目录不存在或非目录时返回 None。
    """
    if not directory:
        return None

    local_dir = Path(ensure_local_path(directory))
    if not local_dir.exists() or not local_dir.is_dir():
        return None

    file_entries = []
    for file_path in local_dir.rglob('*'):
        if not file_path.is_file():
            continue
        try:
            rel_path = file_path.relative_to(local_dir).as_posix()
            size = file_path.stat().st_size
            content_hash = _hash_file_content(file_path)
            file_entries.append(f"{rel_path}\t{size}\t{content_hash}")
        except (OSError, ValueError) as e:
            logger.warning(f"计算文件哈希失败,跳过: {file_path}, error={e}")
            continue

    file_entries.sort()
    manifest = '\n'.join(file_entries).encode('utf-8')
    return hashlib.sha256(manifest).hexdigest()


def _compute_selected_subdirs_hash(root_dir: Path, subdirs, schema_version: str) -> Optional[str]:
    if not root_dir.exists() or not root_dir.is_dir():
        return None

    file_entries = [f"schema\t{schema_version}"]
    for subdir in subdirs:
        subdir_path = root_dir / subdir
        if not subdir_path.exists() or not subdir_path.is_dir():
            continue

        for file_path in subdir_path.rglob('*'):
            if not file_path.is_file():
                continue
            try:
                rel_path = file_path.relative_to(root_dir).as_posix()
                size = file_path.stat().st_size
                content_hash = _hash_file_content(file_path)
                file_entries.append(f"{rel_path}\t{size}\t{content_hash}")
            except (OSError, ValueError) as e:
                logger.warning(f"计算文件哈希失败,跳过: {file_path}, error={e}")
                continue

    file_entries[1:] = sorted(file_entries[1:])
    manifest = '\n'.join(file_entries).encode('utf-8')
    return hashlib.sha256(manifest).hexdigest()


def compute_original_dataset_hash(dataset_id, schema_version: str = ORIGINAL_DATASET_HASH_SCHEMA_VERSION) -> Optional[str]:
    """计算 original_dataset/{dataset_id} 原始 images/json 内容哈希。

    只纳入原始图片和 JSON 标注目录，排除 labels/classes/dataset_analysis
    等派生产物，避免 json2txt 或统计流程改变 hash 语义。
    """
    if dataset_id is None:
        return None
    dataset_dir = PathHandler.build_data_path('original_dataset', str(dataset_id))
    root_dir = Path(ensure_local_path(dataset_dir))
    return _compute_selected_subdirs_hash(
        root_dir,
        subdirs=(
            'train/images',
            'train/json',
            'train/jsons',
            'valid/images',
            'valid/json',
            'valid/jsons',
            'original_train_data/images',
            'original_train_data/json',
            'original_train_data/jsons',
        ),
        schema_version=schema_version,
    )
