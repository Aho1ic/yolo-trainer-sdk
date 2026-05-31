#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集标注目录路径工具。"""

from pathlib import Path

ANNOTATION_DIR_NAMES = ('json', 'jsons')


def resolve_annotation_dir(base_dir: Path) -> Path:
    """返回本地已存在的标注目录，兼容 json/jsons。"""
    base_dir = Path(base_dir)
    for dir_name in ANNOTATION_DIR_NAMES:
        candidate = base_dir / dir_name
        if candidate.exists():
            return candidate
    return base_dir / ANNOTATION_DIR_NAMES[0]
