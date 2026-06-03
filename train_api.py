#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练平台总接口
"""

from flask import Flask, request, jsonify, send_file
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("flask_cors未安装")
import pymysql
import logging
import os
import sys
import subprocess
import json
import shutil
import random
import shlex
import tempfile
from io import BytesIO
from pathlib import Path
from datetime import datetime
import threading
import time
import glob
import csv
import hashlib
import traceback
import urllib.error
import urllib.request
from collections import Counter
import torch
from ultralytics import YOLO
from config import TrainerConfig
from utils.json2txt import UnifiedJsonConverter
from utils.create_dataset_yaml import parse_labels
from utils import DatasetDetailAnalyzer
from utils.path_handler import LEGACY_STORAGE_PREFIXES, PathHandler, ensure_local_path
from utils.dataset_hash import ORIGINAL_DATASET_HASH_SCHEMA_VERSION, compute_original_dataset_hash
from utils.annotation_paths import resolve_annotation_dir
from utils.pose_labels import assign_points_to_rectangles, infer_shape_type as infer_pose_shape_type, sort_keypoint_labels
from utils.predict import Predict
from utils.snowflake import SnowflakeIDGenerator

# 导入新模块
from services.exceptions import BusinessError, ValidationError, NotFoundError
from api.validators import (
    validate_dataset_id as _validate_dataset_id,
    validate_task_id as _validate_task_id,
    validate_positive_id as _validate_positive_id,
    validate_status as _validate_status,
    sanitize_string as _sanitize_string,
    sanitize_filename as _sanitize_filename,
    validate_json_body as _validate_json_body
)

# 尝试导入 SQLAlchemy
try:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# 导入数据库基类（连接池管理）
from database.manager import DatabaseManager as BaseDatabaseManager


snowflake_generator = SnowflakeIDGenerator(datacenter_id=1, machine_id=1)

# 配置日志（使用统一的日志模块）
from utils.logger import setup_logger
logger = setup_logger('trainer.api', log_file='train_api.log')

app = Flask(__name__)

# 跨域
if CORS_AVAILABLE:
    CORS(app, resources={r"/algorithm/*": {"origins": "*"}})


def safe_rmtree(path, max_retries=3, retry_delay=0.2):
    """稳健删除目录，兼容并发场景下的 ENOENT/ENOTEMPTY 抖动"""
    path = Path(path)
    if not path.exists():
        return

    last_error = None
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            last_error = e
            if e.errno in (2, 39) and attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            break

    shutil.rmtree(path, ignore_errors=True)
    if path.exists() and last_error:
        raise last_error


def validate_dataset_id(dataset_id) -> int:
    """验证并转换 dataset_id 为整数，防止路径遍历攻击"""
    return _validate_dataset_id(dataset_id)


def validate_task_id(task_id) -> int:
    """验证并转换 task_id 为整数"""
    return _validate_task_id(task_id)


def validate_positive_id(value, field_name: str = "id") -> int:
    """验证并转换业务 ID 为正整数，防止路径遍历攻击"""
    return _validate_positive_id(value, field_name)


def sanitize_string(value: str, max_length: int = 255) -> str:
    """清理字符串输入，移除潜在危险字符"""
    return _sanitize_string(value, max_length)


def handle_api_exception(e: Exception, context: str = "") -> tuple:
    """
    统一的 API 异常处理函数

    Args:
        e: 异常对象
        context: 上下文描述

    Returns:
        Flask jsonify 响应元组
    """
    # 子类必须在父类之前检查
    if isinstance(e, ValidationError):
        logger.warning(f"[验证异常] {context}: {e.message}")
        return jsonify(e.to_dict()), 400

    if isinstance(e, NotFoundError):
        logger.warning(f"[未找到] {context}: {e.message}")
        return jsonify(e.to_dict()), 404

    if isinstance(e, BusinessError):
        logger.warning(f"[业务异常] {context}: {e.message}")
        return jsonify(e.to_dict()), e.status_code

    # 系统异常：记录详细信息，返回通用错误消息
    error_id = hashlib.md5(str(time.time()).encode(), usedforsecurity=False).hexdigest()[:8]
    logger.error(f"[系统异常] {context} (error_id={error_id}): {str(e)}")
    logger.error(traceback.format_exc())

    return jsonify({
        "code": 1,
        "data": {
            "message": "服务器内部错误，请稍后重试",
            "error_id": error_id
        }
    }), 500


DATASET_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
DATASET_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpg', '.mpeg', '.m4v'}


def prepare_split_dirs_for_original_analysis(original_data_address):
    """为原始数据集统计准备 train/valid 目录"""
    local_original_dir = Path(ensure_local_path(original_data_address))
    local_parent_dir = local_original_dir.parent
    return local_parent_dir / 'train', local_parent_dir / 'valid', local_original_dir


def merge_draw_types(draw_types):
    """合并多个 split 的标注类型"""
    normalized_types = {draw_type for draw_type in draw_types if draw_type}
    if not normalized_types:
        return 'rectangle'
    # pose 数据集（rectangle + point）整体视为 point 类型，与训练侧一致
    if 'point' in normalized_types:
        return 'point'
    if 'mix' in normalized_types or len(normalized_types) > 1:
        return 'mix'
    return next(iter(normalized_types))


def aggregate_original_analysis_results(train_result, valid_result, analysis_output_dir):
    """聚合 train/valid 分析结果，避免生成 merged_data 中间目录"""
    label_counter = Counter()
    label_order = []

    def update_label_counter(result):
        for label_name, count in result.get('tag_num', {}).items():
            if label_name not in label_counter:
                label_order.append(label_name)
            label_counter[label_name] += count

    update_label_counter(train_result)
    update_label_counter(valid_result)

    tag_sum = sum(label_counter.values())
    tag_percentage = {}
    if tag_sum > 0:
        for label_name in label_order:
            percentage = round((label_counter[label_name] / tag_sum) * 100, 2)
            tag_percentage[label_name] = f"{percentage}%"

    correspondence_info = {
        'json_without_image': [
            f"train/{name}" for name in train_result.get('correspondence_check', {}).get('json_without_image', [])
        ] + [
            f"valid/{name}" for name in valid_result.get('correspondence_check', {}).get('json_without_image', [])
        ],
        'image_without_json': [
            f"train/{name}" for name in train_result.get('correspondence_check', {}).get('image_without_json', [])
        ] + [
            f"valid/{name}" for name in valid_result.get('correspondence_check', {}).get('image_without_json', [])
        ]
    }
    correspondence_info['perfect_match'] = (
        not correspondence_info['json_without_image'] and
        not correspondence_info['image_without_json']
    )

    analysis_output_dir = Path(analysis_output_dir)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'dataset_path': str(analysis_output_dir),
        'sample_num': train_result.get('sample_num', 0) + valid_result.get('sample_num', 0),
        'annotation_num': train_result.get('annotation_num', 0) + valid_result.get('annotation_num', 0),
        'labels': json.dumps(label_order, ensure_ascii=False),
        'label_num': len(label_order),
        'tag_num': dict(label_counter),
        'tag_sum': tag_sum,
        'tag_percentage': tag_percentage,
        'draw_type': merge_draw_types([
            train_result.get('draw_type'),
            valid_result.get('draw_type')
        ]),
        'correspondence_check': correspondence_info
    }

    result_file = analysis_output_dir / 'dataset_analysis.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"聚合分析结果已保存到: {result_file}")

    return result


def analyze_original_dataset_from_splits(original_data_address):
    """按 train/valid 两个 split 直接聚合原始数据集统计"""
    local_train_dir, local_valid_dir, analysis_output_dir = prepare_split_dirs_for_original_analysis(original_data_address)

    if not local_train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在: {local_train_dir}")
    if not local_valid_dir.exists():
        raise FileNotFoundError(f"验证集目录不存在: {local_valid_dir}")

    logger.info(
        f"原始数据集分析目录: train={local_train_dir}, valid={local_valid_dir}, output={analysis_output_dir}"
    )

    train_result = DatasetDetailAnalyzer(str(local_train_dir)).analyze_dataset()
    valid_result = DatasetDetailAnalyzer(str(local_valid_dir)).analyze_dataset()

    return aggregate_original_analysis_results(train_result, valid_result, analysis_output_dir)


def resolve_dataset_dir_for_conversion(dataset_path, split_name):
    """解析 json2txt 所需的数据目录"""
    return materialize_dataset_dir_for_conversion(dataset_path, split_name)

trainer_config = TrainerConfig()
DB_CONFIG = trainer_config.get_db_config()
DATA_ROOT = trainer_config.get_data_root()
PROCESS_CALLBACK_BASE_URL = trainer_config.get_process_callback_base_url()
PROCESS_CALLBACK_PATH = '/system/trainTask/open/process-callback'
PROCESS_CALLBACK_TIMEOUT = 10
DATASET_BUILD_WATCHDOG_INTERVAL = 60
DATASET_BUILD_WATCHDOG_IDLE_SECONDS = {
    'build': 600,
    'configure': 300,
}
DATASET_BUILD_PROGRESS_FILE_INTERVAL = 100
DATASET_BUILD_PROGRESS_SECONDS_INTERVAL = 5
STORAGE_OBJECT_KEY_PREFIX = os.environ.get('TRAINER_STORAGE_OBJECT_KEY_PREFIX', 'algorithm/trainer')
LABEL_TASK_STORAGE_DB_PREFIX = os.environ.get('LABEL_TASK_STORAGE_DB_PREFIX', '/app/storage/task')
LABEL_TASK_STORAGE_LOCAL_PREFIX = os.environ.get('LABEL_TASK_STORAGE_LOCAL_PREFIX', '/volume2/task')
AUTO_LABEL_REDIS_LOCK_TTL_SECONDS = int(os.environ.get('AUTO_LABEL_REDIS_LOCK_TTL_SECONDS', '604800'))
MODEL_CONVERT_CONTAINER_ID = os.environ.get('MODEL_CONVERT_CONTAINER_ID', '96350d6935d8')
MODEL_CONVERT_CONTAINER_WORKDIR = os.environ.get('MODEL_CONVERT_CONTAINER_WORKDIR', '/workspace')
MODEL_CONVERT_SUPPORTED_PLATFORMS = {'bm1684', 'bm1684x', 'rk3588'}
MODEL_CONVERT_FORMAT_BY_PLATFORM = {
    'bm1684': 'bmodel',
    'bm1684x': 'bmodel',
    'rk3588': 'rknn',
}
RKNN_CONVERT_PYTHON = os.environ.get('RKNN_CONVERT_PYTHON', sys.executable)
MODEL_CONVERT_STATUS_PENDING = 0
MODEL_CONVERT_STATUS_RUNNING = 1
MODEL_CONVERT_STATUS_SUCCESS = 2
MODEL_CONVERT_STATUS_FAILED = 3

# 线程安全的全局变量
_running_processes_lock = threading.Lock()
_preloaded_models_lock = threading.Lock()
_model_info_lock = threading.Lock()
_model_convert_tasks_lock = threading.Lock()

running_processes = {}
preloaded_models = {}
model_info = {}
model_convert_tasks = {}


def build_storage_path(*parts):
    """构造本地数据路径，例如 /data/Sucai1/algorithm/trainer/train_data/123"""
    return PathHandler.build_data_path(*parts)


def build_storage_local_path(*parts, require_exists=False):
    """构造本地数据路径，返回 Path 对象，支持 require_exists 参数"""
    path_str = PathHandler.resolve_local_path(*parts, require_exists=require_exists)
    return Path(path_str)


def normalize_address_for_db(path: str) -> str:
    """返回本地路径（兼容旧的远程 URI）"""
    return ensure_local_path(path)


def normalize_storage_path_for_db(path: str):
    """将本地绝对路径转换为数据库中的 algorithm/trainer 相对路径"""
    if path is None:
        return None

    path_str = str(path).strip().replace('\\', '/')
    if not path_str:
        return None

    for legacy_prefix in LEGACY_STORAGE_PREFIXES:
        if path_str.startswith(legacy_prefix):
            path_str = path_str[len(legacy_prefix):].lstrip('/')
            break

    prefix = str(STORAGE_OBJECT_KEY_PREFIX).strip('/').replace('\\', '/')
    if path_str == prefix or path_str.startswith(f'{prefix}/'):
        return path_str.strip('/')

    prefix_index = path_str.find(prefix)
    if prefix_index >= 0:
        return path_str[prefix_index:].strip('/')

    if path_str.startswith('/'):
        return path_str.lstrip('/')

    return f"{prefix}/{path_str.lstrip('/')}"


def resolve_storage_path_to_local(path: str):
    """normalize_storage_path_for_db 的逆操作：把数据库中保存的相对路径还原为本地绝对路径"""
    if path is None:
        return None

    path_str = str(path).strip()
    if not path_str:
        return path_str

    return ensure_local_path(path_str)


def _resolve_path(path: str) -> Path:
    """将相对路径解析为基于数据根目录的绝对路径"""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(PathHandler.get_data_root()) / p


def list_local_files(directory: str, suffixes=None):
    """列出本地目录下的文件（仅顶层，不递归），返回带目录前缀的相对路径"""
    dir_path = _resolve_path(directory)
    if not dir_path.exists():
        return []

    data_root = Path(PathHandler.get_data_root())
    results = []
    for file_path in dir_path.iterdir():
        if not file_path.is_file():
            continue
        if suffixes and file_path.suffix.lower() not in suffixes:
            continue
        # 返回相对于数据根目录的完整路径
        results.append(str(file_path.relative_to(data_root)))
    return sorted(results)


def read_local_file(file_path: str) -> bytes:
    """读取本地文件"""
    return _resolve_path(file_path).read_bytes()


def read_local_json(file_path: str):
    """读取本地 JSON 文件"""
    return json.loads(_resolve_path(file_path).read_text(encoding='utf-8'))


def write_local_file(file_path: str, data: bytes):
    """写入本地文件"""
    p = _resolve_path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def write_local_json(file_path: str, data):
    """写入本地 JSON 文件"""
    p = _resolve_path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def delete_local_file(file_path: str):
    """删除本地文件"""
    p = _resolve_path(file_path)
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        logger.info(f"文件删除成功: {file_path}")


def copy_file_to_storage(local_path: Path, target_path: str):
    """复制文件到存储目录"""
    local_path = Path(local_path)
    target = _resolve_path(str(target_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(local_path), str(target))
    logger.debug(f"文件复制成功: {local_path} -> {target}")
    return str(target)


def file_sha256(local_path: Path) -> str:
    digest = hashlib.sha256()
    with open(local_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def copy_directory_to_storage(
    local_dir: Path,
    target_dir: str,
    progress_callback=None,
    progress_file_interval: int = DATASET_BUILD_PROGRESS_FILE_INTERVAL,
    progress_seconds_interval: int = DATASET_BUILD_PROGRESS_SECONDS_INTERVAL,
):
    """复制目录到存储目录"""
    local_dir = Path(local_dir)
    target = _resolve_path(str(target_dir))
    success_count = 0
    fail_count = 0
    failed_files = []
    files_to_copy = [file_path for file_path in local_dir.rglob('*') if file_path.is_file()]
    total_files = len(files_to_copy)
    processed_files = 0
    last_progress_time = time.time()

    if progress_callback:
        progress_callback(processed_files, total_files)

    for file_path in files_to_copy:
        relative_path = file_path.relative_to(local_dir)
        target_file = target / relative_path
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(file_path), str(target_file))
            success_count += 1
        except Exception as e:
            logger.error(f"复制文件失败: {file_path} -> {target_file}, error={e}")
            fail_count += 1
            failed_files.append(str(file_path))
        finally:
            processed_files += 1
            now = time.time()
            should_report = (
                progress_callback
                and (
                    processed_files == total_files
                    or processed_files % progress_file_interval == 0
                    or now - last_progress_time >= progress_seconds_interval
                )
            )
            if should_report:
                progress_callback(processed_files, total_files)
                last_progress_time = now

    return success_count, fail_count, failed_files


def persist_directory_to_storage(local_dir, storage_prefix, progress_callback=None):
    """将本地构建产物持久化到存储路径，安全处理源/目标重叠的情况。

    本地存储迁移后，构建产物通常已经位于目标路径（或其子目录）。旧逻辑沿用
    远程对象存储时代的"先 delete_storage_prefix 清空目标，再从同一目录 upload"
    模式，在本地架构下会先把刚生成的数据删掉，再从已被清空的目录拷贝，结果是
    数据集/推理结果被悄悄清空且不报错。这里按源与目标的实际关系分别处理：
    - 源与目标为同一路径：数据已就位，直接统计文件数返回，不做任何删除/拷贝。
    - 源是目标的子目录：把子目录内容上提到目标，再移除空的子目录。
    - 源与目标无重叠：保留"清空目标再拷贝"的旧语义（兼容跨目录场景）。
    """
    src = Path(local_dir).resolve()
    target = _resolve_path(str(storage_prefix)).resolve()

    def _count_and_report():
        files = [p for p in target.rglob('*') if p.is_file()]
        if progress_callback:
            progress_callback(len(files), len(files))
        return len(files), 0, []

    if src == target:
        return _count_and_report()

    if target in src.parents:
        # 源在目标内部（如 YOLO 把结果写进了 results/predict 子目录）：
        # 上提内容而非先删目标，否则会连源一起删掉。
        target.mkdir(parents=True, exist_ok=True)
        for child in list(src.iterdir()):
            dest = target / child.name
            if dest.exists():
                if dest.is_dir():
                    safe_rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(child), str(dest))
        try:
            src.rmdir()
        except OSError:
            pass
        return _count_and_report()

    if target.exists():
        safe_rmtree(target)
    return copy_directory_to_storage(src, str(target), progress_callback=progress_callback)


def parse_storage_relative_path(path):
    """将存储地址解析为本地绝对路径。

    历史上这里返回"去掉前导斜杠的相对键"，但本地存储迁移后该形态既不是
    合法的数据根相对键，也不是绝对路径：传给 _resolve_path 会被二次拼接到
    数据根目录下（/data/.../trainer/data/.../trainer/...），导致删除/复制
    目标错位、旧结果残留。

    现统一返回本地绝对路径（DB 相对键/旧远程 URI <-> 本地绝对路径），
    与 delete_storage_prefix / upload_directory_to_storage 等同走 _resolve_path
    的语义保持一致。
    """
    return ensure_local_path(str(path or '').strip())


read_storage_object_bytes = read_local_file
read_storage_json_object = read_local_json
put_storage_json_object = write_local_json
delete_storage_object = delete_local_file
list_storage_objects = list_local_files
upload_file_to_storage = copy_file_to_storage
upload_directory_to_storage = copy_directory_to_storage


def copy_directory_files_parallel(
    src_dir: Path,
    dest_dir: Path,
    max_workers: int = 16,
    progress_callback=None,
    progress_file_interval: int = DATASET_BUILD_PROGRESS_FILE_INTERVAL,
    progress_seconds_interval: int = DATASET_BUILD_PROGRESS_SECONDS_INTERVAL,
) -> int:
    """并行复制目录中的一级文件"""
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [file_path for file_path in src_dir.iterdir() if file_path.is_file()]
    total_files = len(files_to_copy)
    if progress_callback:
        progress_callback(0, total_files)
    if not files_to_copy:
        return 0

    def copy_file(file_path: Path) -> bool:
        target_file = dest_dir / file_path.name
        try:
            shutil.copy2(str(file_path), str(target_file))
            return True
        except Exception as e:
            logger.error(f"复制文件失败: {file_path} -> {target_file}, error={e}")
            return False

    success_count = 0
    processed_files = 0
    last_progress_time = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_file, file_path) for file_path in files_to_copy]
        for future in as_completed(futures):
            processed_files += 1
            if future.result():
                success_count += 1
            now = time.time()
            should_report = (
                progress_callback
                and (
                    processed_files == total_files
                    or processed_files % progress_file_interval == 0
                    or now - last_progress_time >= progress_seconds_interval
                )
            )
            if should_report:
                progress_callback(processed_files, total_files)
                last_progress_time = now

    return success_count


def build_storage_object_key(*parts) -> str:
    """构造本地数据根目录下的相对路径键"""
    normalized = [str(part).strip().strip('/') for part in parts if str(part).strip()]
    return '/'.join(normalized)


def build_storage_db_object_key(*parts) -> str:
    """构造写入数据库的对象键，例如 algorithm/trainer/original_dataset/123/train/images/a.jpg"""
    normalized = [str(STORAGE_OBJECT_KEY_PREFIX).strip().strip('/')]
    normalized.extend(str(part).strip().strip('/') for part in parts if str(part).strip())
    return '/'.join(part for part in normalized if part)


def build_process_callback_url(base_url: str) -> str:
    """构造数据处理完成回调地址"""
    base_url = str(base_url or '').strip().rstrip('/')
    if not base_url:
        return ''
    return f"{base_url}{PROCESS_CALLBACK_PATH}"


def send_process_callback(dataset_id, success, message=None):
    """向 uav-server 回调数据集处理结果"""
    callback_url = build_process_callback_url(PROCESS_CALLBACK_BASE_URL)
    if not callback_url:
        logger.warning(f"未配置数据处理回调地址，跳过回调: dataset_id={dataset_id}, success={success}")
        return False

    payload = {
        'datasetId': int(dataset_id),
        'success': bool(success),
        'message': None if success else (str(message or '')[:500] or None)
    }
    request_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(
        callback_url,
        data=request_data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    try:
        with urllib.request.urlopen(request, timeout=PROCESS_CALLBACK_TIMEOUT) as response:
            response_text = response.read().decode('utf-8').strip()
            status_code = response.getcode()

        if status_code != 200:
            raise RuntimeError(f"HTTP {status_code}")

        response_data = json.loads(response_text) if response_text else {}
        if response_data.get('code') != 0:
            raise RuntimeError(f"callback code={response_data.get('code')}, body={response_text or 'empty'}")

        logger.info(
            f"数据处理回调成功: dataset_id={dataset_id}, success={success}, callback_url={callback_url}"
        )
        return True
    except Exception as e:
        logger.error(
            f"数据处理回调失败: dataset_id={dataset_id}, success={success}, callback_url={callback_url}, error={e}"
        )
        return False


def verify_storage_object_matches_file(local_path, target_path):
    """本地模式下校验文件存在与大小"""
    local_path = _resolve_path(str(local_path))
    target = _resolve_path(str(target_path)) if target_path else local_path
    if not target.exists():
        raise RuntimeError(f"文件不存在: {target}")
    size = target.stat().st_size
    digest = file_sha256(target)
    logger.info(f"文件校验成功: path={target}, size={size}, sha256={digest}")
    return {'size': size, 'sha256': digest}


def download_storage_prefix(prefix: str, local_dir: Path, suffixes=None):
    """兼容旧接口：本地模式下数据已在本地，只列出文件数"""
    source_dir = _resolve_path(prefix)
    if not source_dir.exists():
        return 0
    files = list_local_files(str(source_dir), suffixes=suffixes)
    return len(files)


def delete_storage_prefix(prefix: str):
    """删除本地目录前缀下的所有文件"""
    dir_path = _resolve_path(prefix)
    if dir_path.exists():
        shutil.rmtree(dir_path)
        logger.info(f"目录删除成功: {prefix}")
    return 0


def materialize_dataset_dir_for_conversion(dataset_path, split_name):
    """json2txt 所需的数据目录"""
    local_dir = ensure_local_path(dataset_path)
    local_path = Path(local_dir)
    json_dir = local_path / 'json'
    if not json_dir.exists():
        raise FileNotFoundError(f"{split_name} json目录不存在: {json_dir}")
    (local_path / 'images').mkdir(parents=True, exist_ok=True)
    return str(local_path)


def materialize_dataset_dir_for_create(dataset_path, split_name):
    """构建训练集需要真实 images 和 labels 文件"""
    local_dir = ensure_local_path(dataset_path)
    local_path = Path(local_dir)
    images_dir = local_path / 'images'
    labels_dir = local_path / 'labels'
    if not images_dir.exists():
        raise FileNotFoundError(f"{split_name} images目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"{split_name} labels目录不存在: {labels_dir}")
    return str(local_path)


def upload_converted_labels(local_dataset_dir, dataset_path):
    """复制 json2txt 生成的 labels/classes/class_mapping 到目标目录"""
    local_dataset_dir = Path(local_dataset_dir).resolve()
    target_dir = Path(ensure_local_path(dataset_path)).resolve()

    labels_dir = local_dataset_dir / 'labels'
    if labels_dir.exists():
        target_labels = target_dir / 'labels'
        if labels_dir != target_labels:
            if target_labels.exists():
                shutil.rmtree(target_labels)
            shutil.copytree(str(labels_dir), str(target_labels))
            logger.info(f"labels 目录复制完成: {labels_dir} -> {target_labels}")

    for file_name in ('classes.txt', 'class_mapping.json'):
        local_file = local_dataset_dir / file_name
        if local_file.exists():
            target_file = target_dir / file_name
            if local_file != target_file:
                shutil.copy2(str(local_file), str(target_file))


def get_storage_image_size(file_path: str):
    """读取图片尺寸"""
    try:
        from PIL import Image

        with Image.open(_resolve_path(file_path)) as img:
            return img.size
    except Exception as e:
        logger.warning(f"读取图片尺寸失败: {file_path}, error={e}")
        return None


def get_image_size(image_path: Path):
    """读取图片宽高，优先使用 Pillow，失败后回退到 OpenCV"""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size
    except Exception as pil_error:
        try:
            import cv2

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("cv2.imread 返回空")
            height, width = image.shape[:2]
            return width, height
        except Exception as cv_error:
            raise RuntimeError(
                f"无法读取图片尺寸: {image_path}，PIL错误: {pil_error}，OpenCV错误: {cv_error}"
            )


def create_negative_sample_json(json_path: Path, image_path: Path):
    """为缺少标注的图片创建空 shapes 的负样本 JSON"""
    image_width, image_height = get_image_size(image_path)
    negative_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(negative_json, f, ensure_ascii=False, indent=2)


def list_dataset_images(images_dir: Path):
    """按 stem 收集图片文件，重复 stem 仅保留第一张并记录警告"""
    image_files = {}

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in DATASET_IMAGE_EXTENSIONS:
            continue

        stem = image_path.stem
        if stem in image_files:
            logger.warning(f"检测到同名图片 stem 冲突，忽略后续文件: {image_path}，保留: {image_files[stem]}")
            continue
        image_files[stem] = image_path

    return image_files


def infer_shape_type(shape):
    """推断单个标注的类型，兼容缺失 shape_type 的 JSON"""
    return infer_pose_shape_type(shape)


def detect_draw_type_from_shapes(shape_types):
    """根据一组 shape_type 推断 draw_type

    判断优先级：含 point => 'point'（pose 数据集），rectangle+polygon => 'mix'，
    单一 polygon => 'polygon'，否则 'rectangle'。
    """
    normalized_types = {shape_type for shape_type in shape_types if shape_type}
    if not normalized_types:
        return 'rectangle'
    if 'point' in normalized_types:
        return 'point'
    if 'polygon' in normalized_types and 'rectangle' in normalized_types:
        return 'mix'
    if 'polygon' in normalized_types:
        return 'polygon'
    return 'rectangle'


def format_percentage_dict(counter_dict, total_count):
    """将标签计数格式化为百分比字典"""
    percentage_dict = {}
    if total_count <= 0:
        return percentage_dict

    for label_name, count in counter_dict.items():
        percentage = round((count / total_count) * 100, 2)
        percentage_dict[label_name] = f"{percentage}%"
    return percentage_dict


def resolve_split_annotation_prefix(dataset_id, split_name):
    """返回 stats 使用的 split 标注目录，兼容 json/jsons。"""
    split_prefix = build_storage_object_key('original_dataset', str(dataset_id), split_name)
    candidates = [f"{split_prefix}/json", f"{split_prefix}/jsons"]

    listed_candidates = []
    for candidate in candidates:
        objects = list_storage_objects(candidate, suffixes={'.json'})
        if objects:
            return candidate, objects
        listed_candidates.append((candidate, objects))

    for candidate, objects in listed_candidates:
        if _resolve_path(candidate).exists():
            return candidate, objects

    return listed_candidates[0]


def repair_and_collect_split_stats(dataset_id, split_name, data_type):
    """基于本地路径修复单个 split 的 images/json 对应关系，并统计标签信息"""
    split_prefix = build_storage_object_key('original_dataset', str(dataset_id), split_name)
    images_prefix = f"{split_prefix}/images"
    json_prefix, json_object_list = resolve_split_annotation_prefix(dataset_id, split_name)

    image_objects = {}
    for object_name in list_storage_objects(images_prefix, suffixes=DATASET_IMAGE_EXTENSIONS):
        stem = Path(object_name).stem
        if stem in image_objects:
            logger.warning(f"检测到同名图片 stem 冲突，忽略后续文件: {object_name}，保留: {image_objects[stem]}")
            continue
        image_objects[stem] = object_name

    if not image_objects:
        raise FileNotFoundError(f"{split_name} images目录不存在或为空: {images_prefix}")

    json_objects = {
        Path(object_name).stem: object_name
        for object_name in json_object_list
    }

    removed_json_files = []
    created_negative_json_files = []
    failed_json_files = []

    for stem, json_object in list(json_objects.items()):
        if stem not in image_objects:
            delete_storage_object(json_object)
            removed_json_files.append(Path(json_object).name)
            json_objects.pop(stem, None)
            logger.info(f"删除无对应图片的JSON文件: {json_object}")

    for stem, image_object in image_objects.items():
        if stem in json_objects:
            continue
        negative_json_object = f"{json_prefix}/{stem}.json"
        image_size = get_storage_image_size(image_object) or (1, 1)
        image_width, image_height = image_size
        negative_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": Path(image_object).name,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }
        put_storage_json_object(negative_json_object, negative_json)
        json_objects[stem] = negative_json_object
        created_negative_json_files.append(Path(negative_json_object).name)
        logger.info(f"为缺少标注的图片生成负样本JSON文件: {image_object} -> {negative_json_object}")

    label_counter = Counter()
    label_order = []
    label_cover_map = {}
    shape_types = []
    # pose 相关：每个矩形对应的关键点标签集合 + 整个 split 出现过的关键点标签
    keypoint_labels_in_split = []  # 保留首次出现顺序

    parsed_entries = []
    split_has_point = False

    for stem in sorted(json_objects.keys()):
        json_object = json_objects[stem]
        try:
            data = read_storage_json_object(json_object)
        except Exception as e:
            logger.error(f"读取JSON文件失败: {json_object}, error={str(e)}")
            failed_json_files.append({'object': json_object, 'error': str(e)})
            continue

        shapes = data.get('shapes') or []
        assignment = assign_points_to_rectangles(shapes)
        parsed_entries.append((stem, json_object, shapes, assignment))
        shape_types.extend(assignment.shape_types)
        if 'point' in assignment.shape_types:
            split_has_point = True

        for kp_label in assignment.keypoint_labels:
            if kp_label not in keypoint_labels_in_split:
                keypoint_labels_in_split.append(kp_label)

        if assignment.unassigned_points:
            logger.warning(
                f"{json_object} 存在 {len(assignment.unassigned_points)} 个框外 point，"
                f"不计入 kpt_labels: {[point.label for point in assignment.unassigned_points[:5]]}"
            )
        if assignment.duplicate_points:
            logger.warning(
                f"{json_object} 存在 {len(assignment.duplicate_points)} 个重复 point，"
                f"统计时保留首个: {[point.label for point in assignment.duplicate_points[:5]]}"
            )

    for stem, json_object, shapes, assignment in parsed_entries:
        labels_in_file = []
        for shape in shapes:
            label_name = str(shape.get('label') or '').strip()
            if not label_name:
                continue

            shape_type = infer_shape_type(shape)

            # point 类型不计入"类别"统计，而是作为关键点标签收集
            if shape_type == 'point':
                continue

            # 一旦 split 中出现 point，整体按 pose 数据处理：仅 rectangle 作为目标类别。
            # polygon 等形状不参与 pose 类别统计，也不会进入 YOLO pose TXT。
            if split_has_point and shape_type != 'rectangle':
                continue

            if label_name not in label_counter:
                label_order.append(label_name)
            label_counter[label_name] += 1
            labels_in_file.append(label_name)

        if labels_in_file and stem in image_objects:
            object_key = build_storage_db_object_key(
                'original_dataset', str(dataset_id), split_name, 'images', Path(image_objects[stem]).name
            )
            for label_name in dict.fromkeys(labels_in_file):
                label_cover_map.setdefault(label_name, object_key)

    ordered_tag_num = {label_name: label_counter[label_name] for label_name in label_order}
    tag_sum = sum(ordered_tag_num.values())
    tag_percentage = format_percentage_dict(ordered_tag_num, tag_sum)

    label_rows = []
    for label_name in label_order:
        percentage_str = tag_percentage.get(label_name, '0%')
        proportion = round(float(percentage_str.replace('%', '')), 2) if percentage_str else 0.0
        label_rows.append({
            'label_name': label_name,
            'tag_num': ordered_tag_num[label_name],
            'proportion': proportion,
            'data_type': data_type,
            'object_key': label_cover_map.get(label_name)
        })

    return {
        'split_name': split_name,
        'split_dir': build_storage_path('original_dataset', str(dataset_id), split_name),
        'sample_num': len(image_objects),
        'annotation_num': len(json_objects),
        'tag_sum': tag_sum,
        'tag_num': ordered_tag_num,
        'tag_percentage': tag_percentage,
        'draw_type': detect_draw_type_from_shapes(shape_types),
        'label_order': label_order,
        'label_rows': label_rows,
        'keypoint_labels': keypoint_labels_in_split,
        'repair': {
            'removed_json_files': removed_json_files,
            'created_negative_json_files': created_negative_json_files,
            'failed_json_files': failed_json_files
        }
    }


def normalize_label_mapping(label_mapping):
    """标准化标签映射，空字符串/None 都代表删除该标签

    value 支持两种写法：
    - 字符串/None：匹配所有 shape_type 下的同名 label
    - dict：{"new_label": <str|None>, "shape_type": <str|None>}
        new_label 缺失/None/空串 → 删除
        shape_type 缺失/None → 匹配所有形状
        shape_type 给值（point/rectangle/polygon 等）→ 只匹配 shape.shape_type 等于该值的形状
    """
    if not isinstance(label_mapping, dict) or not label_mapping:
        raise ValueError(
            "label_mapping 必须是非空对象，例如 {'旧标签': '新标签', '待删除标签': '', "
            "'关键点A': {'new_label': '关键点B', 'shape_type': 'point'}}"
        )

    normalized_mapping = {}
    for old_label, raw_value in label_mapping.items():
        normalized_old_label = str(old_label).strip()
        if not normalized_old_label:
            raise ValueError("label_mapping 中存在空的原标签名")

        if isinstance(raw_value, dict):
            new_label_raw = raw_value.get('new_label')
            shape_type_raw = raw_value.get('shape_type')
        elif raw_value is None or isinstance(raw_value, (str, int, float)):
            new_label_raw = raw_value
            shape_type_raw = None
        else:
            raise ValueError(
                f"label_mapping['{normalized_old_label}'] 的值类型不支持，仅支持字符串/None/对象"
            )

        if new_label_raw is None:
            target_label = None
        else:
            normalized_new_label = str(new_label_raw).strip()
            target_label = normalized_new_label or None

        if shape_type_raw is None:
            shape_type_filter = None
        else:
            normalized_shape_type = str(shape_type_raw).strip().lower()
            shape_type_filter = normalized_shape_type or None

        normalized_mapping[normalized_old_label] = {
            'target_label': target_label,
            'shape_type_filter': shape_type_filter,
        }

    return normalized_mapping


def batch_modify_labels_in_split(dataset_id, split_name, label_mapping):
    """基于本地路径批量修改单个 split/json 目录中的标签，可按 shape_type 过滤"""
    json_prefix = build_storage_object_key('original_dataset', str(dataset_id), split_name, 'json')
    json_objects = list_storage_objects(json_prefix, suffixes={'.json'})
    operation_stats = {}
    for old_label, config in label_mapping.items():
        target_label = config['target_label']
        operation_stats[old_label] = {
            'target_label': target_label,
            'shape_type_filter': config['shape_type_filter'],
            'action': 'delete' if target_label is None else 'rename',
            'matched_shapes': 0,
        }

    result = {
        'split_name': split_name,
        'json_prefix': json_prefix,
        'total_files': len(json_objects),
        'modified_files': 0,
        'renamed_shapes': 0,
        'deleted_shapes': 0,
        'operation_stats': operation_stats,
        'failed_files': []
    }

    for json_object in json_objects:
        try:
            data = read_storage_json_object(json_object)

            original_shapes = data.get('shapes') or []
            updated_shapes = []
            file_modified = False

            for shape in original_shapes:
                if not isinstance(shape, dict):
                    updated_shapes.append(shape)
                    continue

                shape_label = str(shape.get('label') or '').strip()
                if not shape_label or shape_label not in label_mapping:
                    updated_shapes.append(shape)
                    continue

                config = label_mapping[shape_label]
                shape_type_filter = config['shape_type_filter']
                shape_type_value = str(shape.get('shape_type') or '').strip().lower()
                if shape_type_filter is not None and shape_type_value != shape_type_filter:
                    updated_shapes.append(shape)
                    continue

                operation_stats[shape_label]['matched_shapes'] += 1
                target_label = config['target_label']

                if target_label is None:
                    result['deleted_shapes'] += 1
                    file_modified = True
                    continue

                if target_label != shape_label:
                    shape['label'] = target_label
                    result['renamed_shapes'] += 1
                    file_modified = True

                updated_shapes.append(shape)

            if file_modified:
                data['shapes'] = updated_shapes
                put_storage_json_object(json_object, data)
                result['modified_files'] += 1

        except Exception as e:
            logger.error(f"批量修改标签失败: object={json_object}, error={str(e)}")
            result['failed_files'].append({
                'file': json_object,
                'error': str(e)
            })

    result['failed_count'] = len(result['failed_files'])
    result['unmatched_labels'] = [
        old_label for old_label, stats in operation_stats.items()
        if stats['matched_shapes'] == 0
    ]

    return result


def collect_dataset_stats_result(dataset_id):
    """执行数据集统计并同步更新数据库"""
    train_stats = repair_and_collect_split_stats(dataset_id, 'train', 1)
    val_stats = repair_and_collect_split_stats(dataset_id, 'valid', 2)

    combined_label_counter = Counter()
    combined_label_order = []
    for split_stats in (train_stats, val_stats):
        for label_name, count in split_stats['tag_num'].items():
            if label_name not in combined_label_counter:
                combined_label_order.append(label_name)
            combined_label_counter[label_name] += count

    class_mapping = {label_name: idx for idx, label_name in enumerate(combined_label_order)}
    total_tag_num = {label_name: combined_label_counter[label_name] for label_name in combined_label_order}
    total_tag_sum = sum(total_tag_num.values())
    total_tag_percentage = format_percentage_dict(total_tag_num, total_tag_sum)
    draw_type = merge_draw_types([train_stats['draw_type'], val_stats['draw_type']])

    sample_num = train_stats['sample_num'] + val_stats['sample_num']
    annotation_num = train_stats['annotation_num'] + val_stats['annotation_num']
    labels_json = json.dumps(combined_label_order, ensure_ascii=False)

    # pose 数据集：聚合两个 split 的关键点标签，按出现顺序得到规范关键点列表
    keypoint_labels = []
    for split_stats in (train_stats, val_stats):
        for kp_label in split_stats.get('keypoint_labels', []) or []:
            if kp_label not in keypoint_labels:
                keypoint_labels.append(kp_label)
    # 数字化关键点标签（"1","2","3"）按数值排序，保证 1<2<3 的稳定顺序
    keypoint_labels = sort_keypoint_labels(keypoint_labels)
    kpt_num = len(keypoint_labels)
    is_pose_dataset = draw_type == 'point'

    if not db_manager.update_dataset_class_mapping(dataset_id, class_mapping):
        raise RuntimeError("更新train_dataset.class失败")

    if not db_manager.update_divided_dataset_detail(
        dataset_id=dataset_id,
        train_tag_sum=train_stats['tag_sum'],
        val_tag_sum=val_stats['tag_sum'],
        train_tag_num=json.dumps(train_stats['tag_num'], ensure_ascii=False),
        val_tag_num=json.dumps(val_stats['tag_num'], ensure_ascii=False),
        train_tag_percentage=json.dumps(train_stats['tag_percentage'], ensure_ascii=False),
        val_tag_percentage=json.dumps(val_stats['tag_percentage'], ensure_ascii=False)
    ):
        raise RuntimeError("更新训练集/验证集统计字段失败")

    if not db_manager.update_dataset_detail(
        dataset_id=dataset_id,
        sample_num=sample_num,
        annotation_num=annotation_num,
        labels=labels_json,
        label_num=len(combined_label_order),
        tag_num=json.dumps(total_tag_num, ensure_ascii=False),
        draw_type=draw_type,
        tag_sum=total_tag_sum,
        tag_percentage=json.dumps(total_tag_percentage, ensure_ascii=False),
        kpt_num=kpt_num if is_pose_dataset else 0,
        kpt_labels=json.dumps(keypoint_labels if is_pose_dataset else [], ensure_ascii=False),
    ):
        raise RuntimeError("更新train_dataset聚合统计字段失败")

    label_rows = train_stats['label_rows'] + val_stats['label_rows']
    success_count, fail_count = db_manager.insert_dataset_labels(dataset_id, label_rows)
    if fail_count > 0:
        raise RuntimeError(
            f"写入train_dataset_label失败: success_count={success_count}, fail_count={fail_count}"
        )

    return {
        'dataset_id': dataset_id,
        'label_num': len(combined_label_order),
        'draw_type': draw_type,
        'class_mapping': class_mapping,
        'kpt_num': kpt_num,
        'keypoint_labels': keypoint_labels,
        'train_stats': {
            'sample_num': train_stats['sample_num'],
            'annotation_num': train_stats['annotation_num'],
            'tag_sum': train_stats['tag_sum'],
            'tag_num': train_stats['tag_num'],
            'tag_percentage': train_stats['tag_percentage'],
            'repair': train_stats['repair']
        },
        'valid_stats': {
            'sample_num': val_stats['sample_num'],
            'annotation_num': val_stats['annotation_num'],
            'tag_sum': val_stats['tag_sum'],
            'tag_num': val_stats['tag_num'],
            'tag_percentage': val_stats['tag_percentage'],
            'repair': val_stats['repair']
        },
        'label_rows': len(label_rows),
        'storage_paths': {
            'train': train_stats['split_dir'],
            'valid': val_stats['split_dir']
        }
    }


def sanitize_model_name(model_name, fallback='model'):
    """转换脚本对文件名和模型名较敏感，只允许ASCII安全字符"""
    raw_name = str(model_name or '').strip()
    if not raw_name:
        return fallback

    safe_chars = []
    for char in raw_name:
        if char.isascii() and (char.isalnum() or char in ('_', '-')):
            safe_chars.append(char)
        else:
            safe_chars.append('_')

    safe_name = ''.join(safe_chars).strip('_-')
    if not safe_name:
        safe_name = fallback
    if safe_name and safe_name[0].isdigit():
        safe_name = f"model_{safe_name}"
    return safe_name


def run_conversion_command(command, description, cwd=None, env=None):
    """执行转换命令并统一处理日志和错误"""
    logger.info(f"[模型转换] {description}: {shlex.join([str(item) for item in command])}")
    result = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True
    )

    if result.stdout:
        logger.info(f"[模型转换] {description} stdout:\n{result.stdout}")
    if result.stderr:
        logger.info(f"[模型转换] {description} stderr:\n{result.stderr}")

    if result.returncode != 0:
        error_detail = (result.stderr or result.stdout or '').strip()
        if len(error_detail) > 900:
            error_detail = error_detail[-900:]
        raise RuntimeError(f"{description}失败，返回码={result.returncode}: {error_detail}")

    return result


def run_docker_bash(command, description):
    """在 TPU Docker 容器中执行 bash 命令"""
    return run_conversion_command(
        ['docker', 'exec', MODEL_CONVERT_CONTAINER_ID, 'bash', '-lc', command],
        description
    )


def export_pt_to_onnx(pt_model_path: Path) -> Path:
    """使用标准 yolo export 将 pt 模型导出为 onnx，供 BM1684 等通用 ONNX 流程使用"""
    if not pt_model_path.exists():
        raise FileNotFoundError(f"PT模型不存在: {pt_model_path}")
    if pt_model_path.suffix.lower() != '.pt':
        raise ValueError(f"源模型必须是.pt文件: {pt_model_path}")

    onnx_path = pt_model_path.with_suffix('.onnx')
    if onnx_path.exists():
        onnx_path.unlink()

    command = [
        'yolo',
        'export',
        f'model={pt_model_path}',
        'format=onnx',
        'imgsz=640,640',
        'simplify=True',
        'dynamic=False',
        'half=False',
        'int8=False'
    ]
    run_conversion_command(command, 'YOLO PT导出ONNX')

    if not onnx_path.exists():
        raise FileNotFoundError(f"YOLO导出完成但未找到ONNX文件: {onnx_path}")

    return onnx_path


def export_pt_to_rknn_optimized_onnx(pt_model_path: Path) -> Path:
    """
    使用项目内 pt2onnx 的 RKNN 优化导出分支生成 ONNX。

    该分支对应 rknn_model_zoo 的 YOLO 后处理假设，会拆分输出头并增加 score-sum 分支；
    不能直接用官方 `yolo export format=onnx` 等价替换。
    """
    if not pt_model_path.exists():
        raise FileNotFoundError(f"PT模型不存在: {pt_model_path}")
    if pt_model_path.suffix.lower() != '.pt':
        raise ValueError(f"源模型必须是.pt文件: {pt_model_path}")

    repo_root = Path(__file__).parent.resolve()
    pt2onnx_dir = repo_root / 'pt2onnx'
    if not pt2onnx_dir.exists():
        raise FileNotFoundError(f"pt2onnx目录不存在: {pt2onnx_dir}")

    onnx_path = pt_model_path.with_suffix('.onnx')
    if onnx_path.exists():
        onnx_path.unlink()

    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH')
    env['PYTHONPATH'] = (
        f"{pt2onnx_dir}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath else str(pt2onnx_dir)
    )

    command = [
        sys.executable,
        '-c',
        'from ultralytics.cfg import entrypoint; entrypoint()',
        'export',
        f'model={pt_model_path}',
        'format=rknn',
        'imgsz=640,640',
        'dynamic=False',
        'half=False',
        'int8=False'
    ]
    run_conversion_command(command, 'RKNN优化ONNX导出', cwd=pt2onnx_dir, env=env)

    if not onnx_path.exists():
        raise FileNotFoundError(f"RKNN优化导出完成但未找到ONNX文件: {onnx_path}")

    return onnx_path


def prepare_pt_for_conversion(local_pt_path: Path, local_convert_dir: Path, safe_model_name: str) -> Path:
    """将源PT复制到本次转换工作目录，避免并发转换共享同一个 .onnx 输出路径"""
    local_convert_dir.mkdir(parents=True, exist_ok=True)
    target_pt_path = local_convert_dir / f'{safe_model_name}.pt'
    if local_pt_path.resolve() != target_pt_path.resolve():
        shutil.copy2(local_pt_path, target_pt_path)
    return target_pt_path


def resolve_dataset_train_images_dir(dataset_info):
    """根据数据集记录定位训练集图片目录"""
    if not dataset_info:
        return None

    candidate_dirs = []
    dataset_address = dataset_info.get('dataset_address')
    if dataset_address:
        try:
            local_dataset_dir = Path(ensure_local_path(dataset_address))
            candidate_dirs.extend([
                local_dataset_dir / 'datasets' / 'train' / 'images',
                local_dataset_dir / 'train' / 'images'
            ])
        except Exception as e:
            logger.warning(f"[模型转换] 下载dataset_address失败，跳过本地候选: {dataset_address}, error={str(e)}")

    original_data_address = dataset_info.get('original_data_address')
    if original_data_address:
        try:
            local_original_dir = Path(ensure_local_path(original_data_address))
            candidate_dirs.extend([
                local_original_dir / 'images',
                local_original_dir.parent / 'train' / 'images'
            ])
        except Exception as e:
            logger.warning(f"[模型转换] 下载original_data_address失败，跳过本地候选: {original_data_address}, error={str(e)}")

    for candidate_dir in candidate_dirs:
        if candidate_dir.exists() and any(
            item.is_file() and item.suffix.lower() in DATASET_IMAGE_EXTENSIONS
            for item in candidate_dir.iterdir()
        ):
            return candidate_dir

    return None


def get_train_image_object_prefix_candidates(dataset_info):
    """根据数据集记录推导训练图片本地路径候选"""
    candidates = []

    def add_prefix(prefix):
        prefix = str(prefix or '').strip()
        if not prefix:
            return
        # 绝对路径保持原样（_resolve_path 会原样返回）；相对键才去掉首尾斜杠
        if not prefix.startswith('/'):
            prefix = prefix.strip('/')
        if prefix and prefix not in candidates:
            candidates.append(prefix)

    def add_images_prefix(address):
        if address and PathHandler.is_legacy_remote_uri(address):
            add_prefix(f"{parse_storage_relative_path(address)}/images")

    add_images_prefix(dataset_info.get('original_train_data_address'))

    dataset_address = dataset_info.get('dataset_address')
    if dataset_address and PathHandler.is_legacy_remote_uri(dataset_address):
        dataset_object_path = parse_storage_relative_path(dataset_address)
        add_prefix(f"{dataset_object_path}/datasets/train/images")
        add_prefix(f"{dataset_object_path}/train/images")

    original_data_address = dataset_info.get('original_data_address')
    if original_data_address and PathHandler.is_legacy_remote_uri(original_data_address):
        original_object_path = parse_storage_relative_path(original_data_address)
        add_prefix(f"{original_object_path}/images")
        original_parent_path = original_object_path.rstrip('/').rsplit('/', 1)[0]
        add_prefix(f"{original_parent_path}/train/images")

    dataset_id = dataset_info.get('id') or dataset_info.get('dataset_id')
    if dataset_id:
        add_prefix(build_storage_object_key('original_dataset', str(dataset_id), 'train', 'images'))

    return candidates


def copy_random_train_images_from_storage(dataset_info, target_images_dir: Path, sample_count=5):
    """从训练图片本地路径候选中随机复制少量图片"""
    target_images_dir.mkdir(parents=True, exist_ok=True)
    last_errors = []

    for image_prefix in get_train_image_object_prefix_candidates(dataset_info):
        try:
            image_objects = list_storage_objects(image_prefix, suffixes=DATASET_IMAGE_EXTENSIONS)
        except Exception as e:
            last_errors.append(f"{image_prefix}: {str(e)}")
            logger.warning(f"[模型转换] 列出训练图片文件失败: path={image_prefix}, error={str(e)}")
            continue

        if not image_objects:
            logger.info(f"[模型转换] 训练图片目录为空: {image_prefix}")
            continue

        selected_objects = random.sample(image_objects, min(sample_count, len(image_objects)))
        copied_images = []
        used_names = set()

        for image_object in selected_objects:
            file_name = Path(image_object).name
            if file_name in used_names:
                file_name = f"{Path(image_object).stem}_{len(used_names)}{Path(image_object).suffix}"
            used_names.add(file_name)

            target_path = target_images_dir / file_name
            target_path.write_bytes(read_storage_object_bytes(image_object))
            copied_images.append(target_path)

        logger.info(
            f"[模型转换] 已从本地路径抽样复制{len(copied_images)}张训练图片: "
            f"{image_prefix} -> {target_images_dir}"
        )
        return copied_images

    if last_errors:
        logger.warning(f"[模型转换] 本地训练图片抽样失败详情: {last_errors[:5]}")
    return []


def copy_random_train_images(dataset_info, target_images_dir: Path, sample_count=5):
    """从训练集中随机抽取图片复制到转换工作目录"""
    copied_images = copy_random_train_images_from_storage(dataset_info, target_images_dir, sample_count=sample_count)
    if copied_images:
        return copied_images

    images_dir = resolve_dataset_train_images_dir(dataset_info)
    if not images_dir:
        raise FileNotFoundError("未找到可用于模型转换的训练集图片目录")

    image_files = [
        image_path for image_path in images_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in DATASET_IMAGE_EXTENSIONS
    ]
    if not image_files:
        raise FileNotFoundError(f"训练集图片目录为空: {images_dir}")

    selected_images = random.sample(image_files, min(sample_count, len(image_files)))
    target_images_dir.mkdir(parents=True, exist_ok=True)

    copied_images = []
    for image_path in selected_images:
        target_path = target_images_dir / image_path.name
        shutil.copy2(image_path, target_path)
        copied_images.append(target_path)

    logger.info(f"[模型转换] 已从本地训练集随机复制{len(copied_images)}张图片: {images_dir} -> {target_images_dir}")
    return copied_images


def run_bm1684_convert_pipeline(
    model_name,
    onnx_path: Path,
    dataset_info,
    local_convert_dir: Path,
    work_dir_name=None,
    chip='bm1684'
):
    """执行 BM1684/BM1684X 的 ONNX -> BModel 转换流程"""
    safe_model_name = sanitize_model_name(model_name)
    chip = str(chip or 'bm1684').strip().lower()
    if chip not in {'bm1684', 'bm1684x'}:
        raise ValueError(f"不支持的BModel转换芯片: {chip}")

    chip_label = chip.upper()
    container_dir_name = sanitize_model_name(work_dir_name or safe_model_name)
    local_convert_dir.mkdir(parents=True, exist_ok=True)

    local_onnx_path = local_convert_dir / 'best.onnx'
    shutil.copy2(onnx_path, local_onnx_path)

    local_images_dir = local_convert_dir / 'images'
    if local_images_dir.exists():
        safe_rmtree(local_images_dir)
    copied_images = copy_random_train_images(dataset_info, local_images_dir, sample_count=5)
    test_image_name = copied_images[0].name

    container_work_dir = f"{MODEL_CONVERT_CONTAINER_WORKDIR.rstrip('/')}/{container_dir_name}"
    container_images_dir = f"{container_work_dir}/images"

    run_docker_bash(
        f"rm -rf {shlex.quote(container_work_dir)} && mkdir -p {shlex.quote(container_images_dir)}",
        f'初始化{chip_label}容器转换目录'
    )
    run_conversion_command(
        ['docker', 'cp', str(local_onnx_path), f'{MODEL_CONVERT_CONTAINER_ID}:{container_work_dir}/best.onnx'],
        f'复制ONNX到{chip_label}容器'
    )
    run_conversion_command(
        ['docker', 'cp', f'{local_images_dir}/.', f'{MODEL_CONVERT_CONTAINER_ID}:{container_images_dir}/'],
        f'复制测试图片到{chip_label}容器'
    )

    transform_model_name = f'{safe_model_name}_1684x' if chip == 'bm1684x' else safe_model_name
    test_result_name = (
        f'{safe_model_name}_onnx_top_outputs.npz'
        if chip == 'bm1684x'
        else f'{safe_model_name}_1684_origin_top_outputs.npz'
    )
    mlir_name = f'{safe_model_name}.mlir'
    deploy_test_input_name = f'{transform_model_name}_in_f32.npz'
    bmodel_name = (
        f'{safe_model_name}_1684x_f32.bmodel'
        if chip == 'bm1684x'
        else f'{safe_model_name}.bmodel'
    )

    transform_args = [
        'model_transform.py',
        '--model_name', transform_model_name,
        '--model_def', './best.onnx',
        '--input_shapes', '[[1,3,640,640]]',
        '--mean', '0.0,0.0,0.0',
        '--scale', '0.0039216,0.0039216,0.0039216',
        '--keep_aspect_ratio',
        '--pixel_format', 'rgb',
        '--test_input', f'./images/{test_image_name}',
        '--test_result', test_result_name,
        '--mlir', mlir_name
    ]
    run_docker_bash(
        f"cd {shlex.quote(container_work_dir)} && {shlex.join(transform_args)}",
        f'{chip_label} model_transform'
    )

    deploy_args = [
        'model_deploy.py',
        '--mlir', f'./{mlir_name}',
        '--quantize', 'F32',
        '--chip', chip,
        '--test_input', deploy_test_input_name,
        '--test_reference', test_result_name,
        '--tolerance', '0.99,0.99',
        '--model', bmodel_name
    ]
    run_docker_bash(
        f"cd {shlex.quote(container_work_dir)} && {shlex.join(deploy_args)}",
        f'{chip_label} model_deploy'
    )

    validate_command = (
        "if command -v model_tool >/dev/null 2>&1; then "
        f"model_tool --info {shlex.quote(bmodel_name)}; "
        "else echo 'model_tool not found, skip bmodel info validation'; fi"
    )
    run_docker_bash(
        f"cd {shlex.quote(container_work_dir)} && {validate_command}",
        f'{chip_label} BModel产物校验'
    )

    local_bmodel_path = local_convert_dir / bmodel_name
    if local_bmodel_path.exists():
        local_bmodel_path.unlink()
    run_conversion_command(
        ['docker', 'cp', f'{MODEL_CONVERT_CONTAINER_ID}:{container_work_dir}/{bmodel_name}', str(local_bmodel_path)],
        '复制BModel到本地转换工作目录'
    )

    if not local_bmodel_path.exists():
        raise FileNotFoundError(f"{chip_label}转换完成但未找到BModel文件: {local_bmodel_path}")

    return local_bmodel_path


def get_rknn_convert_script(duty_type: str) -> Path:
    """根据任务类型选择 rknn_model_zoo 的 ONNX->RKNN 转换脚本"""
    normalized_duty_type = str(duty_type or 'detect').strip().lower()
    repo_root = Path(__file__).parent.resolve()

    if normalized_duty_type == 'detect':
        convert_script = repo_root / 'rknn_model_zoo' / 'examples' / 'yolo11' / 'python' / 'convert.py'
    elif normalized_duty_type == 'segment':
        convert_script = repo_root / 'rknn_model_zoo' / 'examples' / 'yolov8_seg' / 'python' / 'convert.py'
    elif normalized_duty_type == 'pose':
        convert_script = repo_root / 'rknn_model_zoo' / 'examples' / 'yolov8_pose' / 'python' / 'convert.py'
    else:
        raise ValueError(f"RK3588转换暂不支持duty_type={duty_type}，仅支持detect/segment/pose")

    if not convert_script.exists():
        raise FileNotFoundError(f"RKNN转换脚本不存在: {convert_script}")

    return convert_script


def run_rk3588_convert_pipeline(model_name, onnx_path: Path, local_convert_dir: Path, duty_type='detect', dtype='i8'):
    """执行 RK3588 的 ONNX -> RKNN 转换流程"""
    safe_model_name = sanitize_model_name(model_name)
    local_convert_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX模型不存在: {onnx_path}")

    local_onnx_path = local_convert_dir / 'best.onnx'
    if onnx_path.resolve() != local_onnx_path.resolve():
        shutil.copy2(onnx_path, local_onnx_path)

    local_rknn_path = local_convert_dir / f'{safe_model_name}.rknn'
    if local_rknn_path.exists():
        local_rknn_path.unlink()

    convert_script = get_rknn_convert_script(duty_type)
    rknn_model_zoo_root = Path(__file__).parent.resolve() / 'rknn_model_zoo'
    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH')
    env['PYTHONPATH'] = (
        f"{rknn_model_zoo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath else str(rknn_model_zoo_root)
    )

    run_conversion_command(
        [
            RKNN_CONVERT_PYTHON,
            str(convert_script),
            str(local_onnx_path),
            'rk3588',
            dtype,
            str(local_rknn_path)
        ],
        'RK3588 ONNX转RKNN',
        cwd=convert_script.parent,
        env=env
    )

    if not local_rknn_path.exists():
        raise FileNotFoundError(f"RK3588转换完成但未找到RKNN文件: {local_rknn_path}")

    return local_rknn_path

#模型初始化
class ModelManager:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.pre_model_path = self.base_path / "pre_model"
        self.trained_weights_path = self.base_path / "trained_weights"
        self.supported_extensions = ['.pt', '.onnx', '.engine']
        
    def scan_models(self, directory):
        models = []
        if not directory.exists():
            logger.warning(f"模型目录不存在: {directory}")
            return models
            
        try:
            for ext in self.supported_extensions:
                pattern = str(directory / f"*{ext}")
                found_models = glob.glob(pattern)
                models.extend(found_models)
            
            logger.info(f"在 {directory} 中发现 {len(models)} 个模型文件")
            return models
            
        except Exception as e:
            logger.error(f"扫描模型目录失败 {directory}: {str(e)}")
            return []
    
    def validate_model(self, model_path):
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return False, f"模型文件不存在: {model_path}"
            
            if model_path.stat().st_size == 0:
                return False, f"模型文件为空: {model_path}"
            
            if model_path.suffix == '.pt':
                try:
                    # 本地预训练模型文件，非用户上传，weights_only=False 是安全的
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    if 'model' not in checkpoint and 'state_dict' not in checkpoint:
                        try:
                            model = YOLO(str(model_path))
                            model_info = {
                                'type': 'yolo',
                                'model_name': getattr(model.model, 'yaml', {}).get('nc', 'unknown'),
                                'parameters': sum(p.numel() for p in model.model.parameters()) if hasattr(model, 'model') else 0
                            }
                            return True, model_info
                        except Exception:
                            return False, f"无法解析YOLO模型: {model_path}"
                    else:
                        return True, {'type': 'pytorch', 'format': 'checkpoint'}
                except Exception as e:
                    return False, f"模型文件损坏: {str(e)}"
            
            return True, {'type': 'other', 'format': model_path.suffix}
            
        except Exception as e:
            return False, f"验证模型失败: {str(e)}"
    
    def preload_model(self, model_path, load_to_memory=False):
        try:
            model_path = Path(model_path)
            model_key = model_path.name

            is_valid, info = self.validate_model(model_path)
            if not is_valid:
                logger.error(f"模型验证失败: {info}")
                return False

            with _model_info_lock:
                model_info[model_key] = {
                    'path': str(model_path),
                    'size': model_path.stat().st_size,
                    'modified': model_path.stat().st_mtime,
                    'info': info,
                    'loaded': False
                }

            if load_to_memory and model_path.suffix == '.pt':
                try:
                    if info.get('type') == 'yolo':
                        model = YOLO(str(model_path))
                        with _preloaded_models_lock:
                            preloaded_models[model_key] = model
                        with _model_info_lock:
                            model_info[model_key]['loaded'] = True
                        logger.info(f"模型已加载到内存: {model_key}")
                    else:
                        logger.info(f"模型已验证: {model_key} (非-YOLO模型，未加载到内存)")
                except Exception as e:
                    logger.warning(f"加载模型到内存失败 {model_key}: {str(e)}，将在需要时加载")
                    with _model_info_lock:
                        model_info[model_key]['loaded'] = False
            else:
                logger.info(f"模型已扫描和验证: {model_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"预加载模型失败 {model_path}: {str(e)}")
            return False
    
    def initialize_all_models(self, load_to_memory=False):
        logger.info("开始初始化模型")
        
        total_models = 0
        success_models = 0
        
        logger.info(f"扫描预训练模型目录: {self.pre_model_path}")
        pre_models = self.scan_models(self.pre_model_path)
        for model_path in pre_models:
            total_models += 1
            if self.preload_model(model_path, load_to_memory):
                success_models += 1

        logger.info(f"扫描训练权重目录: {self.trained_weights_path}")
        trained_models = self.scan_models(self.trained_weights_path)
        for model_path in trained_models:
            total_models += 1
            if self.preload_model(model_path, load_to_memory):
                success_models += 1
        
        logger.info(f"模型初始化完成: {success_models}/{total_models} 个模型成功初始化")

        if model_info:
            logger.info("可用模型列表:")
            for model_name, info in model_info.items():
                size_mb = info['size'] / (1024 * 1024)
                status = "已加载" if info['loaded'] else "已验证"
                logger.info(f"  - {model_name}: {size_mb:.2f}MB, {status}")
        
        return success_models, total_models
    
    def get_model(self, model_name):
        with _preloaded_models_lock:
            if model_name in preloaded_models:
                logger.info(f"使用预加载的模型: {model_name}")
                return preloaded_models[model_name]

        with _model_info_lock:
            if model_name in model_info:
                model_path = model_info[model_name]['path']
                logger.info(f"从文件加载模型: {model_name}")
                return YOLO(model_path)

        logger.warning(f"未找到模型: {model_name}，尝试使用默认加载")
        return None

model_manager = ModelManager()
try:
    success_count, total_count = model_manager.initialize_all_models(load_to_memory=False)
    logger.info(f"模型初始化结果: {success_count}/{total_count}")
except Exception as e:
    logger.error(f"模型初始化失败: {str(e)}")

def copy_to_storage_and_get_path(local_path, folder_type, sub_folder=None):
    try:
        local_path_obj = Path(local_path).resolve()
        file_name = Path(local_path).name
        path_parts = [folder_type]
        if sub_folder:
            path_parts.append(str(sub_folder))
        path_parts.append(file_name)

        target_storage_path = build_storage_path(*path_parts)
        upload_file_to_storage(local_path_obj, parse_storage_relative_path(target_storage_path))

        logger.info(f"文件已保存到本地工作区: {local_path_obj} -> {target_storage_path}")
        return target_storage_path
    except Exception as e:
        logger.error(f"保存本地工作区路径时出错: {str(e)}")
        return str(local_path)


class TrainerDatabaseManager(BaseDatabaseManager):
    """业务数据库管理器，继承连接池基类，包含业务查询方法"""

    def get_connection(self):
        """获取数据库连接（非上下文管理器版本，兼容现有业务方法）"""
        try:
            if self._engine:
                connection = self._engine.raw_connection()
            else:
                connect_config = {**self.config, 'connect_timeout': 10}
                connection = pymysql.connect(**connect_config)
            return connection
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            return None

    def _build_default_original_data_address(self, dataset_id):
        """根据 dataset_id 构造默认的原始数据本地路径"""
        return PathHandler.build_data_path('original_dataset', str(dataset_id), 'original_train_data')

    def _ensure_original_data_address(self, connection, cursor, dataset_id, original_data_address):
        """当 original_data_address 为空时，按约定路径自动补全并回写数据库"""
        if original_data_address:
            return original_data_address

        default_path = self._build_default_original_data_address(dataset_id)
        default_path_for_db = normalize_storage_path_for_db(default_path)

        try:
            sql = "UPDATE train_dataset SET original_data_address = %s WHERE id = %s"
            cursor.execute(sql, (default_path_for_db, dataset_id))
            connection.commit()
            logger.info(f"original_data_address为空，已按dataset_id自动补全: dataset_id={dataset_id}, path={default_path_for_db}")
        except Exception as e:
            connection.rollback()
            logger.warning(f"回写默认original_data_address失败，将继续使用拼接路径: dataset_id={dataset_id}, error={str(e)}")

        return default_path_for_db

    def _build_split_addresses(self, original_data_address):
        """根据 original_train_data 根路径推导 train/valid 划分路径"""
        base_path = original_data_address.rstrip('/').rsplit('/', 1)[0]
        return {
            'original_train_data_address': f"{base_path}/train",
            'original_val_data_address': f"{base_path}/valid"
        }

    def get_dataset_original_hash(self, dataset_id):
        """读取 train_dataset.original_dataset_hash 字段。

        字段不存在(老库未迁移)或查询失败时返回 None,使调用方退化为正常流程。
        """
        info = self.get_dataset_original_hash_info(dataset_id)
        return info.get('hash') if info else None

    def get_dataset_original_hash_info(self, dataset_id):
        """读取 original_dataset_hash 及其 schema 版本。"""
        connection = self.get_connection()
        if not connection:
            return None

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                try:
                    sql = "SELECT original_dataset_hash, original_dataset_hash_schema FROM train_dataset WHERE id = %s"
                    cursor.execute(sql, (dataset_id,))
                except Exception as e:
                    if 'original_dataset_hash_schema' not in str(e):
                        raise
                    connection.rollback()
                    sql = "SELECT original_dataset_hash FROM train_dataset WHERE id = %s"
                    cursor.execute(sql, (dataset_id,))
                result = cursor.fetchone()
                if not result:
                    return None
                return {
                    'hash': result.get('original_dataset_hash'),
                    'schema': result.get('original_dataset_hash_schema')
                }
        except Exception as e:
            logger.warning(f"读取original_dataset_hash失败,跳过哈希校验: dataset_id={dataset_id}, error={str(e)}")
            return None
        finally:
            connection.close()

    def update_dataset_original_hash(self, dataset_id, dataset_hash, schema_version=ORIGINAL_DATASET_HASH_SCHEMA_VERSION):
        """写入 train_dataset.original_dataset_hash 字段。"""
        if not dataset_hash:
            return False

        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                try:
                    sql = """UPDATE train_dataset
                             SET original_dataset_hash = %s,
                                 original_dataset_hash_schema = %s
                             WHERE id = %s"""
                    cursor.execute(sql, (dataset_hash, schema_version, dataset_id))
                except Exception as e:
                    if 'original_dataset_hash_schema' not in str(e):
                        raise
                    connection.rollback()
                    sql = "UPDATE train_dataset SET original_dataset_hash = %s WHERE id = %s"
                    cursor.execute(sql, (dataset_hash, dataset_id))
                connection.commit()
                return True
        except Exception as e:
            logger.warning(f"更新original_dataset_hash失败: dataset_id={dataset_id}, error={str(e)}")
            return False
        finally:
            connection.close()
    
    def get_train_task(self, task_id):
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT * FROM train_task WHERE id = %s"
                cursor.execute(sql, (task_id,))
                result = cursor.fetchone()
                return result
        except Exception as e:
            logger.error(f"查询训练任务失败: {str(e)}")
            return None
        finally:
            connection.close()
    
    def get_dataset_info(self, dataset_id):
        connection = self.get_connection()
        if not connection:
            return None

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT yaml_address, draw_type, kpt_num, kpt_labels FROM train_dataset WHERE id = %s"
                cursor.execute(sql, (dataset_id,))
                result = cursor.fetchone()
                return result
        except Exception as e:
            logger.error(f"查询数据集信息失败: {str(e)}")
            return None
        finally:
            connection.close()
    
    def get_dataset_full_info(self, dataset_id):
        connection = self.get_connection()
        if not connection:
            return None

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """SELECT id, duty_type, original_data_address, dataset_address, labels,
                                label_num, yaml_address, draw_type, kpt_num, kpt_labels
                         FROM train_dataset WHERE id = %s"""
                cursor.execute(sql, (dataset_id,))
                result = cursor.fetchone()
                return result
        except Exception as e:
            logger.error(f"查询数据集完整信息失败: {str(e)}")
            return None
        finally:
            connection.close()

    def get_dataset_json2txt_info(self, dataset_id):
        connection = self.get_connection()
        if not connection:
            return None

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT id, duty_type, original_data_address, labels, kpt_num, kpt_labels, draw_type FROM train_dataset WHERE id = %s"
                cursor.execute(sql, (dataset_id,))
                result = cursor.fetchone()

                if not result:
                    return None

                original_data_address = self._ensure_original_data_address(
                    connection=connection,
                    cursor=cursor,
                    dataset_id=dataset_id,
                    original_data_address=result.get('original_data_address')
                )

                split_addresses = self._build_split_addresses(original_data_address)

                return {
                    'duty_type': result.get('duty_type'),
                    'original_train_data_address': split_addresses['original_train_data_address'],
                    'original_val_data_address': split_addresses['original_val_data_address'],
                    'labels': result.get('labels'),
                    'kpt_num': result.get('kpt_num'),
                    'kpt_labels': result.get('kpt_labels'),
                    'draw_type': result.get('draw_type'),
                }
        except Exception as e:
            logger.error(f"查询json2txt数据集信息失败: {str(e)}")
            return None
        finally:
            connection.close()

    def update_dataset_yaml_address(self, dataset_id, yaml_address):
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                sql = "UPDATE train_dataset SET yaml_address = %s WHERE id = %s"
                cursor.execute(sql, (normalize_storage_path_for_db(yaml_address), dataset_id))
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"更新数据集yaml地址失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_task_status(self, task_id, status, process_id=None, start_time=None, end_time=None, 
                         trained_bestmodel_address=None, trained_lastmodel_address=None):
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                update_fields = ["status = %s"]
                values = [status]
                
                if process_id is not None:
                    update_fields.append("task_id = %s")
                    values.append(process_id)
                
                if start_time is not None:
                    update_fields.append("start_time = %s")
                    values.append(start_time)
                
                if end_time is not None:
                    update_fields.append("end_time = %s")
                    values.append(end_time)
                
                if trained_bestmodel_address is not None:
                    update_fields.append("trained_bestmodel_address = %s")
                    values.append(normalize_storage_path_for_db(trained_bestmodel_address))

                if trained_lastmodel_address is not None:
                    update_fields.append("trained_lastmodel_address = %s")
                    values.append(normalize_storage_path_for_db(trained_lastmodel_address))
                
                values.append(task_id)
                sql = f"UPDATE train_task SET {', '.join(update_fields)} WHERE id = %s"
                
                cursor.execute(sql, values)
                connection.commit()
                logger.info(f"更新任务 {task_id} 状态为 {status}")
                return True
        except Exception as e:
            logger.error(f"更新任务状态失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_task_start_time(self, task_id, start_time):
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                sql = "UPDATE train_task SET start_time = %s, status = 'running' WHERE id = %s"
                cursor.execute(sql, (start_time, task_id))
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"更新任务开始时间失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_task_completion(self, task_id, end_time, status='completed', 
                             trained_bestmodel_address=None, trained_lastmodel_address=None):
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                update_fields = ["status = %s", "end_time = %s"]
                values = [status, end_time]
                
                if trained_bestmodel_address:
                    update_fields.append("trained_bestmodel_address = %s")
                    values.append(normalize_storage_path_for_db(trained_bestmodel_address))

                if trained_lastmodel_address:
                    update_fields.append("trained_lastmodel_address = %s")
                    values.append(normalize_storage_path_for_db(trained_lastmodel_address))
                
                values.append(task_id)
                sql = f"UPDATE train_task SET {', '.join(update_fields)} WHERE id = %s"
                
                cursor.execute(sql, values)
                connection.commit()
                logger.info(f"更新任务完成信息: {task_id}, 状态: {status}")
                if trained_bestmodel_address:
                    logger.info(f"Best模型路径: {trained_bestmodel_address}")
                if trained_lastmodel_address:
                    logger.info(f"Last模型路径: {trained_lastmodel_address}")
                return True
        except Exception as e:
            logger.error(f"更新任务完成信息失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def get_original_data_address(self, dataset_id):
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT id, original_data_address FROM train_dataset WHERE id = %s"
                cursor.execute(sql, (dataset_id,))
                result = cursor.fetchone()
                if not result:
                    return None

                result['original_data_address'] = self._ensure_original_data_address(
                    connection=connection,
                    cursor=cursor,
                    dataset_id=dataset_id,
                    original_data_address=result.get('original_data_address')
                )
                return result
        except Exception as e:
            logger.error(f"查询数据集原始路径失败: {str(e)}")
            return None
        finally:
            connection.close()
    
    def get_divided_data_address(self, dataset_id):
        # 与 get_dataset_split_info 查询、解析逻辑完全一致，保留独立方法名仅为
        # 兼容历史调用点，实现委托以避免重复维护两份相同的 SQL/拆分逻辑。
        return self.get_dataset_split_info(dataset_id)

    def update_divided_dataset_detail(self, dataset_id, train_tag_sum, val_tag_sum, 
                                       train_tag_num, val_tag_num, 
                                       train_tag_percentage, val_tag_percentage):
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                sql = """UPDATE train_dataset 
                        SET train_tag_sum = %s, val_tag_sum = %s, 
                            train_tag_num = %s, val_tag_num = %s, 
                            train_tag_percentage = %s, val_tag_percentage = %s, 
                            update_time = NOW()
                        WHERE id = %s"""
                cursor.execute(sql, (train_tag_sum, val_tag_sum, 
                                    train_tag_num, val_tag_num, 
                                    train_tag_percentage, val_tag_percentage, 
                                    dataset_id))
                connection.commit()
                logger.info(f"更新划分数据集详情成功: dataset_id={dataset_id}")
                return True
        except Exception as e:
            logger.error(f"更新划分数据集详情失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def insert_dataset_labels(self, dataset_id, label_data_list):
        """插入数据集标签信息"""
        connection = self.get_connection()
        if not connection:
            return 0, len(label_data_list)
        
        success_count = 0
        fail_count = 0
        
        try:
            with connection.cursor() as cursor:
                delete_sql = "DELETE FROM train_dataset_label WHERE dataset_id = %s"
                cursor.execute(delete_sql, (dataset_id,))
                logger.info(f"已删除dataset_id={dataset_id}的旧标签数据")

                insert_sql = """INSERT INTO train_dataset_label 
                    (id, dataset_id, object_key, label_name, tag_num, proportion, data_type, create_time, update_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())"""
                
                for row in label_data_list:
                    try:
                        label_id = snowflake_generator.generate_id()
                        cursor.execute(insert_sql, (
                            label_id,
                            dataset_id,
                            row.get('object_key'),
                            row.get('label_name'),
                            row.get('tag_num'),
                            row.get('proportion'),
                            row.get('data_type')
                        ))
                        success_count += 1
                    except Exception as e:
                        logger.error(f"插入标签数据失败 label_name={row.get('label_name')}: {str(e)}")
                        fail_count += 1
                
                connection.commit()
                logger.info(f"插入数据集标签数据完成: dataset_id={dataset_id}, 成功={success_count}, 失败={fail_count}")
                return success_count, fail_count
        except Exception as e:
            logger.error(f"插入数据集标签数据失败: {str(e)}")
            return success_count, fail_count
        finally:
            connection.close()
    
    def update_dataset_detail(self, dataset_id, sample_num, annotation_num, labels, label_num,
                              tag_num, draw_type=None, tag_sum=None, tag_percentage=None,
                              kpt_num=None, kpt_labels=None):
        """更新数据集详情信息"""
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                # 动态拼接字段，避免在没有 draw_type/kpt_num 时也强写
                fields = [
                    "sample_num = %s", "annotation_num = %s", "labels = %s",
                    "label_num = %s", "tag_num = %s",
                    "tag_sum = %s", "tag_percentage = %s",
                    "update_time = NOW()",
                ]
                values = [sample_num, annotation_num, labels, label_num, tag_num, tag_sum, tag_percentage]

                if draw_type is not None:
                    fields.insert(5, "draw_type = %s")
                    values.insert(5, draw_type)

                # kpt_num 仅在显式传入（包括 0）时才更新
                if kpt_num is not None:
                    fields.append("kpt_num = %s")
                    values.append(int(kpt_num))

                if kpt_labels is not None:
                    fields.append("kpt_labels = %s")
                    values.append(kpt_labels)

                values.append(dataset_id)
                sql = f"UPDATE train_dataset SET {', '.join(fields)} WHERE id = %s"
                cursor.execute(sql, values)
                connection.commit()
                logger.info(f"更新数据集详情成功: dataset_id={dataset_id}")
                return True
        except Exception as e:
            logger.error(f"更新数据集详情失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def get_dataset_split_info(self, dataset_id):
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT id, original_data_address FROM train_dataset WHERE id = %s"
                cursor.execute(sql, (dataset_id,))
                result = cursor.fetchone()
                
                if not result:
                    return None

                original_data_address = self._ensure_original_data_address(
                    connection=connection,
                    cursor=cursor,
                    dataset_id=dataset_id,
                    original_data_address=result.get('original_data_address')
                )

                return self._build_split_addresses(original_data_address)
        except Exception as e:
            logger.error(f"查询数据集创建信息失败: {str(e)}")
            return None
        finally:
            connection.close()
    
    def update_dataset_update_by(self, dataset_id, user_name):
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                sql = "UPDATE train_dataset SET update_by = %s WHERE id = %s"
                cursor.execute(sql, (user_name, dataset_id))
                connection.commit()
                logger.info(f"更新数据集update_by成功: dataset_id={dataset_id}, user_name={user_name}")
                return True
        except Exception as e:
            logger.error(f"更新数据集update_by失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_dataset_address(self, dataset_id, dataset_address):
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                sql = "UPDATE train_dataset SET dataset_address = %s WHERE id = %s"
                cursor.execute(sql, (normalize_storage_path_for_db(dataset_address), dataset_id))
                connection.commit()
                logger.info(f"更新数据集dataset_address成功: dataset_id={dataset_id}, dataset_address={dataset_address}")
                return True
        except Exception as e:
            logger.error(f"更新数据集dataset_address失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_original_data_address(self, dataset_id, original_data_address):
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                sql = "UPDATE train_dataset SET original_data_address = %s WHERE id = %s"
                cursor.execute(sql, (normalize_storage_path_for_db(original_data_address), dataset_id))
                connection.commit()
                logger.info(f"更新数据集original_data_address成功: dataset_id={dataset_id}, original_data_address={original_data_address}")
                return True
        except Exception as e:
            logger.error(f"更新数据集original_data_address失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_dataset_class_mapping(self, dataset_id, class_mapping):
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                class_mapping_str = json.dumps(class_mapping, ensure_ascii=False)
                sql = "UPDATE train_dataset SET class = %s WHERE id = %s"
                cursor.execute(sql, (class_mapping_str, dataset_id))
                connection.commit()
                logger.info(f"更新数据集class映射成功: dataset_id={dataset_id}, class_mapping={class_mapping_str}")
                return True
        except Exception as e:
            logger.error(f"更新数据集class映射失败: {str(e)}")
            return False
        finally:
            connection.close()

    def update_dataset_keypoint_config(self, dataset_id, kpt_labels, kpt_num=None):
        """更新 pose 关键点标签顺序与数量。"""
        connection = self.get_connection()
        if not connection:
            return False

        normalized_labels = sort_keypoint_labels(kpt_labels or [])
        if kpt_num is None:
            kpt_num = len(normalized_labels)

        try:
            with connection.cursor() as cursor:
                sql = """UPDATE train_dataset
                         SET kpt_num = %s,
                             kpt_labels = %s,
                             update_time = NOW()
                         WHERE id = %s"""
                cursor.execute(
                    sql,
                    (int(kpt_num or 0), json.dumps(normalized_labels, ensure_ascii=False), dataset_id)
                )
                connection.commit()
                logger.info(
                    f"更新数据集关键点配置成功: dataset_id={dataset_id}, "
                    f"kpt_num={kpt_num}, kpt_labels={normalized_labels}"
                )
                return True
        except Exception as e:
            logger.error(f"更新数据集关键点配置失败: dataset_id={dataset_id}, error={str(e)}")
            return False
        finally:
            connection.close()

    def update_dataset_process_status(self, dataset_id, process_status, error_message=None):
        connection = self.get_connection()
        if not connection:
            return False

        truncated_error = str(error_message)[:1024] if error_message is not None else None
        update_sql_candidates = []

        if truncated_error is None:
            update_sql_candidates = [
                ("UPDATE train_dataset SET process_status = %s, error_message = NULL, update_time = NOW() WHERE id = %s", (process_status, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, build_error = NULL, update_time = NOW() WHERE id = %s", (process_status, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, status_message = NULL, update_time = NOW() WHERE id = %s", (process_status, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, process_error = NULL, update_time = NOW() WHERE id = %s", (process_status, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, update_time = NOW() WHERE id = %s", (process_status, dataset_id)),
            ]
        else:
            update_sql_candidates = [
                ("UPDATE train_dataset SET process_status = %s, error_message = %s, update_time = NOW() WHERE id = %s", (process_status, truncated_error, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, build_error = %s, update_time = NOW() WHERE id = %s", (process_status, truncated_error, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, status_message = %s, update_time = NOW() WHERE id = %s", (process_status, truncated_error, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, process_error = %s, update_time = NOW() WHERE id = %s", (process_status, truncated_error, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, error = %s, update_time = NOW() WHERE id = %s", (process_status, truncated_error, dataset_id)),
                ("UPDATE train_dataset SET process_status = %s, update_time = NOW() WHERE id = %s", (process_status, dataset_id)),
            ]

        last_error = None
        try:
            for sql, params in update_sql_candidates:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(sql, params)
                    connection.commit()
                    if truncated_error is None:
                        logger.info(
                            f"更新数据集process_status成功: dataset_id={dataset_id}, process_status={process_status}"
                        )
                    else:
                        logger.info(
                            f"更新数据集process_status和错误信息成功: dataset_id={dataset_id}, process_status={process_status}, error_message={truncated_error}"
                        )
                    return True
                except Exception as e:
                    last_error = e
                    connection.rollback()
                    continue

            if last_error:
                logger.error(f"更新数据集process_status失败: {str(last_error)}")
            return False
        finally:
            connection.close()
    
    def update_task_log_path(self, task_id, log_path):
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                sql = "UPDATE train_task SET log_path = %s WHERE id = %s"
                cursor.execute(sql, (normalize_storage_path_for_db(log_path), task_id))
                connection.commit()
                logger.info(f"更新任务日志路径成功: task_id={task_id}, log_path={log_path}")
                return True
        except Exception as e:
            logger.error(f"更新任务日志路径失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_task_chart_path(self, task_id, chart_path):
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                sql = "UPDATE train_task SET chart_path = %s WHERE id = %s"
                cursor.execute(sql, (normalize_storage_path_for_db(chart_path), task_id))
                connection.commit()
                logger.info(f"更新任务图表路径成功: task_id={task_id}, chart_path={chart_path}")
                return True
        except Exception as e:
            logger.error(f"更新任务图表路径失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def get_task_chart_path(self, task_id):
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT chart_path FROM train_task WHERE id = %s"
                cursor.execute(sql, (task_id,))
                result = cursor.fetchone()
                return result['chart_path'] if result else None
        except Exception as e:
            logger.error(f"获取任务图表路径失败: {str(e)}")
            return None
        finally:
            connection.close()
    
    def get_task_log_path(self, task_id):
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT log_path FROM train_task WHERE id = %s"
                cursor.execute(sql, (task_id,))
                result = cursor.fetchone()
                return result.get('log_path') if result else None
        except Exception as e:
            logger.error(f"获取任务日志路径失败: {str(e)}")
            return None
        finally:
            connection.close()
    
    def get_predict_params(self, predict_dataset_id, train_weights_id):
        """获取推理预测所需的参数"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # 查询数据集信息
                dataset_sql = "SELECT dataset_address FROM train_dataset WHERE id = %s"
                cursor.execute(dataset_sql, (predict_dataset_id,))
                dataset_result = cursor.fetchone()
                
                if not dataset_result:
                    logger.error(f"未找到数据集: {predict_dataset_id}")
                    return None
                
                # 查询模型权重信息
                weights_sql = """SELECT duty_type, img_size, bestmodel_address 
                                FROM trained_weights WHERE id = %s"""
                cursor.execute(weights_sql, (train_weights_id,))
                weights_result = cursor.fetchone()
                
                if not weights_result:
                    logger.error(f"未找到模型权重: {train_weights_id}")
                    return None
                
                return {
                    'dataset_address': dataset_result['dataset_address'],
                    'duty_type': weights_result['duty_type'],
                    'img_size': weights_result['img_size'],
                    'model_address': weights_result['bestmodel_address']
                }
        except Exception as e:
            logger.error(f"获取推理参数失败: {str(e)}")
            return None
        finally:
            connection.close()
    
    def update_predicted_result_address(self, train_weights_id, predicted_result_address):
        """更新模型预测结果路径"""
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                sql = """UPDATE trained_weights
                        SET predicted_result_address = %s
                        WHERE id = %s"""
                cursor.execute(sql, (normalize_storage_path_for_db(predicted_result_address), train_weights_id))
                connection.commit()
                logger.info(f"更新预测结果路径成功: weights_id={train_weights_id}, path={predicted_result_address}")
                return True
        except Exception as e:
            logger.error(f"更新预测结果路径失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_test_status(self, train_weights_id, test_status):
        """更新模型权重的测试状态"""
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                sql = """UPDATE trained_weights 
                        SET test_status = %s 
                        WHERE id = %s"""
                cursor.execute(sql, (test_status, train_weights_id))
                connection.commit()
                logger.info(f"更新测试状态成功: weights_id={train_weights_id}, test_status={test_status}")
                return True
        except Exception as e:
            logger.error(f"更新测试状态失败: {str(e)}")
            return False
        finally:
            connection.close()

    def get_trained_weight_for_convert(self, model_id):
        connection = self.get_connection()
        if not connection:
            return None

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT * FROM trained_weights WHERE id = %s"
                cursor.execute(sql, (model_id,))
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"查询待转换模型失败: {str(e)}")
            return None
        finally:
            connection.close()

    def get_dataset_record(self, dataset_id):
        connection = self.get_connection()
        if not connection:
            return None

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = "SELECT * FROM train_dataset WHERE id = %s"
                cursor.execute(sql, (dataset_id,))
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"查询数据集记录失败: {str(e)}")
            return None
        finally:
            connection.close()

    def upsert_model_convert_processing(self, model_id, type_code, source_path, convert_format, task_id=None):
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                sql = """INSERT INTO model_convert_path
                    (model_id, type_code, source_path, convert_path, convert_format, file_size,
                     status, progress, task_id, error_message, started_at, finished_at, create_time, update_time)
                    VALUES (%s, %s, %s, NULL, %s, NULL, %s, 0, %s, NULL, NOW(), NULL, NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                        source_path = VALUES(source_path),
                        convert_path = NULL,
                        convert_format = VALUES(convert_format),
                        file_size = NULL,
                        status = VALUES(status),
                        progress = 0,
                        task_id = VALUES(task_id),
                        error_message = NULL,
                        started_at = NOW(),
                        finished_at = NULL,
                        update_time = NOW()"""
                cursor.execute(sql, (
                    model_id,
                    type_code,
                    normalize_storage_path_for_db(source_path),
                    convert_format,
                    MODEL_CONVERT_STATUS_RUNNING,
                    task_id
                ))

                # 防御性更新：确保旧失败记录被本次新任务明确置为“转换中”。
                force_sql = """UPDATE model_convert_path
                               SET status = %s,
                                   progress = 0,
                                   convert_path = NULL,
                                   file_size = NULL,
                                   error_message = NULL,
                                   started_at = NOW(),
                                   finished_at = NULL,
                                   update_time = NOW()
                               WHERE model_id = %s AND type_code = %s"""
                cursor.execute(force_sql, (MODEL_CONVERT_STATUS_RUNNING, model_id, type_code))
                connection.commit()
                logger.info(f"模型转换状态已置为转换中: model_id={model_id}, type_code={type_code}")
                return True
        except Exception as e:
            logger.error(f"写入模型转换中状态失败: {str(e)}")
            return False
        finally:
            connection.close()

    def update_model_convert_success(self, model_id, type_code, source_path, convert_path, convert_format, file_size_mb, task_id=None):
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                sql = """INSERT INTO model_convert_path
                    (model_id, type_code, source_path, convert_path, convert_format, file_size,
                     status, progress, task_id, error_message, started_at, finished_at, create_time, update_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 100, %s, NULL, NOW(), NOW(), NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                        source_path = VALUES(source_path),
                        convert_path = VALUES(convert_path),
                        convert_format = VALUES(convert_format),
                        file_size = VALUES(file_size),
                        status = %s,
                        progress = 100,
                        task_id = VALUES(task_id),
                        error_message = NULL,
                        finished_at = NOW(),
                        update_time = NOW()"""
                cursor.execute(sql, (
                    model_id,
                    type_code,
                    normalize_storage_path_for_db(source_path),
                    normalize_storage_path_for_db(convert_path),
                    convert_format,
                    file_size_mb,
                    MODEL_CONVERT_STATUS_SUCCESS,
                    task_id,
                    MODEL_CONVERT_STATUS_SUCCESS
                ))
                connection.commit()
                logger.info(
                    f"模型转换成功状态已写入: model_id={model_id}, type_code={type_code}, convert_path={convert_path}"
                )
                return True
        except Exception as e:
            logger.error(f"写入模型转换成功状态失败: {str(e)}")
            return False
        finally:
            connection.close()

    def update_model_convert_failure(self, model_id, type_code, source_path, error_message, convert_format=None, task_id=None):
        connection = self.get_connection()
        if not connection:
            return False

        try:
            with connection.cursor() as cursor:
                truncated_error = str(error_message or '')[:1024]
                sql = """INSERT INTO model_convert_path
                    (model_id, type_code, source_path, convert_path, convert_format, file_size,
                     status, progress, task_id, error_message, started_at, finished_at, create_time, update_time)
                    VALUES (%s, %s, %s, NULL, %s, NULL, %s, 0, %s, %s, NOW(), NOW(), NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                        source_path = VALUES(source_path),
                        convert_format = VALUES(convert_format),
                        status = %s,
                        task_id = VALUES(task_id),
                        error_message = VALUES(error_message),
                        finished_at = NOW(),
                        update_time = NOW()"""
                cursor.execute(sql, (
                    model_id,
                    type_code,
                    normalize_storage_path_for_db(source_path),
                    convert_format,
                    MODEL_CONVERT_STATUS_FAILED,
                    task_id,
                    truncated_error,
                    MODEL_CONVERT_STATUS_FAILED
                ))
                connection.commit()
                logger.info(f"模型转换失败状态已写入: model_id={model_id}, type_code={type_code}")
                return True
        except Exception as e:
            logger.error(f"写入模型转换失败状态失败: {str(e)}")
            return False
        finally:
            connection.close()

    def get_label_task_assignee_dirs(self, label_task_id):
        """读取标注任务分配目录。"""
        connection = self.get_connection()
        if not connection:
            return None

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """SELECT task_id, user_id, user_nickname, task_dir
                         FROM label_task_assignee
                         WHERE task_id = %s AND deleted = b'0'"""
                cursor.execute(sql, (label_task_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"查询标注任务分配目录失败: label_task_id={label_task_id}, error={str(e)}")
            return None
        finally:
            connection.close()

    def get_trained_weight_for_auto_label(self, model_id):
        """读取自动标注所需模型信息。"""
        connection = self.get_connection()
        if not connection:
            return None

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """SELECT id, bestmodel_address, duty_type, img_size, `class`
                         FROM trained_weights
                         WHERE id = %s"""
                cursor.execute(sql, (model_id,))
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"查询自动标注模型信息失败: model_id={model_id}, error={str(e)}")
            return None
        finally:
            connection.close()

    def update_auto_label_items(self, label_task_id, auto_label_records):
        """把生成的自动标注 JSON 路径写回 label_task_item。"""
        if not auto_label_records:
            return {'updated_count': 0, 'missing_items': []}

        connection = self.get_connection()
        if not connection:
            return None

        updated_count = 0
        missing_items = []
        try:
            with connection.cursor() as cursor:
                update_by_path_sql = """UPDATE label_task_item
                    SET auto_label_path = %s,
                        auto_labeled = 1,
                        update_time = NOW()
                    WHERE task_id = %s
                      AND deleted = b'0'
                      AND image_path = %s"""
                update_by_name_sql = """UPDATE label_task_item
                    SET auto_label_path = %s,
                        auto_labeled = 1,
                        update_time = NOW()
                    WHERE task_id = %s
                      AND deleted = b'0'
                      AND image_name = %s"""

                for record in auto_label_records:
                    json_db_path = record.get('json_db_path')
                    image_db_path = record.get('image_db_path')
                    image_name = record.get('image_name')

                    cursor.execute(update_by_path_sql, (json_db_path, label_task_id, image_db_path))
                    affected = cursor.rowcount
                    if affected == 0 and image_name:
                        cursor.execute(update_by_name_sql, (json_db_path, label_task_id, image_name))
                        affected = cursor.rowcount

                    if affected:
                        updated_count += affected
                    else:
                        missing_items.append({
                            'image_name': image_name,
                            'image_path': image_db_path,
                            'auto_label_path': json_db_path
                        })

            connection.commit()
            logger.info(
                f"自动标注结果写回完成: label_task_id={label_task_id}, "
                f"updated_count={updated_count}, missing_count={len(missing_items)}"
            )
            return {'updated_count': updated_count, 'missing_items': missing_items}
        except Exception as e:
            connection.rollback()
            logger.error(f"写回自动标注结果失败: label_task_id={label_task_id}, error={str(e)}")
            return None
        finally:
            connection.close()
    
    def insert_train_chart(self, train_task_id, chart_data_list):
        connection = self.get_connection()
        if not connection:
            return 0, len(chart_data_list)
        
        success_count = 0
        fail_count = 0
        
        try:
            with connection.cursor() as cursor:
                delete_sql = "DELETE FROM train_chart WHERE train_task_id = %s"
                cursor.execute(delete_sql, (train_task_id,))
                logger.info(f"已删除train_task_id={train_task_id}的旧图表数据")

                insert_sql = """INSERT INTO train_chart 
                    (train_task_id, epoch, time, train_box_loss, train_cls_loss, train_dfl_loss,
                     metrics_precision_B, metrics_recall_B, metrics_mAP50_B, metrics_mAP50_95_B,
                     val_box_loss, val_cls_loss, val_dfl_loss, lr_pg0, lr_pg1, lr_pg2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                
                for row in chart_data_list:
                    try:
                        cursor.execute(insert_sql, (
                            train_task_id,
                            row.get('epoch'),
                            row.get('time'),
                            row.get('train_box_loss'),
                            row.get('train_cls_loss'),
                            row.get('train_dfl_loss'),
                            row.get('metrics_precision_B'),
                            row.get('metrics_recall_B'),
                            row.get('metrics_mAP50_B'),
                            row.get('metrics_mAP50_95_B'),
                            row.get('val_box_loss'),
                            row.get('val_cls_loss'),
                            row.get('val_dfl_loss'),
                            row.get('lr_pg0'),
                            row.get('lr_pg1'),
                            row.get('lr_pg2')
                        ))
                        success_count += 1
                    except Exception as e:
                        logger.error(f"插入图表数据失败 epoch={row.get('epoch')}: {str(e)}")
                        fail_count += 1
                
                connection.commit()
                logger.info(f"插入训练图表数据完成: train_task_id={train_task_id}, 成功={success_count}, 失败={fail_count}")
                return success_count, fail_count
        except Exception as e:
            logger.error(f"插入训练图表数据失败: {str(e)}")
            return success_count, fail_count
        finally:
            connection.close()
    
    def insert_single_train_chart(self, train_task_id, chart_row):
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                check_sql = "SELECT 1 FROM train_chart WHERE train_task_id = %s AND epoch = %s LIMIT 1"
                cursor.execute(check_sql, (train_task_id, chart_row.get('epoch')))
                exists = cursor.fetchone()
                
                if exists:
                    # 已存在则更新
                    sql = """UPDATE train_chart SET
                        time = %s, train_box_loss = %s, train_cls_loss = %s, train_dfl_loss = %s,
                        metrics_precision_B = %s, metrics_recall_B = %s, metrics_mAP50_B = %s, metrics_mAP50_95_B = %s,
                        val_box_loss = %s, val_cls_loss = %s, val_dfl_loss = %s, lr_pg0 = %s, lr_pg1 = %s, lr_pg2 = %s
                        WHERE train_task_id = %s AND epoch = %s"""
                    cursor.execute(sql, (
                        chart_row.get('time'),
                        chart_row.get('train_box_loss'),
                        chart_row.get('train_cls_loss'),
                        chart_row.get('train_dfl_loss'),
                        chart_row.get('metrics_precision_B'),
                        chart_row.get('metrics_recall_B'),
                        chart_row.get('metrics_mAP50_B'),
                        chart_row.get('metrics_mAP50_95_B'),
                        chart_row.get('val_box_loss'),
                        chart_row.get('val_cls_loss'),
                        chart_row.get('val_dfl_loss'),
                        chart_row.get('lr_pg0'),
                        chart_row.get('lr_pg1'),
                        chart_row.get('lr_pg2'),
                        train_task_id,
                        chart_row.get('epoch')
                    ))
                else:
                    # 不存在则插入
                    sql = """INSERT INTO train_chart 
                        (train_task_id, epoch, time, train_box_loss, train_cls_loss, train_dfl_loss,
                         metrics_precision_B, metrics_recall_B, metrics_mAP50_B, metrics_mAP50_95_B,
                         val_box_loss, val_cls_loss, val_dfl_loss, lr_pg0, lr_pg1, lr_pg2)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                    cursor.execute(sql, (
                        train_task_id,
                        chart_row.get('epoch'),
                        chart_row.get('time'),
                        chart_row.get('train_box_loss'),
                        chart_row.get('train_cls_loss'),
                        chart_row.get('train_dfl_loss'),
                        chart_row.get('metrics_precision_B'),
                        chart_row.get('metrics_recall_B'),
                        chart_row.get('metrics_mAP50_B'),
                        chart_row.get('metrics_mAP50_95_B'),
                        chart_row.get('val_box_loss'),
                        chart_row.get('val_cls_loss'),
                        chart_row.get('val_dfl_loss'),
                        chart_row.get('lr_pg0'),
                        chart_row.get('lr_pg1'),
                        chart_row.get('lr_pg2')
                    ))
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"插入单条图表数据失败 epoch={chart_row.get('epoch')}: {str(e)}")
            return False
        finally:
            connection.close()

    def upsert_train_chart_batch(self, train_task_id, chart_rows):
        """批量 upsert 多个 epoch 的图表数据，复用单个连接。

        与逐行调用 insert_single_train_chart 语义一致（按 epoch 存在则更新、
        否则插入），但 CSV 监控线程每 2s 会处理一批新行——逐行各开一条新连接
        在长训练下持续放大连接开销，这里改为整批共用一条连接，仅一次提交。
        """
        if not chart_rows:
            return 0
        connection = self.get_connection()
        if not connection:
            return 0

        update_sql = """UPDATE train_chart SET
            time = %s, train_box_loss = %s, train_cls_loss = %s, train_dfl_loss = %s,
            metrics_precision_B = %s, metrics_recall_B = %s, metrics_mAP50_B = %s, metrics_mAP50_95_B = %s,
            val_box_loss = %s, val_cls_loss = %s, val_dfl_loss = %s, lr_pg0 = %s, lr_pg1 = %s, lr_pg2 = %s
            WHERE train_task_id = %s AND epoch = %s"""
        insert_sql = """INSERT INTO train_chart
            (train_task_id, epoch, time, train_box_loss, train_cls_loss, train_dfl_loss,
             metrics_precision_B, metrics_recall_B, metrics_mAP50_B, metrics_mAP50_95_B,
             val_box_loss, val_cls_loss, val_dfl_loss, lr_pg0, lr_pg1, lr_pg2)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        check_sql = "SELECT 1 FROM train_chart WHERE train_task_id = %s AND epoch = %s LIMIT 1"

        saved = 0
        try:
            with connection.cursor() as cursor:
                for chart_row in chart_rows:
                    epoch = chart_row.get('epoch')
                    try:
                        cursor.execute(check_sql, (train_task_id, epoch))
                        if cursor.fetchone():
                            cursor.execute(update_sql, (
                                chart_row.get('time'),
                                chart_row.get('train_box_loss'),
                                chart_row.get('train_cls_loss'),
                                chart_row.get('train_dfl_loss'),
                                chart_row.get('metrics_precision_B'),
                                chart_row.get('metrics_recall_B'),
                                chart_row.get('metrics_mAP50_B'),
                                chart_row.get('metrics_mAP50_95_B'),
                                chart_row.get('val_box_loss'),
                                chart_row.get('val_cls_loss'),
                                chart_row.get('val_dfl_loss'),
                                chart_row.get('lr_pg0'),
                                chart_row.get('lr_pg1'),
                                chart_row.get('lr_pg2'),
                                train_task_id,
                                epoch,
                            ))
                        else:
                            cursor.execute(insert_sql, (
                                train_task_id,
                                epoch,
                                chart_row.get('time'),
                                chart_row.get('train_box_loss'),
                                chart_row.get('train_cls_loss'),
                                chart_row.get('train_dfl_loss'),
                                chart_row.get('metrics_precision_B'),
                                chart_row.get('metrics_recall_B'),
                                chart_row.get('metrics_mAP50_B'),
                                chart_row.get('metrics_mAP50_95_B'),
                                chart_row.get('val_box_loss'),
                                chart_row.get('val_cls_loss'),
                                chart_row.get('val_dfl_loss'),
                                chart_row.get('lr_pg0'),
                                chart_row.get('lr_pg1'),
                                chart_row.get('lr_pg2'),
                            ))
                        saved += 1
                    except Exception as e:
                        logger.error(f"批量写入图表数据失败 epoch={epoch}: {str(e)}")
                connection.commit()
            return saved
        except Exception as e:
            logger.error(f"批量写入图表数据失败 train_task_id={train_task_id}: {str(e)}")
            return saved
        finally:
            connection.close()

    def clear_train_chart(self, train_task_id):
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                sql = "DELETE FROM train_chart WHERE train_task_id = %s"
                cursor.execute(sql, (train_task_id,))
                connection.commit()
                logger.info(f"已清除train_task_id={train_task_id}的图表数据")
                return True
        except Exception as e:
            logger.error(f"清除图表数据失败: {str(e)}")
            return False
        finally:
            connection.close()
    
    def update_task_metrics(self, task_id, precision=None, recall=None, accuracy=None):
        """更新训练任务指标（精度、召回率、准确率）"""
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            with connection.cursor() as cursor:
                update_fields = []
                values = []
                
                if precision is not None:
                    update_fields.append("`precision` = %s")
                    values.append(precision)
                
                if recall is not None:
                    update_fields.append("recall = %s")
                    values.append(recall)
                
                if accuracy is not None:
                    update_fields.append("accuracy = %s")
                    values.append(accuracy)
                
                if not update_fields:
                    logger.warning(f"没有需要更新的指标字段: task_id={task_id}")
                    return True
                
                values.append(task_id)
                sql = f"UPDATE train_task SET {', '.join(update_fields)} WHERE id = %s"
                
                cursor.execute(sql, values)
                connection.commit()
                logger.info(f"更新训练指标成功: task_id={task_id}, precision={precision}, recall={recall}, accuracy={accuracy}")
                return True
        except Exception as e:
            logger.error(f"更新训练指标失败: {str(e)}")
            return False
        finally:
            connection.close()

    def get_queued_running_tasks(self):
        connection = self.get_connection()
        if not connection:
            return []
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """SELECT id, status, name, epochs, create_time 
                        FROM train_task WHERE status IN ('queued', 'running')"""
                cursor.execute(sql)
                results = cursor.fetchall()
                return results
        except Exception as e:
            logger.error(f"查询queued/running任务失败: {str(e)}")
            return []
        finally:
            connection.close()
    
    def get_task_for_redis(self, task_id):
        """获取单个任务的Redis同步所需信息"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """SELECT id, status, name, epochs, create_time 
                        FROM train_task WHERE id = %s"""
                cursor.execute(sql, (task_id,))
                result = cursor.fetchone()
                return result
        except Exception as e:
            logger.error(f"查询任务Redis信息失败: {str(e)}")
            return None
        finally:
            connection.close()

    def insert_train_weights(self, train_task_id, bestmodel_address=None, lastmodel_address=None, 
                            precision=None, recall=None, accuracy=None):
        """训练完成后，将训练任务信息插入train_weights表"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                task_sql = """SELECT id, name, model_size, creator, updater, img_size, epochs, 
                             duty_type, dataset_id, status, start_time, end_time
                             FROM train_task WHERE id = %s"""
                cursor.execute(task_sql, (train_task_id,))
                task_info = cursor.fetchone()
                
                if not task_info:
                    logger.error(f"未找到train_task记录: {train_task_id}")
                    return None
                
                # 从train_dataset表中获取labels和class字段
                labels_value = None
                class_value = None
                dataset_id = task_info.get('dataset_id')
                if dataset_id:
                    dataset_sql = "SELECT labels, class FROM train_dataset WHERE id = %s"
                    cursor.execute(dataset_sql, (dataset_id,))
                    dataset_info = cursor.fetchone()
                    if dataset_info:
                        labels_value = dataset_info.get('labels')
                        class_value = dataset_info.get('class')
                        logger.info(f"从train_dataset获取labels: dataset_id={dataset_id}, labels={labels_value}")
                        logger.info(f"从train_dataset获取class: dataset_id={dataset_id}, class={class_value}")
                
                file_size_mb = None
                if bestmodel_address:
                    try:
                        local_best_path = ensure_local_path(bestmodel_address)
                        if Path(local_best_path).exists():
                            file_size_bytes = Path(local_best_path).stat().st_size
                            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
                            logger.info(f"best.pt文件大小: {file_size_mb} MB")
                    except Exception as e:
                        logger.warning(f"获取best.pt文件大小失败: {str(e)}")
                weights_id = snowflake_generator.generate_id()
                
                if labels_value is not None:
                    if isinstance(labels_value, str):
                        try:
                            json.loads(labels_value)
                        except json.JSONDecodeError:
                            labels_value = json.dumps([labels_value], ensure_ascii=False)
                    elif isinstance(labels_value, (list, dict)):
                        labels_value = json.dumps(labels_value, ensure_ascii=False)
                
                insert_sql = """INSERT INTO trained_weights 
                    (id, train_task_id, name, model_size, creator, updater, img_size, epochs,
                     duty_type, `labels`, dataset_id, bestmodel_address, lastmodel_address,
                     file_size, status, recall, `precision`, accuracy, start_time, end_time, class, create_time, update_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())"""
                
                cursor.execute(insert_sql, (
                    weights_id,
                    train_task_id,
                    task_info.get('name'),
                    task_info.get('model_size'),
                    task_info.get('creator'),
                    task_info.get('updater'),
                    task_info.get('img_size'),
                    task_info.get('epochs'),
                    task_info.get('duty_type'),
                    labels_value,
                    task_info.get('dataset_id'),
                    normalize_storage_path_for_db(bestmodel_address),
                    normalize_storage_path_for_db(lastmodel_address),
                    file_size_mb,
                    task_info.get('status'),
                    recall,
                    precision,
                    accuracy,
                    task_info.get('start_time'),
                    task_info.get('end_time'),
                    class_value
                ))
                
                connection.commit()
                logger.info(f"插入train_weights成功: id={weights_id}, train_task_id={train_task_id}, file_size={file_size_mb}MB")
                return weights_id
                
        except Exception as e:
            logger.error(f"插入train_weights失败: {str(e)}")
            return None
        finally:
            connection.close()


class RedisManager:
    """Redis管理器，用于同步训练任务状态"""
    
    KEY_PREFIX = "train:task"
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        try:
            import redis
            self.client = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                db=self.config['db'],
                password=self.config['password'],
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"Redis连接成功: {self.config['host']}:{self.config['port']}, db={self.config['db']}")
        except ImportError:
            logger.error("redis模块未安装，请运行: pip install redis")
            self.client = None
        except Exception as e:
            logger.error(f"Redis连接失败: {str(e)}")
            self.client = None
    
    def _build_key(self, task_id):
        return f"{self.KEY_PREFIX}:{task_id}"
    
    def _build_hash_data(self, task_info):
        create_time = task_info.get('create_time')
        if create_time:
            if hasattr(create_time, 'strftime'):
                create_time = create_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                create_time = str(create_time)
        
        return {
            'id': str(task_info.get('id', '')),
            'status': str(task_info.get('status', '')),
            'name': str(task_info.get('name', '') or ''),
            'epochs': str(task_info.get('epochs', 0) or 0),
            'current_epoch': '0',
            'eta_seconds': '0',
            'createTime': create_time or ''
        }
    
    def sync_task_status(self, task_id, status, task_info=None):
        if not self.client:
            return False
        
        try:
            key = self._build_key(task_id)
            if status in ('queued', 'running'):
                if task_info:
                    hash_data = self._build_hash_data(task_info)
                else:
                    hash_data = {'id': str(task_id), 'status': status}
                
                self.client.hset(key, mapping=hash_data)
                logger.debug(f"Redis同步(Hash): {key} = {hash_data}")
            else:
                if self.client.exists(key):
                    self.client.hset(key, 'status', status)
                    logger.debug(f"Redis更新状态: {key}.status = {status}")
                else:
                    hash_data = {'id': str(task_id), 'status': status}
                    self.client.hset(key, mapping=hash_data)
                    logger.debug(f"Redis创建(Hash): {key} = {hash_data}")
            return True
        except Exception as e:
            logger.error(f"Redis同步任务状态失败: {str(e)}")
            return False
    
    def sync_all_tasks(self, tasks):
        if not self.client:
            return False
        
        try:
            keys = self.client.keys(f"{self.KEY_PREFIX}:*")
            if keys:
                self.client.delete(*keys)
                logger.info(f"Redis清除旧数据: {len(keys)}条")
            
            for task in tasks:
                key = self._build_key(task['id'])
                hash_data = self._build_hash_data(task)
                self.client.hset(key, mapping=hash_data)
            
            logger.info(f"Redis批量同步完成(Hash): {len(tasks)}条任务")
            return True
        except Exception as e:
            logger.error(f"Redis批量同步失败: {str(e)}")
            return False
    
    def update_task_field(self, task_id, field, value):
        """更新任务的单个字段"""
        if not self.client:
            return False
        
        try:
            key = self._build_key(task_id)
            if self.client.exists(key):
                self.client.hset(key, field, str(value))
                logger.debug(f"Redis更新字段: {key}.{field} = {value}")
                return True
            return False
        except Exception as e:
            logger.error(f"Redis更新字段失败: {str(e)}")
            return False
    
    def get_all_tasks(self):
        """获取Redis中所有任务状态"""
        if not self.client:
            return {}
        
        try:
            keys = self.client.keys(f"{self.KEY_PREFIX}:*")
            result = {}
            for key in keys:
                task_id = key.replace(f"{self.KEY_PREFIX}:", "")
                task_data = self.client.hgetall(key)
                result[task_id] = task_data
            return result
        except Exception as e:
            logger.error(f"Redis获取任务状态失败: {str(e)}")
            return {}


def get_most_free_gpu():
    """获取显存最空闲的GPU设备ID"""
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU")
            return 'cpu'
        
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.warning("没有可用的GPU，使用CPU")
            return 'cpu'
        
        free_memory = []
        for i in range(gpu_count):
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader', f'--id={i}'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    free_mem = int(result.stdout.strip().split('\n')[0])
                    free_memory.append((i, free_mem))
                    logger.info(f"GPU {i}: 空闲显存 {free_mem} MB")
                else:
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    cached_memory = torch.cuda.memory_reserved(i)
                    free_mem = (total_memory - cached_memory) / (1024 * 1024)  # 转换为MB
                    free_memory.append((i, free_mem))
                    logger.info(f"GPU {i}: 估算空闲显存 {free_mem:.0f} MB")
            except Exception as e:
                logger.warning(f"获取GPU {i} 显存信息失败: {e}")
                continue
        
        if not free_memory:
            logger.warning("无法获取任何GPU的显存信息，默认使用GPU 0")
            return '0'
        
        best_gpu = max(free_memory, key=lambda x: x[1])
        logger.info(f"选择最空闲的GPU: {best_gpu[0]}，空闲显存: {best_gpu[1]} MB")
        return str(best_gpu[0])
        
    except Exception as e:
        logger.error(f"获取最空闲GPU失败: {e}，默认使用GPU 0")
        return '0'


DatabaseManager = TrainerDatabaseManager  # 兼容旧测试/外部引用；实际业务类名使用 TrainerDatabaseManager。


class TrainingManager:
    def __init__(self, db_manager, redis_manager=None):
        self.db_manager = db_manager
        self.redis_manager = redis_manager
        self.script_path = Path(__file__).parent / "train_method" / "trainer.py"

    def _wait_for_dataset_ready(self, dataset_id, max_wait_time=600, wait_interval=2):
        """训练启动前等待异步数据集构建流程完成，避免边构建边训练"""
        if not dataset_id:
            return True, None

        dataset_id_str = str(dataset_id)
        waited_time = 0
        task_info = dataset_build_tasks.get(dataset_id_str)

        while task_info and task_info.get('status') == 'processing' and waited_time < max_wait_time:
            if waited_time == 0:
                logger.info(f"训练启动前等待数据集构建完成: dataset_id={dataset_id}")
            time.sleep(wait_interval)
            waited_time += wait_interval
            task_info = dataset_build_tasks.get(dataset_id_str)
            if waited_time % 10 == 0:
                logger.info(f"数据集仍在构建中: dataset_id={dataset_id}, 已等待 {waited_time} 秒")

        task_info = dataset_build_tasks.get(dataset_id_str)
        if task_info and task_info.get('status') == 'failed':
            error_msg = task_info.get('error', '未知错误')
            return False, f"数据集构建失败，禁止启动训练: {error_msg}"

        if task_info and task_info.get('status') == 'processing':
            error_msg = f"等待数据集构建超时（{max_wait_time}秒）"
            update_dataset_build_status(
                dataset_id, status='failed', error=error_msg, end_time=datetime.now()
            )
            return False, error_msg

        return True, None
    
    def build_command(self, task_params, task_id=None):
        """已废弃：训练不再通过 trainer.py CLI 子进程启动。"""
        raise RuntimeError("build_command 已废弃；请使用 start_training_direct 的进程内训练入口")
    
    def build_training_params(self, task_params, dataset_info=None, task_id=None):
        try:
            yaml_path = 'dataset.yaml'
            kpt_num = 0

            logger.info(f"=== 构建训练参数调试 ===")
            logger.info(f"task_params.dataset_id: {task_params.get('dataset_id')}")
            logger.info(f"传入的dataset_info: {dataset_info}")

            if dataset_info and dataset_info.get('yaml_address'):
                yaml_path = ensure_local_path(dataset_info.get('yaml_address'))
                logger.info(f"从传入的dataset_info获取数据集配置: {dataset_info.get('yaml_address')} -> {yaml_path}")
            elif task_params.get('dataset_id'):
                dataset_info = self.db_manager.get_dataset_info(task_params.get('dataset_id'))
                logger.info(f"从数据库查询的dataset_info: {dataset_info}")
                if dataset_info and dataset_info.get('yaml_address'):
                    yaml_path = ensure_local_path(dataset_info.get('yaml_address'))
                    logger.info(f"从数据库获取数据集配置: {dataset_info.get('yaml_address')} -> {yaml_path}")
                else:
                    logger.warning(f"数据库中yaml_address为空，使用默认值: {yaml_path}")
            else:
                logger.warning(f"没有dataset_id，使用默认yaml路径: {yaml_path}")

            if dataset_info and dataset_info.get('kpt_num'):
                try:
                    kpt_num = int(dataset_info.get('kpt_num') or 0)
                except (TypeError, ValueError):
                    kpt_num = 0

            logger.info(f"最终yaml_path: {yaml_path}")
            logger.info(f"=== 构建训练参数调试结束 ===")

            train_results_path = build_storage_local_path("train_results", require_exists=False)
            train_results_path.mkdir(parents=True, exist_ok=True)
            experiment_name = str(task_id) if task_id else str(task_params.get('id', 'exp'))

            train_params = {
                'dutyType': str(task_params.get('duty_type', 'detect')),
                'model_type': int(task_params.get('model_type', 0)),
                'model_size': str(task_params.get('model_size', 'n')),
                'incremental_model_address': str(resolve_storage_path_to_local(task_params.get('incremental_model_address')) or ''),
                'batch': int(task_params.get('batch', 8)),
                'device': str(task_params.get('device', '0')),
                'imgsz': int(task_params.get('img_size', 640)),
                'epoch': int(task_params.get('epochs', 100)),
                'yaml': str(yaml_path),
                'model': '',
                'project': str(train_results_path),
                'name': experiment_name,
                'lr0': float(task_params.get('lr0', 0.0001)),
                'lrf': float(task_params.get('lrf', 0.001)),
                'cos_lr': bool(task_params.get('cos_lr', True)),
                'warmup_epochs': int(task_params.get('warmup_epochs', 5)),
                'warmup_bias_lr': float(task_params.get('warmup_bias_lr', 0.1)),
                'momentum': float(task_params.get('momentum', 0.937)),
                'weight_decay': float(task_params.get('weight_decay', 0.0005)),
                'fliplr': float(task_params.get('fliplr', 0.5)),
                'amp': bool(task_params.get('amp', False)),
                'patience': int(task_params.get('patience', 0)),
                'save_period': int(task_params.get('save_period', -1)),
                'workers': int(task_params.get('workers', 8)),
                'resume': str(task_params.get('resume_path', '')),
                'kpt_num': kpt_num
            }

            return train_params

        except Exception as e:
            logger.error(f"构建训练参数失败: {str(e)}")
            return None

    def _release_training_reservation(self, task_id):
        """释放尚未移交给训练线程的占位记录（仅释放 starting 状态的占位）"""
        with _running_processes_lock:
            info = running_processes.get(task_id)
            if info and info.get('status') == 'starting':
                running_processes.pop(task_id, None)

    def start_training_direct(self, task_id):
        reservation_active = False
        try:
            task_params = self.db_manager.get_train_task(task_id)
            if not task_params:
                logger.error(f"未找到训练任务: {task_id}")
                return False, "未找到训练任务"

            logger.info(f"从数据库获取的字段名: {list(task_params.keys())}")
            logger.info(f"duty_type字段值: {task_params.get('duty_type', 'NOT_FOUND')}")
            logger.info(f"dutyType字段值: {task_params.get('dutyType', 'NOT_FOUND')}")

            # 单进程单训练：训练在本进程线程内执行，sys.stdout/stderr 与
            # ultralytics logger 都是全局状态，并发训练会互相污染日志与输出，
            # 任一任务结束又会恢复全局输出。因此原子地占位，确保同一时刻
            # 只有一个训练任务在跑（日志通过 TrainingLogWriter 落盘 + /log 接口读取）。
            with _running_processes_lock:
                if task_id in running_processes:
                    logger.warning(f"训练任务 {task_id} 已在运行中")
                    return False, "训练任务已在运行中"
                if running_processes:
                    other_ids = list(running_processes.keys())
                    logger.warning(
                        f"已有训练任务在运行，拒绝并发训练: running={other_ids}, 新请求={task_id}"
                    )
                    return False, (
                        f"已有训练任务在运行中（task_id={other_ids[0]}），"
                        f"当前仅支持单进程单训练，请等待其完成后再试"
                    )
                running_processes[task_id] = {
                    'status': 'starting',
                    'type': 'direct',
                    'start_time': datetime.now()
                }
            reservation_active = True

            db_device = task_params.get('device', '0')
            logger.info(f"数据库device字段值: {db_device}")
            
            db_device_str = str(db_device).strip()
            
            if db_device_str in ('0', '1'):
                selected_device = get_most_free_gpu()
                logger.info(f"device={db_device_str}，自动选择最空闲的GPU: {selected_device}")
                task_params['device'] = selected_device
            elif db_device_str == '2':
                selected_device = '0,1'
                logger.info(f"device=2，使用双卡训练: {selected_device}")
                task_params['device'] = selected_device
            else:
                logger.info(f"device={db_device_str}，保持原值")
            
            dataset_info = None
            if task_params.get('dataset_id'):
                ready, wait_message = self._wait_for_dataset_ready(task_params.get('dataset_id'))
                if not ready:
                    logger.error(wait_message)
                    self._release_training_reservation(task_id)
                    reservation_active = False
                    return False, wait_message
                dataset_info = self.db_manager.get_dataset_full_info(task_params.get('dataset_id'))

            train_params = self.build_training_params(task_params, dataset_info, task_id)
            if not train_params:
                logger.error("构建训练参数失败")
                self._release_training_reservation(task_id)
                reservation_active = False
                return False, "构建训练参数失败"
            
            logger.info(f"启动训练任务 {task_id}")
            logger.info(f"训练参数: {train_params}")
            
            sys.path.append(str(Path(__file__).parent / "train_method"))
            from trainer import YOLOv8Trainer
            
            train_results_path = build_storage_local_path("train_results", require_exists=False)
            train_results_path.mkdir(parents=True, exist_ok=True)
            
            trainer = YOLOv8Trainer(task_id=str(task_id), db_manager=self.db_manager, redis_manager=self.redis_manager)
            
            train_log_dir = build_storage_local_path("train_log", require_exists=False)
            log_file_path = train_log_dir / str(task_id) / 'training.log'
            logger.info(f"训练日志将保存到: {log_file_path}")
            
            def training_thread():
                start_time = datetime.now()
                csv_monitor_stop_event = threading.Event()
                csv_monitor_thread = None
                
                try:
                    self.db_manager.update_task_status(task_id, 'not_started')
                    self.db_manager.clear_train_chart(task_id)
                    current_pid = os.getpid()
                    self.db_manager.update_task_status(task_id, 'running', process_id=current_pid, start_time=start_time)
                    if self.redis_manager:
                        task_info_for_redis = self.db_manager.get_task_for_redis(task_id)
                        self.redis_manager.sync_task_status(task_id, 'running', task_info_for_redis)
                    logger.info(f"开始执行训练任务 {task_id}，开始时间: {start_time}，进程PID: {current_pid}")
                    
                    results_csv_path = train_results_path / str(task_id) / 'results.csv'
                    total_epochs = train_params.get('epoch', 100)
                    csv_monitor_thread = threading.Thread(
                        target=self._monitor_csv_and_save,
                        args=(task_id, results_csv_path, csv_monitor_stop_event, total_epochs),
                        daemon=True
                    )
                    csv_monitor_thread.start()
                    logger.info(f"CSV监控线程已启动: {results_csv_path}, total_epochs={total_epochs}")
                    
                    train_result = trainer.train(**train_params)
                    end_time = datetime.now()
                    
                    csv_monitor_stop_event.set()
                    if csv_monitor_thread and csv_monitor_thread.is_alive():
                        csv_monitor_thread.join(timeout=5)
                    
                    self._sync_csv_to_db(task_id, results_csv_path)
                    
                    if train_result == 'stopped' or (
                        getattr(trainer, 'training_stopped', False)
                        and getattr(trainer, 'stop_event', None)
                        and trainer.stop_event.is_set()
                    ):
                        self.db_manager.update_task_completion(task_id, end_time, 'stopped')
                        if self.redis_manager:
                            self.redis_manager.sync_task_status(task_id, 'stopped')
                        logger.info(f"训练任务 {task_id} 已停止，结束时间: {end_time}")
                    elif train_result:
                        best_model_storage_path = None
                        last_model_storage_path = None
                        chart_storage_path = None
                        log_storage_path = None
                        
                        try:
                            local_best_path = None
                            local_last_path = None
                            
                            if hasattr(trainer, 'best_model_path') and trainer.best_model_path:
                                local_best_path = str(Path(trainer.best_model_path).resolve())
                                logger.info(f"从trainer对象获取best模型路径: {local_best_path}")
                            if hasattr(trainer, 'last_model_path') and trainer.last_model_path:
                                local_last_path = str(Path(trainer.last_model_path).resolve())
                                logger.info(f"从trainer对象获取last模型路径: {local_last_path}")
                            
                            if not local_best_path or not local_last_path:
                                local_results_path = build_storage_local_path("train_results", require_exists=False)
                                experiment_name = str(task_id)
                                model_dir = local_results_path / experiment_name / 'weights'
                                
                                if model_dir.exists():
                                    if not local_best_path:
                                        best_model = model_dir / 'best.pt'
                                        if best_model.exists():
                                            local_best_path = str(best_model.resolve())
                                    
                                    if not local_last_path:
                                        last_model = model_dir / 'last.pt'
                                        if last_model.exists():
                                            local_last_path = str(last_model.resolve())
                            
                            if local_best_path and Path(local_best_path).exists():
                                best_model_storage_path = local_best_path

                            if local_last_path and Path(local_last_path).exists():
                                last_model_storage_path = local_last_path
                                
                        except Exception as me:
                            logger.warning(f"获取模型路径失败: {str(me)}")
                        
                        try:
                            chart_results_path = build_storage_local_path("train_results", require_exists=False)
                            experiment_name = str(task_id)
                            results_csv = chart_results_path / experiment_name / 'results.csv'
                            
                            if results_csv.exists():
                                chart_storage_path = copy_to_storage_and_get_path(results_csv, 'train_results', task_id)
                                self.db_manager.update_task_chart_path(task_id, chart_storage_path)
                                logger.info(f"图表存储路径已保存: {chart_storage_path}")
                            else:
                                logger.warning(f"图表文件不存在: {results_csv}")
                        except Exception as ce:
                            logger.warning(f"处理图表路径失败: {str(ce)}")
                        
                        try:
                            if hasattr(trainer, 'get_log_path'):
                                log_storage_path = trainer.get_log_path()
                                if log_storage_path:
                                    self.db_manager.update_task_log_path(task_id, log_storage_path)
                                    logger.info(f"日志存储路径已保存（从trainer获取）: {log_storage_path}")
                                else:
                                    logger.warning(f"trainer未返回日志存储路径")
                            elif log_file_path.exists():
                                log_storage_path = copy_to_storage_and_get_path(log_file_path, 'train_log', task_id)
                                self.db_manager.update_task_log_path(task_id, log_storage_path)
                                logger.info(f"日志存储路径已保存（从本地上传）: {log_storage_path}")
                            else:
                                logger.warning(f"日志文件不存在且 trainer 未提供存储路径: {log_file_path}")
                        except Exception as le:
                            logger.warning(f"处理日志路径失败: {str(le)}")
                        
                        self.db_manager.update_task_completion(
                            task_id, end_time, 'completed', 
                            trained_bestmodel_address=best_model_storage_path,
                            trained_lastmodel_address=last_model_storage_path
                        )
                        if self.redis_manager:
                            self.redis_manager.sync_task_status(task_id, 'completed')
                        # 读取并更新最终训练指标
                        final_metrics = None
                        try:
                            final_metrics = self._get_final_metrics_from_csv(results_csv_path)
                            if final_metrics:
                                self.db_manager.update_task_metrics(
                                    task_id,
                                    precision=final_metrics.get('precision'),
                                    recall=final_metrics.get('recall'),
                                    accuracy=final_metrics.get('accuracy')
                                )
                                logger.info(f"训练指标已更新: precision={final_metrics.get('precision')}, recall={final_metrics.get('recall')}, accuracy={final_metrics.get('accuracy')}")
                            else:
                                logger.warning(f"无法读取最终训练指标: task_id={task_id}")
                        except Exception as me:
                            logger.warning(f"更新训练指标失败: {str(me)}")
                        
                        try:
                            weights_id = self.db_manager.insert_train_weights(
                                train_task_id=task_id,
                                bestmodel_address=best_model_storage_path,
                                lastmodel_address=last_model_storage_path,
                                precision=final_metrics.get('precision') if final_metrics else None,
                                recall=final_metrics.get('recall') if final_metrics else None,
                                accuracy=final_metrics.get('accuracy') if final_metrics else None
                            )
                            if weights_id:
                                logger.info(f"训练权重记录已创建: weights_id={weights_id}, train_task_id={task_id}")
                            else:
                                logger.warning(f"创建训练权重记录失败: train_task_id={task_id}")
                        except Exception as we:
                            logger.warning(f"插入train_weights失败: {str(we)}")
                        
                        logger.info(f"训练任务 {task_id} 完成，结束时间: {end_time}")
                        if best_model_storage_path:
                            logger.info(f"Best 模型存储路径: {best_model_storage_path}")
                        if last_model_storage_path:
                            logger.info(f"Last 模型存储路径: {last_model_storage_path}")
                        if chart_storage_path:
                            logger.info(f"Chart 存储路径: {chart_storage_path}")
                        if log_storage_path:
                            logger.info(f"Log 存储路径: {log_storage_path}")
                    else:
                        self.db_manager.update_task_completion(task_id, end_time, 'failed')
                        if self.redis_manager:
                            self.redis_manager.sync_task_status(task_id, 'failed')
                        logger.error(f"训练任务 {task_id} 失败，结束时间: {end_time}")
                        
                except Exception as e:
                    end_time = datetime.now()
                    csv_monitor_stop_event.set()
                    if csv_monitor_thread and csv_monitor_thread.is_alive():
                        csv_monitor_thread.join(timeout=5)
                    self.db_manager.update_task_completion(task_id, end_time, 'failed')
                    if self.redis_manager:
                        self.redis_manager.sync_task_status(task_id, 'failed')
                    logger.error(f"训练任务 {task_id} 异常: {str(e)}，结束时间: {end_time}")
                finally:
                    csv_monitor_stop_event.set()
                    with _running_processes_lock:
                        if task_id in running_processes:
                            del running_processes[task_id]

            thread = threading.Thread(target=training_thread, daemon=True)

            # 先用完整记录覆盖占位（含未启动的线程对象），再启动线程：
            # 避免线程极快结束后其 finally 删除条目、随后再注册产生残留脏记录。
            with _running_processes_lock:
                running_processes[task_id] = {
                    'thread': thread,
                    'trainer': trainer,
                    'start_time': datetime.now(),
                    'task_params': task_params,
                    'type': 'direct'
                }
            # 所有权移交训练线程的 finally 清理，外层不再负责释放占位
            reservation_active = False

            try:
                thread.start()
            except Exception:
                with _running_processes_lock:
                    running_processes.pop(task_id, None)
                raise

            logger.info(f"训练任务 {task_id} 启动成功")
            return True, f"训练任务启动成功"

        except Exception as e:
            logger.error(f"启动训练任务失败: {str(e)}")
            return False, f"启动训练任务失败: {str(e)}"
        finally:
            # 任何未移交所有权的提前返回/异常路径，释放占位避免训练永久被锁
            if reservation_active:
                self._release_training_reservation(task_id)
    
    def start_training(self, task_id):
        """兼容旧调用路径：统一走进程内训练线程。"""
        logger.warning("start_training 子进程入口已废弃，自动切换到 start_training_direct")
        return self.start_training_direct(task_id)
    
    def stop_training(self, task_id):
        try:
            logger.info(f"收到停止训练请求: task_id={task_id}")

            with _running_processes_lock:
                process_info = running_processes.get(task_id)
                if not process_info:
                    logger.warning(f"任务 {task_id} 不在当前进程运行记录中")
                    return False, "训练任务未在当前进程中运行"

                if 'thread' in process_info:
                    trainer = process_info.get('trainer')
                    if not trainer:
                        return False, "训练任务缺少 trainer 实例，无法发送停止信号"

                    trainer.stop_training()
                    if hasattr(trainer, 'stop_event'):
                        trainer.stop_event.set()
                    trainer.training_stopped = True

                    if hasattr(trainer, 'model_trainer') and trainer.model_trainer:
                        trainer.model_trainer.stop = True
                        if hasattr(trainer.model_trainer, 'stop_training'):
                            trainer.model_trainer.stop_training = True
                        logger.info(f"已设置 YOLO trainer 停止标志: {task_id}")

                    process_info['status'] = 'stopping'
                    process_info['stop_requested_at'] = datetime.now()
                    logger.info(f"已发送协作式停止信号: task_id={task_id}")
                    return True, "已发送停止信号，训练线程将协作式停止"

                if 'process' in process_info:
                    logger.error("检测到已废弃的子进程训练记录，无法通过协作式停止处理")
                    return False, "当前任务来自已废弃的子进程训练入口，无法协作式停止"

                return False, "训练任务运行记录格式不正确"

        except Exception as e:
            logger.error(f"停止训练任务失败: {str(e)}")
            return False, f"停止训练任务失败: {str(e)}"
    
    def _parse_csv_row(self, row):
        return {
            'epoch': int(row.get('epoch', 0)),
            'time': float(row.get('time', 0)) if row.get('time') else None,
            'train_box_loss': float(row.get('train/box_loss', 0)) if row.get('train/box_loss') else None,
            'train_cls_loss': float(row.get('train/cls_loss', 0)) if row.get('train/cls_loss') else None,
            'train_dfl_loss': float(row.get('train/dfl_loss', 0)) if row.get('train/dfl_loss') else None,
            'metrics_precision_B': float(row.get('metrics/precision(B)', 0)) if row.get('metrics/precision(B)') else None,
            'metrics_recall_B': float(row.get('metrics/recall(B)', 0)) if row.get('metrics/recall(B)') else None,
            'metrics_mAP50_B': float(row.get('metrics/mAP50(B)', 0)) if row.get('metrics/mAP50(B)') else None,
            'metrics_mAP50_95_B': float(row.get('metrics/mAP50-95(B)', 0)) if row.get('metrics/mAP50-95(B)') else None,
            'val_box_loss': float(row.get('val/box_loss', 0)) if row.get('val/box_loss') else None,
            'val_cls_loss': float(row.get('val/cls_loss', 0)) if row.get('val/cls_loss') else None,
            'val_dfl_loss': float(row.get('val/dfl_loss', 0)) if row.get('val/dfl_loss') else None,
            'lr_pg0': float(row.get('lr/pg0', 0)) if row.get('lr/pg0') else None,
            'lr_pg1': float(row.get('lr/pg1', 0)) if row.get('lr/pg1') else None,
            'lr_pg2': float(row.get('lr/pg2', 0)) if row.get('lr/pg2') else None
        }
    
    def _monitor_csv_and_save(self, task_id, csv_path, stop_event, total_epochs=None):
        last_line_count = 0
        check_interval = 2
        training_start_time = None
        
        logger.info(f"开始监控CSV文件: {csv_path}")
        
        while not stop_event.is_set():
            try:
                if csv_path.exists():
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        current_line_count = len(rows)
                        
                        if current_line_count > last_line_count:
                            new_rows = rows[last_line_count:]
                            parsed_rows = []
                            for row in new_rows:
                                try:
                                    parsed_rows.append(self._parse_csv_row(row))
                                except Exception as e:
                                    logger.error(f"解析CSV行失败: {str(e)}")
                            saved = self.db_manager.upsert_train_chart_batch(task_id, parsed_rows)
                            last_line_count = current_line_count
                            logger.info(f"CSV监控: 已写入 {current_line_count} 条记录到数据库 (本轮 {saved})")
                            
                            if self.redis_manager and total_epochs:
                                try:
                                    current_epoch = current_line_count
                                    
                                    if training_start_time is None:
                                        training_start_time = time.time()
                                    
                                    eta_seconds = 0
                                    if current_epoch > 0 and training_start_time:
                                        elapsed_time = time.time() - training_start_time
                                        avg_epoch_time = elapsed_time / current_epoch
                                        remaining_epochs = total_epochs - current_epoch
                                        eta_seconds = int(avg_epoch_time * remaining_epochs)
                                    
                                    self.redis_manager.update_task_field(task_id, 'current_epoch', current_epoch)
                                    self.redis_manager.update_task_field(task_id, 'eta_seconds', eta_seconds)
                                    logger.info(f"Redis进度更新: task_id={task_id}, current_epoch={current_epoch}/{total_epochs}, eta_seconds={eta_seconds}")
                                except Exception as e:
                                    logger.error(f"更新Redis训练进度失败: {str(e)}")
                
            except Exception as e:
                logger.error(f"监控CSV文件时出错: {str(e)}")
            
            stop_event.wait(check_interval)
        
        logger.info(f"CSV监控线程已停止: {csv_path}")
    
    def _sync_csv_to_db(self, task_id, csv_path):
        """同步CSV文件到数据库"""
        try:
            if not csv_path.exists():
                logger.warning(f"CSV文件不存在，跳过最终同步: {csv_path}")
                return
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                parsed_rows = []
                for row in reader:
                    try:
                        parsed_rows.append(self._parse_csv_row(row))
                    except Exception as e:
                        logger.error(f"最终同步解析CSV行失败 epoch={row.get('epoch')}: {str(e)}")
                self.db_manager.upsert_train_chart_batch(task_id, parsed_rows)

            logger.info(f"CSV最终同步完成: {csv_path}")
        except Exception as e:
            logger.error(f"最终同步CSV到数据库失败: {str(e)}")
    
    def _get_final_metrics_from_csv(self, csv_path):
        try:
            if not csv_path.exists():
                logger.warning(f"CSV文件不存在，无法读取最终指标: {csv_path}")
                return None
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    logger.warning(f"CSV文件为空，无法读取最终指标: {csv_path}")
                    return None
                
                last_row = rows[-1]
                
                precision = float(last_row.get('metrics/precision(B)', 0)) if last_row.get('metrics/precision(B)') else None
                recall = float(last_row.get('metrics/recall(B)', 0)) if last_row.get('metrics/recall(B)') else None
                accuracy = float(last_row.get('metrics/mAP50-95(B)', 0)) if last_row.get('metrics/mAP50-95(B)') else None
                
                logger.info(f"读取最终指标: precision={precision}, recall={recall}, accuracy(mAP50-95)={accuracy}")
                
                return {
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy
                }
        except Exception as e:
            logger.error(f"读取最终指标失败: {str(e)}")
            return None
    
    def _monitor_process(self, task_id, process):
        """已废弃：训练不再通过 trainer.py CLI 子进程启动。"""
        raise RuntimeError("_monitor_process 已废弃；训练状态由进程内训练线程收敛")

db_manager = TrainerDatabaseManager(DB_CONFIG)

redis_config = trainer_config.get_redis_config()
redis_manager = RedisManager(redis_config)

def init_redis_sync():
    try:
        tasks = db_manager.get_queued_running_tasks()
        if tasks:
            redis_manager.sync_all_tasks(tasks)
            logger.info(f"启动时Redis同步完成: {len(tasks)}条queued/running任务")
        else:
            # 清空Redis中的旧数据
            redis_manager.sync_all_tasks([])
            logger.info("启动时Redis同步完成: 无queued/running任务")
    except Exception as e:
        logger.error(f"启动时Redis同步失败: {str(e)}")

init_redis_sync()

training_manager = TrainingManager(db_manager, redis_manager)


@app.route('/algorithm/models/convert', methods=['POST'])
def convert_trained_model():
    """将训练完成的PT模型转换为目标平台模型"""
    try:
        data = request.get_json(silent=True) or {}
        logger.info(f"[模型转换] 收到转换请求: remote_addr={request.remote_addr}, body={data}")
        model_id = data.get('model_id')
        target_platform = str(data.get('target_platform') or '').strip().lower()

        if model_id is None:
            return jsonify({"code": 1, "data": {"message": "缺少必需参数 model_id"}}), 400
        try:
            model_id = validate_positive_id(model_id, "model_id")
        except ValidationError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400
        if not target_platform:
            return jsonify({"code": 1, "data": {"message": "缺少必需参数 target_platform"}}), 400
        if target_platform not in MODEL_CONVERT_SUPPORTED_PLATFORMS:
            return jsonify({
                "code": 1,
                "data": {
                    "message": f"不支持的target_platform: {target_platform}",
                    "supported": sorted(MODEL_CONVERT_SUPPORTED_PLATFORMS)
                }
            }), 400

        convert_format = MODEL_CONVERT_FORMAT_BY_PLATFORM.get(target_platform)
        if not convert_format:
            return jsonify({
                "code": 1,
                "data": {
                    "message": f"{target_platform}模型转换暂未实现",
                    "model_id": model_id,
                    "target_platform": target_platform
                }
            }), 501

        weight_info = db_manager.get_trained_weight_for_convert(model_id)
        if not weight_info:
            return jsonify({"code": 1, "data": {"message": f"未找到model_id={model_id}的训练权重记录"}}), 404

        source_path = weight_info.get('bestmodel_address')
        if not source_path:
            return jsonify({"code": 1, "data": {"message": f"model_id={model_id}缺少bestmodel_address"}}), 400

        task_id = f"convert_{model_id}_{target_platform}_{int(time.time())}"
        convert_key = (model_id, target_platform)
        with _model_convert_tasks_lock:
            existing_task = model_convert_tasks.get(convert_key)
            if existing_task and existing_task.get('status') == 'processing':
                logger.warning(
                    f"[模型转换] 重复转换请求已忽略: model_id={model_id}, "
                    f"target_platform={target_platform}, running_task_id={existing_task.get('task_id')}"
                )
                return jsonify({
                    "code": 0,
                    "data": {
                        "message": "模型转换任务已在进行中",
                        "model_id": model_id,
                        "target_platform": target_platform,
                        "convert_format": convert_format,
                        "status": MODEL_CONVERT_STATUS_RUNNING,
                        "task_id": existing_task.get('task_id')
                    }
                })
            model_convert_tasks[convert_key] = {
                'status': 'processing',
                'task_id': task_id,
                'start_time': datetime.now()
            }

        if not db_manager.upsert_model_convert_processing(
            model_id=model_id,
            type_code=target_platform,
            source_path=source_path,
            convert_format=convert_format,
            task_id=task_id
        ):
            with _model_convert_tasks_lock:
                model_convert_tasks.pop(convert_key, None)
            return jsonify({"code": 1, "data": {"message": "写入模型转换初始状态失败"}}), 500

        def async_convert():
            convert_status = 'failed'
            try:
                raw_model_name = (
                    weight_info.get('model_name') or
                    weight_info.get('name') or
                    f"model_{model_id}"
                )
                safe_model_name = sanitize_model_name(raw_model_name, fallback=f"model_{model_id}")
                duty_type = str(weight_info.get('duty_type') or 'detect').strip().lower()

                local_pt_path = Path(ensure_local_path(source_path))
                if not local_pt_path.exists():
                    raise FileNotFoundError(f"源PT模型不存在: {local_pt_path}")

                local_convert_dir = build_storage_local_path(
                    'model_convert', str(model_id), target_platform, require_exists=False
                )
                if local_convert_dir.exists():
                    safe_rmtree(local_convert_dir)
                local_convert_dir.mkdir(parents=True, exist_ok=True)

                logger.info(
                    f"[模型转换] 开始{target_platform}转换: model_id={model_id}, "
                    f"model_name={raw_model_name}, safe_model_name={safe_model_name}, "
                    f"duty_type={duty_type}, source={local_pt_path}"
                )

                export_pt_path = prepare_pt_for_conversion(local_pt_path, local_convert_dir, safe_model_name)
                if target_platform in {'bm1684', 'bm1684x'}:
                    dataset_info = None
                    dataset_id = weight_info.get('dataset_id')
                    if dataset_id:
                        dataset_info = db_manager.get_dataset_record(dataset_id)
                    if not dataset_info:
                        raise FileNotFoundError(f"未找到model_id={model_id}关联的数据集: dataset_id={dataset_id}")

                    onnx_path = export_pt_to_onnx(export_pt_path)
                    converted_model_path = run_bm1684_convert_pipeline(
                        model_name=safe_model_name,
                        onnx_path=onnx_path,
                        dataset_info=dataset_info,
                        local_convert_dir=local_convert_dir,
                        work_dir_name=f"{safe_model_name}_{model_id}_{target_platform}",
                        chip=target_platform
                    )
                elif target_platform == 'rk3588':
                    onnx_path = export_pt_to_rknn_optimized_onnx(export_pt_path)
                    converted_model_path = run_rk3588_convert_pipeline(
                        model_name=safe_model_name,
                        onnx_path=onnx_path,
                        local_convert_dir=local_convert_dir,
                        duty_type=duty_type,
                        dtype='i8'
                    )
                else:
                    raise NotImplementedError(f"{target_platform}模型转换暂未实现")

                convert_path = normalize_address_for_db(str(converted_model_path.resolve()))
                if PathHandler.is_legacy_remote_uri(convert_path):
                    upload_file_to_storage(
                        converted_model_path,
                        parse_storage_relative_path(convert_path),
                        content_type='application/octet-stream'
                    )
                    verify_storage_object_matches_file(converted_model_path, convert_path)
                    logger.info(f"[模型转换] {convert_format}已保存到本地工作区: {convert_path}")
                file_size_mb = round(converted_model_path.stat().st_size / (1024 * 1024), 2)

                if not db_manager.update_model_convert_success(
                    model_id=model_id,
                    type_code=target_platform,
                    source_path=source_path,
                    convert_path=convert_path,
                    convert_format=convert_format,
                    file_size_mb=file_size_mb,
                    task_id=task_id
                ):
                    raise RuntimeError("写入模型转换成功状态失败")

                logger.info(
                    f"[模型转换] {target_platform}转换完成: model_id={model_id}, convert_path={convert_path}, "
                    f"file_size={file_size_mb}MB"
                )
                convert_status = 'completed'

            except Exception as e:
                error_message = str(e)
                logger.error(f"[模型转换] {target_platform}转换失败: model_id={model_id}, error={error_message}")
                logger.error(traceback.format_exc())
                db_manager.update_model_convert_failure(
                    model_id=model_id,
                    type_code=target_platform,
                    source_path=source_path,
                    error_message=error_message,
                    convert_format=convert_format,
                    task_id=task_id
                )
            finally:
                with _model_convert_tasks_lock:
                    model_convert_tasks.pop(convert_key, None)
                logger.info(
                    f"[模型转换] 转换任务锁已释放: model_id={model_id}, "
                    f"target_platform={target_platform}, task_id={task_id}, status={convert_status}"
                )

        thread = threading.Thread(target=async_convert, daemon=True)
        thread.start()

        return jsonify({
            "code": 0,
            "data": {
                "message": "模型转换任务已启动",
                "model_id": model_id,
                "target_platform": target_platform,
                "convert_format": convert_format,
                "status": MODEL_CONVERT_STATUS_RUNNING,
                "task_id": task_id
            }
        })

    except Exception as e:
        return handle_api_exception(e, "模型转换")


@app.route('/algorithm/train', methods=['POST'])
def manage_training():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "data": {"message": "请求数据为空"}}), 400
        
        train_task_id = data.get('train_task_id')
        status = data.get('status')

        if train_task_id is None or status is None:
            return jsonify({"code": 1, "data": {"message": "缺少必需参数 train_task_id 或 status"}}), 400

        try:
            train_task_id = validate_task_id(train_task_id)
        except ValidationError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400

        if not isinstance(status, int) or status not in (0, 1):
            return jsonify({"code": 1, "data": {"message": "status 必须是 0（启动）或 1（停止）"}}), 400

        logger.info(f"收到训练管理请求: task_id={train_task_id}, status={status}")
        
        if status == 0:
            success, message = training_manager.start_training_direct(train_task_id)
            if success:
                return jsonify({"code": 0, "data": {"message": message, "task_id": train_task_id}})
            else:
                return jsonify({"code": 1, "data": {"message": message}})
        
        elif status == 1:
            success, message = training_manager.stop_training(train_task_id)
            if success:
                return jsonify({
                    'code': 0,
                    'data': {
                        'message': message,
                        'train_task_id': train_task_id,
                        'action': 'stop'
                    }
                }), 200
            else:
                return jsonify({
                    'code': 1,
                    'data': {
                        'message': message,
                        'train_task_id': train_task_id,
                        'action': 'stop'
                    }
                }), 500
            
        else:
            return jsonify({
                'code': 1,
                'data': {
                    'message': '无效的status值，应为0（启动）或1（停止）'
                }
            }), 400
            
    except Exception as e:
        return handle_api_exception(e, "训练管理请求")


@app.route('/algorithm/datasets/yaml', methods=['POST'])
def generate_dataset_yaml():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "data": {"message": "请求数据为空"}}), 400
        
        dataset_id = data.get('dataset_id')
        if dataset_id is None:
            return jsonify({"code": 1, "data": {"message": "缺少必需参数 dataset_id"}}), 400

        try:
            dataset_id = validate_dataset_id(dataset_id)
        except ValidationError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400

        logger.info(f"收到生成数据集YAML请求: dataset_id={dataset_id}")

        dataset_info = db_manager.get_dataset_full_info(dataset_id)
        if not dataset_info:
            return jsonify({"code": 1, "data": {"message": f"未找到dataset_id={dataset_id}的数据集信息"}}), 404
        
        dataset_address = dataset_info.get('dataset_address')
        labels = dataset_info.get('labels')
        label_num = dataset_info.get('label_num')
        duty_type = str(dataset_info.get('duty_type') or '').strip().lower()
        draw_type = str(dataset_info.get('draw_type') or '').strip().lower()
        
        if not dataset_address or not labels or not label_num:
            return jsonify({"code": 1, "data": {"message": "数据集信息不完整，缺少dataset_address、labels或label_num"}}), 400
        
        logger.info(f"数据集信息: address={dataset_address}, labels={labels}, label_num={label_num}")

        local_dataset_address = ensure_local_path(dataset_address)
        logger.info(f"使用本地工作区路径生成YAML: {dataset_address} -> {local_dataset_address}")
        Path(local_dataset_address).mkdir(parents=True, exist_ok=True)
        
        from utils import DatasetYamlGenerator
        
        generator = DatasetYamlGenerator()
        yaml_output_path = Path(local_dataset_address) / "dataset.yaml"
        duty_type_for_yaml = 'pose' if duty_type == 'pose' or draw_type == 'point' else None
        kpt_labels = None
        kpt_num = None
        if duty_type_for_yaml == 'pose':
            try:
                kpt_labels, kpt_num = resolve_pose_keypoint_config(dataset_id, dataset_info, persist=True)
            except ValueError as e:
                return jsonify({"code": 1, "data": {"message": str(e)}}), 400

        result = generator.generate_silent(
            dataset_address=local_dataset_address,
            labels=labels,
            label_num=label_num,
            output_path=str(yaml_output_path),
            duty_type=duty_type_for_yaml,
            kpt_num=kpt_num
        )
        
        if result['success']:
            local_yaml_path = result['yaml_path']
            logger.info(f"YAML文件生成成功: {local_yaml_path}")
            
            yaml_storage_path = normalize_address_for_db(local_yaml_path)
            if PathHandler.is_legacy_remote_uri(yaml_storage_path):
                upload_file_to_storage(
                    Path(local_yaml_path),
                    parse_storage_relative_path(yaml_storage_path),
                    content_type='application/x-yaml'
                )
                logger.info(f"YAML文件已保存到本地工作区: {yaml_storage_path}")

            if db_manager.update_dataset_yaml_address(dataset_id, yaml_storage_path):
                logger.info(f"数据库yaml_address字段更新成功: {yaml_storage_path}")
                return jsonify({
                    "code": 0, 
                    "data": {
                        "message": "YAML文件生成成功",
                        "dataset_id": dataset_id,
                        "yaml_path": yaml_storage_path,
                        "local_yaml_path": local_yaml_path,
                        "kpt_labels": kpt_labels,
                        "kpt_num": kpt_num
                    }
                })
            else:
                logger.warning("YAML文件生成成功但数据库更新失败")
                return jsonify({
                    "code": 0, 
                    "data": {
                        "message": "YAML文件生成成功，但数据库更新失败",
                        "dataset_id": dataset_id,
                        "yaml_path": yaml_storage_path
                    }
                })
        else:
            error_msg = result.get('error', '未知错误')
            logger.error(f"YAML文件生成失败: {error_msg}")
            return jsonify({"code": 1, "data": {"message": f"YAML文件生成失败: {error_msg}"}}), 500
            
    except Exception as e:
        return handle_api_exception(e, "生成数据集YAML")


@app.route('/algorithm/datasets/json2txt', methods=['POST'])
def convert_dataset_json2txt():
    """处理训练集和验证集"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "data": {"message": "请求数据为空"}}), 400
        
        dataset_id = data.get('dataset_id')
        if dataset_id is None:
            return jsonify({"code": 1, "data": {"message": "缺少必需参数 dataset_id"}}), 400

        try:
            dataset_id = validate_dataset_id(dataset_id)
        except ValidationError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400

        logger.info(f"收到JSON转TXT请求: dataset_id={dataset_id}")

        dataset_info = db_manager.get_dataset_json2txt_info(dataset_id)
        if not dataset_info:
            return jsonify({"code": 1, "data": {"message": f"未找到dataset_id={dataset_id}的数据集信息"}}), 404
        
        duty_type = str(dataset_info.get('duty_type') or '').strip().lower()
        original_train_data_address = dataset_info.get('original_train_data_address')
        original_val_data_address = dataset_info.get('original_val_data_address')
        labels_str = dataset_info.get('labels')
        
        if not duty_type or not labels_str:
            return jsonify({
                "code": 1,
                "data": {
                    "message": "数据集信息不完整，缺少duty_type或labels"
                }
            }), 400
        
        if not original_train_data_address and not original_val_data_address:
            return jsonify({
                "code": 1,
                "data": {
                    "message": "数据集信息不完整，original_train_data_address和original_val_data_address都为空"
                }
            }), 400
        
        try:
            labels_list = parse_labels(labels_str)
        except Exception as e:
            logger.error(f"解析标签失败: {str(e)}")
            return jsonify({"code": 1, "data": {"message": f"解析标签失败: {str(e)}"}}), 500

        kpt_labels = None
        kpt_num = None
        if str(duty_type).strip().lower() == 'pose':
            try:
                kpt_labels, kpt_num = resolve_pose_keypoint_config(
                    dataset_id,
                    dataset_info,
                    split_addresses=[original_train_data_address, original_val_data_address],
                    persist=True,
                )
                logger.info(f"同步JSON转TXT pose 关键点配置: kpt_labels={kpt_labels}, kpt_num={kpt_num}")
            except ValueError as e:
                return jsonify({"code": 1, "data": {"message": str(e)}}), 400
        
        train_result = None
        val_result = None
        train_labels_dir = None
        val_labels_dir = None
        
        # 处理训练集
        if original_train_data_address:
            try:
                local_train_address = resolve_dataset_dir_for_conversion(original_train_data_address, '训练集')
                logger.info(f"处理训练集路径: {original_train_data_address} -> {local_train_address}")
                train_labels_path = Path(local_train_address) / 'labels'
                if train_labels_path.exists():
                    safe_rmtree(train_labels_path)
                
                train_converter = UnifiedJsonConverter(
                    duty_type=duty_type,
                    original_train_data_address=local_train_address,
                    predefined_labels=labels_list,
                    kpt_labels=kpt_labels,
                    kpt_num=kpt_num,
                )
                train_result = train_converter.convert()
                train_labels_dir = train_converter.labels_dir
                upload_converted_labels(local_train_address, original_train_data_address)
                logger.info(f"训练集JSON转TXT完成: {train_result}")
                    
            except FileNotFoundError as e:
                logger.error(f"训练集JSON转TXT失败，路径不存在: {str(e)}")
                return jsonify({"code": 1, "data": {"message": f"训练集路径不存在: {str(e)}"}}), 400
            except Exception as e:
                logger.error(f"训练集JSON转TXT转换失败: {str(e)}")
                return jsonify({"code": 1, "data": {"message": f"训练集JSON转TXT转换失败: {str(e)}"}}), 500
        
        # 处理验证集
        if original_val_data_address:
            try:
                local_val_address = resolve_dataset_dir_for_conversion(original_val_data_address, '验证集')
                logger.info(f"处理验证集路径: {original_val_data_address} -> {local_val_address}")
                val_labels_path = Path(local_val_address) / 'labels'
                if val_labels_path.exists():
                    safe_rmtree(val_labels_path)
                
                val_converter = UnifiedJsonConverter(
                    duty_type=duty_type,
                    original_train_data_address=local_val_address,
                    predefined_labels=labels_list,
                    kpt_labels=kpt_labels,
                    kpt_num=kpt_num,
                )
                val_result = val_converter.convert()
                val_labels_dir = val_converter.labels_dir
                upload_converted_labels(local_val_address, original_val_data_address)
                logger.info(f"验证集JSON转TXT完成: {val_result}")
                    
            except FileNotFoundError as e:
                logger.error(f"验证集JSON转TXT失败，路径不存在: {str(e)}")
                return jsonify({"code": 1, "data": {"message": f"验证集路径不存在: {str(e)}"}}), 400
            except Exception as e:
                logger.error(f"验证集JSON转TXT转换失败: {str(e)}")
                return jsonify({"code": 1, "data": {"message": f"验证集JSON转TXT转换失败: {str(e)}"}}), 500
        
        return jsonify({
            "code": 0,
            "data": {
                "message": "JSON转TXT成功",
                "dataset_id": dataset_id,
                "train_stats": train_result,
                "val_stats": val_result,
                "train_labels_dir": str(train_labels_dir) if train_labels_dir else None,
                "val_labels_dir": str(val_labels_dir) if val_labels_dir else None,
                "kpt_labels": kpt_labels,
                "kpt_num": kpt_num
            }
        })
        
    except Exception as e:
        return handle_api_exception(e, "JSON转TXT")


def _copy_labels_to_storage(labels_dir, source_storage_path):
    """把 json2txt 生成的 labels 写回原始 split 本地目录"""
    local_dataset_dir = Path(labels_dir).parent
    upload_converted_labels(local_dataset_dir, source_storage_path)


def parse_json_array_field(value):
    """解析数据库中保存的 JSON 数组字段。"""
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    if isinstance(value, list):
        raw_items = value
    else:
        value_str = str(value).strip()
        if not value_str:
            return []
        try:
            raw_items = json.loads(value_str)
        except json.JSONDecodeError:
            raw_items = [item.strip().strip('"\'') for item in value_str.strip('[]').split(',')]

    if not isinstance(raw_items, list):
        return []
    return [str(item).strip() for item in raw_items if str(item).strip()]


def parse_bool_param(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def resolve_pose_keypoint_config(dataset_id, dataset_info, split_addresses=None, persist=True):
    """解析 pose 的关键点顺序和数量，优先使用 DB kpt_labels，不完整时从 JSON 重新扫描。"""
    duty_type = str(dataset_info.get('duty_type') or '').strip().lower()
    draw_type = str(dataset_info.get('draw_type') or '').strip().lower()
    if duty_type != 'pose' and draw_type != 'point':
        return None, None

    db_kpt_labels = sort_keypoint_labels(parse_json_array_field(dataset_info.get('kpt_labels')))
    db_kpt_num_raw = dataset_info.get('kpt_num')
    try:
        db_kpt_num = int(db_kpt_num_raw or 0)
    except (TypeError, ValueError):
        db_kpt_num = 0

    kpt_labels = db_kpt_labels
    if not split_addresses:
        original_data_address = dataset_info.get('original_data_address')
        if original_data_address:
            base_path = str(original_data_address).rstrip('/').rsplit('/', 1)[0]
            split_addresses = [f"{base_path}/train", f"{base_path}/valid"]

    if not kpt_labels and split_addresses:
        kpt_labels = sort_keypoint_labels(_collect_keypoint_labels_from_split_dirs(split_addresses))

    kpt_num = len(kpt_labels) if kpt_labels else db_kpt_num
    if kpt_num <= 0:
        raise ValueError("pose 任务缺少 kpt_labels/kpt_num，请先调用 /algorithm/datasets/stats 或确认 point 落在 rectangle 内")

    if kpt_labels and len(kpt_labels) != kpt_num:
        kpt_num = len(kpt_labels)

    should_persist = (
        persist
        and kpt_labels
        and (
            kpt_labels != db_kpt_labels
            or kpt_num != db_kpt_num
        )
    )
    if should_persist:
        db_manager.update_dataset_keypoint_config(dataset_id, kpt_labels, kpt_num)

    return kpt_labels, kpt_num


def _build_yolo_dataset_structure(dataset_id, dataset_info, heartbeat=None):
    """构建 train/valid YOLO 数据集目录结构的共享核心。

    create_dataset（同步路由）与 _process_create_dataset（异步管线阶段）此前各自
    维护了一份几乎逐行相同的实现——之前的"删空再拷贝"数据丢失 bug 就同时存在于
    两份拷贝、需要改两遍。这里抽出共享核心，仅承载两者完全一致的机械逻辑：
    地址校验 -> 物化源目录 -> 建目录 -> 校验源 -> 并行复制 -> 计数校验 -> 持久化。
    数据库地址更新与响应构建留在各自调用方（它们的返回契约确实不同）。

    heartbeat: 可选，签名同 update_dataset_build_heartbeat 但已绑定 dataset_id
               （即 heartbeat(stage=, sub_step=, message=, ...)）。为 None 时
               （同步路由场景）不上报任何进度心跳。
    materialize_dataset_dir_for_create 在源目录缺失时抛 FileNotFoundError，
    这里不捕获，由调用方按各自语义处理（路由 -> 400，管线 -> 阶段异常）。

    返回 dict：
      失败 {'success': False, 'error_kind': 'invalid'|'io', 'error': str, 'failed_files'?: list}
      成功 {'success': True, 'dataset_storage_address': str, 'local_dataset_address': str,
            'counts': {'train_images','train_labels','valid_images','valid_labels'}}
    """
    def _copy(src_dir, dest_dir, sub_step, action_name):
        callback = None
        if heartbeat:
            heartbeat(stage='build', sub_step=sub_step, message=f"开始{action_name}",
                      processed_files=0, total_files=None, force_log=True)

            def callback(processed_files, total_files):
                heartbeat(stage='build', sub_step=sub_step,
                          message=f"{action_name}: {processed_files}/{total_files}",
                          processed_files=processed_files, total_files=total_files,
                          force_log=True)
        return copy_directory_files_parallel(src_dir, dest_dir, progress_callback=callback)

    original_train_data_address = dataset_info.get('original_train_data_address')
    original_val_data_address = dataset_info.get('original_val_data_address')
    if not original_train_data_address:
        return {'success': False, 'error_kind': 'invalid',
                'error': '数据集信息不完整，缺少original_train_data_address'}
    if not original_val_data_address:
        return {'success': False, 'error_kind': 'invalid',
                'error': '数据集信息不完整，缺少original_val_data_address'}

    local_train_address = materialize_dataset_dir_for_create(original_train_data_address, '训练集')
    local_val_address = materialize_dataset_dir_for_create(original_val_data_address, '验证集')

    dataset_storage_address = build_storage_path('train_data', str(dataset_id))
    output_root = build_storage_local_path('train_data', str(dataset_id), require_exists=False)
    datasets_root = output_root / "datasets"
    train_images_dir = datasets_root / "train" / "images"
    train_labels_dir = datasets_root / "train" / "labels"
    valid_images_dir = datasets_root / "valid" / "images"
    valid_labels_dir = datasets_root / "valid" / "labels"

    if datasets_root.exists():
        safe_rmtree(datasets_root)
    for d in (train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    src_train_images = Path(local_train_address) / "images"
    src_train_labels = Path(local_train_address) / "labels"
    if not src_train_images.exists():
        return {'success': False, 'error_kind': 'invalid', 'error': f"训练集images目录不存在: {src_train_images}"}
    if not src_train_labels.exists():
        return {'success': False, 'error_kind': 'invalid', 'error': f"训练集labels目录不存在: {src_train_labels}"}

    train_images_count = _copy(src_train_images, train_images_dir, 'train_images', '复制训练集图片')
    train_labels_count = _copy(src_train_labels, train_labels_dir, 'train_labels', '复制训练集标签')

    src_val_images = Path(local_val_address) / "images"
    src_val_labels = Path(local_val_address) / "labels"
    if not src_val_images.exists():
        return {'success': False, 'error_kind': 'invalid', 'error': f"验证集images目录不存在: {src_val_images}"}
    if not src_val_labels.exists():
        return {'success': False, 'error_kind': 'invalid', 'error': f"验证集labels目录不存在: {src_val_labels}"}

    valid_images_count = _copy(src_val_images, valid_images_dir, 'valid_images', '复制验证集图片')
    valid_labels_count = _copy(src_val_labels, valid_labels_dir, 'valid_labels', '复制验证集标签')

    if train_images_count == 0:
        return {'success': False, 'error_kind': 'invalid', 'error': f"训练集图片为空: {src_train_images}"}
    if train_labels_count == 0:
        return {'success': False, 'error_kind': 'invalid', 'error': f"训练集标签为空: {src_train_labels}"}
    if valid_images_count == 0:
        return {'success': False, 'error_kind': 'invalid', 'error': f"验证集图片为空: {src_val_images}"}
    if valid_labels_count == 0:
        return {'success': False, 'error_kind': 'invalid', 'error': f"验证集标签为空: {src_val_labels}"}

    local_dataset_address = str(output_root.resolve())
    logger.info(f"数据集结构构建完成: {local_dataset_address}")

    if heartbeat:
        heartbeat(stage='build', sub_step='prepare_upload_storage',
                  message='准备保存训练数据集结构', processed_files=0, total_files=0, force_log=True)

    persist_callback = None
    if heartbeat:
        def persist_callback(processed_files, total_files):
            heartbeat(stage='build', sub_step='upload_storage',
                      message=f"保存训练数据集结构: {processed_files}/{total_files}",
                      processed_files=processed_files, total_files=total_files, force_log=True)

    dataset_object_path = parse_storage_relative_path(dataset_storage_address)
    success_count, fail_count, failed_files = persist_directory_to_storage(
        datasets_root, f"{dataset_object_path}/datasets", progress_callback=persist_callback
    )
    if fail_count:
        return {'success': False, 'error_kind': 'io',
                'error': f"数据集保存到本地工作区失败: fail_count={fail_count}",
                'failed_files': failed_files[:10]}
    logger.info(f"数据集目录已保存到本地工作区: {dataset_storage_address}/datasets, count={success_count}")

    return {
        'success': True,
        'dataset_storage_address': dataset_storage_address,
        'local_dataset_address': local_dataset_address,
        'counts': {
            'train_images': train_images_count,
            'train_labels': train_labels_count,
            'valid_images': valid_images_count,
            'valid_labels': valid_labels_count,
        },
    }


@app.route('/algorithm/datasets/create', methods=['POST'])
def create_dataset():
    """构建数据集接口"""
    dataset_lock = None
    dataset_lock_acquired = False
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "data": {"message": "请求数据为空"}}), 400
        
        dataset_id = data.get('dataset_id')
        user_name = data.get('user_name')

        if dataset_id is None:
            return jsonify({"code": 1, "data": {"message": "缺少必需参数 dataset_id"}}), 400

        try:
            dataset_id = validate_dataset_id(dataset_id)
        except ValidationError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400

        logger.info(f"收到创建数据集请求: dataset_id={dataset_id}, user_name={user_name}")

        dataset_lock = get_dataset_lock(dataset_id)
        if not dataset_lock.acquire(blocking=False):
            logger.warning(f"数据集创建任务正在进行中，跳过重复请求: dataset_id={dataset_id}")
            return dataset_processing_response(dataset_id, "数据集创建")
        dataset_lock_acquired = True
        
        if user_name:
            if not db_manager.update_dataset_update_by(dataset_id, user_name):
                logger.warning(f"更新update_by字段失败: dataset_id={dataset_id}, user_name={user_name}")

        dataset_info = db_manager.get_dataset_split_info(dataset_id)
        if not dataset_info:
            return jsonify({"code": 1, "data": {"message": f"未找到dataset_id={dataset_id}的数据集信息"}}), 404

        try:
            result = _build_yolo_dataset_structure(dataset_id, dataset_info)
        except FileNotFoundError as e:
            logger.error(f"数据集创建失败，路径不存在: {str(e)}")
            return jsonify({"code": 1, "data": {"message": f"路径不存在: {str(e)}"}}), 400
        except Exception as e:
            logger.error(f"数据集创建失败: {str(e)}")
            return jsonify({"code": 1, "data": {"message": f"数据集创建失败: {str(e)}"}}), 500

        if not result['success']:
            status = 400 if result.get('error_kind') == 'invalid' else 500
            payload = {"message": result['error']}
            if result.get('failed_files'):
                payload['failed_files'] = result['failed_files']
            return jsonify({"code": 1, "data": payload}), status

        dataset_storage_address = result['dataset_storage_address']
        local_dataset_address = result['local_dataset_address']

        if not db_manager.update_dataset_address(dataset_id, dataset_storage_address):
            logger.warning("数据集创建成功但数据库更新dataset_address失败")
            return jsonify({
                "code": 1,
                "data": {
                    "message": "数据集创建成功，但数据库更新dataset_address失败",
                    "dataset_id": dataset_id,
                    "dataset_address": dataset_storage_address
                }
            }), 500

        return jsonify({
            "code": 0,
            "data": {
                "message": "数据集创建成功",
                "dataset_id": dataset_id,
                "dataset_address": dataset_storage_address,
                "storage_address": dataset_storage_address,
                "local_address": local_dataset_address,
                "user_name": user_name
            }
        })

    except Exception as e:
        return handle_api_exception(e, "创建数据集")
    finally:
        if dataset_lock_acquired and dataset_lock:
            dataset_lock.release()


@app.route('/algorithm/datasets/batch_modify_labels', methods=['POST'])
def batch_modify_dataset_labels():
    """固定处理 train/valid/json 目录中的标签，并在完成后刷新统计"""
    data = request.get_json(silent=True) or {}
    dataset_id = data.get('dataset_id')
    label_mapping = data.get('label_mapping')

    if dataset_id is None:
        return jsonify({"code": 1, "data": {"message": "缺少必需参数 dataset_id"}}), 400

    try:
        normalized_mapping = normalize_label_mapping(label_mapping)
    except ValueError as e:
        return jsonify({"code": 1, "data": {"message": str(e)}}), 400

    logger.info(
        f"收到批量修改标签请求: dataset_id={dataset_id}, "
        f"固定处理 split=['train', 'valid'], 自动刷新统计, label_mapping={normalized_mapping}"
    )

    dataset_info = db_manager.get_dataset_full_info(dataset_id)
    if not dataset_info:
        return jsonify({"code": 1, "data": {"message": f"未找到dataset_id={dataset_id}的数据集信息"}}), 404

    dataset_lock = get_dataset_lock(dataset_id)
    if not dataset_lock.acquire(blocking=False):
        logger.warning(f"数据集标签修改任务正在进行中，跳过重复请求: dataset_id={dataset_id}")
        return dataset_processing_response(dataset_id, "数据集标签修改")

    try:
        split_results = []
        total_modified_files = 0
        total_renamed_shapes = 0
        total_deleted_shapes = 0
        total_failed_files = 0

        for split_name in ('train', 'valid'):
            split_result = batch_modify_labels_in_split(dataset_id, split_name, normalized_mapping)
            split_results.append(split_result)
            total_modified_files += split_result['modified_files']
            total_renamed_shapes += split_result['renamed_shapes']
            total_deleted_shapes += split_result['deleted_shapes']
            total_failed_files += split_result['failed_count']

        stats_result = collect_dataset_stats_result(dataset_id)
        try:
            new_hash = compute_original_dataset_hash(dataset_id)
            if new_hash:
                db_manager.update_dataset_original_hash(
                    dataset_id,
                    new_hash,
                    schema_version=ORIGINAL_DATASET_HASH_SCHEMA_VERSION,
                )
                stats_result['original_dataset_hash'] = new_hash
                stats_result['original_dataset_hash_schema'] = ORIGINAL_DATASET_HASH_SCHEMA_VERSION
        except Exception as e:
            logger.warning(f"批量修改标签后写入original_dataset_hash失败: dataset_id={dataset_id}, error={str(e)}")

        return jsonify({
            "code": 0,
            "data": {
                "message": "批量修改标签完成",
                "dataset_id": dataset_id,
                "label_mapping": normalized_mapping,
                "summary": {
                    "total_modified_files": total_modified_files,
                    "total_renamed_shapes": total_renamed_shapes,
                    "total_deleted_shapes": total_deleted_shapes,
                    "total_failed_files": total_failed_files
                },
                "split_results": split_results,
                "stats": stats_result
            }
        })
    except FileNotFoundError as e:
        logger.error(f"批量修改标签路径不存在: {str(e)}")
        return jsonify({"code": 1, "data": {"message": f"路径不存在: {str(e)}"}}), 404
    except Exception as e:
        return handle_api_exception(e, "批量修改标签")
    finally:
        dataset_lock.release()


@app.route('/algorithm/datasets/stats', methods=['POST'])
def collect_dataset_stats():
    """修复 train/valid 对应关系并统计标签信息"""
    data = request.get_json(silent=True) or {}
    dataset_id = data.get('dataset_id')

    if dataset_id is None:
        return jsonify({"code": 1, "data": {"message": "缺少必需参数 dataset_id"}}), 400

    try:
        dataset_id = validate_dataset_id(dataset_id)
    except ValidationError as e:
        return jsonify({"code": 1, "data": {"message": str(e)}}), 400

    logger.info(f"收到数据集统计请求: dataset_id={dataset_id}")

    dataset_lock = get_dataset_lock(dataset_id)
    if not dataset_lock.acquire(blocking=False):
        logger.warning(f"数据集统计任务正在进行中,跳过重复请求: dataset_id={dataset_id}")
        return dataset_processing_response(dataset_id, "数据集统计")

    try:
        dataset_info = db_manager.get_dataset_full_info(dataset_id)
        if not dataset_info:
            return jsonify({"code": 1, "data": {"message": f"未找到dataset_id={dataset_id}的数据集信息"}}), 404

        # 哈希校验:若 original_dataset/{dataset_id} 目录内容未变,跳过统计流程
        force_stats = parse_bool_param(data.get('force'), default=False)
        previous_hash_info = db_manager.get_dataset_original_hash_info(dataset_id) or {}
        previous_hash = previous_hash_info.get('hash')
        previous_hash_schema = previous_hash_info.get('schema')
        try:
            current_hash = compute_original_dataset_hash(dataset_id)
        except Exception as e:
            logger.warning(f"计算original_dataset目录哈希失败,继续执行统计: dataset_id={dataset_id}, error={str(e)}")
            current_hash = None

        pose_fields_complete = True
        dataset_duty_type = str(dataset_info.get('duty_type') or '').strip().lower()
        dataset_draw_type = str(dataset_info.get('draw_type') or '').strip().lower()
        if dataset_duty_type == 'pose' or dataset_draw_type == 'point':
            db_kpt_labels = parse_json_array_field(dataset_info.get('kpt_labels'))
            try:
                db_kpt_num = int(dataset_info.get('kpt_num') or 0)
            except (TypeError, ValueError):
                db_kpt_num = 0
            pose_fields_complete = bool(db_kpt_labels and db_kpt_num == len(db_kpt_labels))

        if (
            not force_stats
            and current_hash
            and previous_hash
            and current_hash == previous_hash
            and (not previous_hash_schema or previous_hash_schema == ORIGINAL_DATASET_HASH_SCHEMA_VERSION)
            and pose_fields_complete
        ):
            logger.info(
                f"original_dataset目录哈希未变,跳过数据集统计: dataset_id={dataset_id}, hash={current_hash}"
            )
            return jsonify({
                "code": 0,
                "data": {
                    "message": "原始数据集目录未变更,跳过统计",
                    "dataset_id": dataset_id,
                    "skipped": True,
                    "original_dataset_hash": current_hash,
                    "original_dataset_hash_schema": ORIGINAL_DATASET_HASH_SCHEMA_VERSION
                }
            })

        stats_result = collect_dataset_stats_result(dataset_id)

        # stats 阶段会修改目录(补负样本 json、删除孤立 json),统计完成后重新计算并写库
        try:
            new_hash = compute_original_dataset_hash(dataset_id)
            if new_hash:
                db_manager.update_dataset_original_hash(
                    dataset_id,
                    new_hash,
                    schema_version=ORIGINAL_DATASET_HASH_SCHEMA_VERSION,
                )
                stats_result['original_dataset_hash'] = new_hash
                stats_result['original_dataset_hash_schema'] = ORIGINAL_DATASET_HASH_SCHEMA_VERSION
        except Exception as e:
            logger.warning(f"统计完成后写入original_dataset_hash失败: dataset_id={dataset_id}, error={str(e)}")

        return jsonify({
            "code": 0,
            "data": {
                "message": "数据集统计完成",
                "skipped": False,
                **stats_result
            }
        })
    except FileNotFoundError as e:
        logger.error(f"数据集统计路径不存在: {str(e)}")
        return jsonify({"code": 1, "data": {"message": f"路径不存在: {str(e)}"}}), 404
    except Exception as e:
        return handle_api_exception(e, "数据集统计")
    finally:
        dataset_lock.release()


@app.route('/algorithm/train/build_yolo_dataset', methods=['POST'])
def build_yolo_dataset():
    """构建YOLO训练数据集接口"""
    try:
        data = request.get_json(force=True, silent=True)
        
        logger.info(f"=== /algorithm/train/build_yolo_dataset 接口入参 ===")
        logger.info(f"原始请求数据: {data}")
        logger.info(f"请求数据类型: {type(data)}")
        
        if isinstance(data, str):
            try:
                data = json.loads(data)
                logger.info(f"字符串解析后的数据: {data}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                return jsonify({"code": 1, "data": {"message": f"请求数据格式错误，无法解析JSON: {str(e)}"}}), 400
        
        if data and isinstance(data, dict):
            for key, value in data.items():
                logger.info(f"  参数 {key}: {value} (类型: {type(value).__name__})")
        logger.info(f"=== 入参打印结束 ===")
        
        if not data:
            return jsonify({"code": 1, "data": {"message": "请求数据为空"}}), 400
        
        dataset_id = data.get('dataset_id')
        if dataset_id is None:
            return jsonify({"code": 1, "data": {"message": "缺少必需参数 dataset_id"}}), 400

        try:
            dataset_id = validate_dataset_id(dataset_id)
        except ValidationError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400

        # 原子地"检查并占位"：在启动线程前持锁写入 processing 状态，
        # 避免两个快速请求同时通过检查后并发删除/重建同一 dataset。
        now = datetime.now()
        with dataset_build_tasks_lock:
            existing_task = dataset_build_tasks.get(str(dataset_id))
            if existing_task and existing_task.get('status') == 'processing':
                logger.warning(f"数据集构建任务已在进行中，跳过重复请求: dataset_id={dataset_id}")
                return jsonify({
                    "code": 0,
                    "data": {
                        "message": "数据集构建任务已在进行中",
                        "dataset_id": dataset_id,
                        "status": "processing"
                    }
                })
            dataset_build_tasks[str(dataset_id)] = {
                'status': 'processing',
                'stage': 'initializing',
                'current_stage_index': 0,
                'total_stages': 3,
                'error': None,
                'start_time': now,
                'end_time': None,
                'stage_started_at': now,
                'last_heartbeat_at': now,
                'last_progress_at': None,
                'message': '数据集构建任务初始化',
                'sub_step': 'initializing',
                'processed_files': None,
                'total_files': None,
                'watchdog_warnings': 0
            }

        logger.info(f"收到构建YOLO数据集请求: dataset_id={dataset_id}，启动异步处理")

        def async_build_pipeline(dataset_id):
            start_time = time.time()
            dataset_id_str = str(dataset_id)
            callback_success = False
            callback_message = None

            watchdog_stop_event = threading.Event()
            start_dataset_build_watchdog(dataset_id, watchdog_stop_event)
            update_dataset_build_heartbeat(
                dataset_id,
                stage='initializing',
                sub_step='initializing',
                message='数据集构建任务初始化',
                force_log=True
            )

            try:
                if not db_manager.update_dataset_process_status(dataset_id, 0):
                    error_msg = "开始构建时更新train_dataset.process_status为0失败"
                    logger.error(f"[异步] {error_msg}: dataset_id={dataset_id}")
                    update_dataset_build_status(
                        dataset_id, status='failed', error=error_msg, end_time=datetime.now()
                    )
                    callback_message = error_msg
                    return

                stages = [
                    {
                        'name': 'convert',
                        'description': '标注格式转换',
                        'handler': _process_json2txt
                    },
                    {
                        'name': 'build',
                        'description': '构建训练数据集结构',
                        'handler': _process_create_dataset
                    },
                    {
                        'name': 'configure',
                        'description': '生成训练配置文件',
                        'handler': _process_generate_yaml
                    }
                ]

                total_stages = len(stages)

                for idx, stage in enumerate(stages, 1):
                    stage_name = stage['name']
                    stage_start = time.time()
                    logger.info(f"[异步][{idx}/{total_stages}] {stage['description']}: dataset_id={dataset_id}")

                    with dataset_build_tasks_lock:
                        dataset_build_tasks[dataset_id_str]['stage'] = stage_name
                        dataset_build_tasks[dataset_id_str]['current_stage_index'] = idx
                        dataset_build_tasks[dataset_id_str]['stage_started_at'] = datetime.now()
                        dataset_build_tasks[dataset_id_str]['processed_files'] = None
                        dataset_build_tasks[dataset_id_str]['total_files'] = None
                    update_dataset_build_heartbeat(
                        dataset_id,
                        stage=stage_name,
                        sub_step='stage_start',
                        message=f"开始{stage['description']}",
                        force_log=True
                    )

                    try:
                        result = stage['handler'](dataset_id)
                        stage_duration = round(time.time() - stage_start, 2)

                        if not result.get('success'):
                            error_msg = result.get('error', '未知错误')
                            logger.error(f"[异步][{idx}/{total_stages}] 失败: {error_msg}")
                            update_dataset_build_status(
                                dataset_id, status='failed', error=error_msg, end_time=datetime.now()
                            )
                            db_manager.update_dataset_process_status(dataset_id, 2, error_msg)
                            callback_message = error_msg
                            return

                        update_dataset_build_heartbeat(
                            dataset_id,
                            stage=stage_name,
                            sub_step='stage_done',
                            message=f"{stage['description']}完成，耗时 {stage_duration}s",
                            force_log=True
                        )
                        logger.info(f"[异步][{idx}/{total_stages}] 完成 ({stage_duration}s)")

                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"[异步][{idx}/{total_stages}] 异常: {error_msg}")
                        logger.error(traceback.format_exc())
                        update_dataset_build_status(
                            dataset_id, status='failed', error=error_msg, end_time=datetime.now()
                        )
                        db_manager.update_dataset_process_status(dataset_id, 2, error_msg)
                        callback_message = error_msg
                        return

                if not db_manager.update_dataset_process_status(dataset_id, 1):
                    error_msg = "全部处理完成，但更新train_dataset.process_status为1失败"
                    logger.error(f"[异步] {error_msg}: dataset_id={dataset_id}")
                    update_dataset_build_status(
                        dataset_id,
                        status='failed',
                        error=error_msg,
                        end_time=datetime.now(),
                        stage='done',
                        current_stage_index=total_stages,
                    )
                    db_manager.update_dataset_process_status(dataset_id, 2, error_msg)
                    callback_message = error_msg
                    return

                total_duration = round(time.time() - start_time, 2)
                logger.info(f"[异步] YOLO数据集构建流程完成: dataset_id={dataset_id}, 耗时: {total_duration}s")

                update_dataset_build_status(
                    dataset_id, status='completed', stage='done', end_time=datetime.now()
                )
                callback_success = True
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[异步] YOLO数据集构建流程异常: dataset_id={dataset_id}, error={error_msg}")
                logger.error(traceback.format_exc())
                update_dataset_build_status(
                    dataset_id, status='failed', error=error_msg, end_time=datetime.now()
                )
                callback_message = error_msg
            finally:
                watchdog_stop_event.set()
                send_process_callback(dataset_id, callback_success, callback_message)

        thread = threading.Thread(target=async_build_pipeline, args=(dataset_id,), daemon=True)
        try:
            thread.start()
        except Exception:
            # 线程启动失败时释放已占位的 processing 槽，避免数据集卡在 processing
            with dataset_build_tasks_lock:
                current = dataset_build_tasks.get(str(dataset_id))
                if current and current.get('status') == 'processing':
                    dataset_build_tasks.pop(str(dataset_id), None)
            raise

        return jsonify({
            "code": 0,
            "data": {
                "message": "开始构建YOLO数据集",
                "dataset_id": dataset_id,
                "status": "processing"
            }
        })
        
    except Exception as e:
        return handle_api_exception(e, "构建YOLO数据集")


def _process_divided_original_detail(dataset_id):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    dataset_info = db_manager.get_divided_data_address(dataset_id)
    if not dataset_info:
        return {'success': False, 'error': f"未找到dataset_id={dataset_id}的数据集信息"}
    
    original_train_data_address = dataset_info.get('original_train_data_address')
    original_val_data_address = dataset_info.get('original_val_data_address')
    
    if not original_train_data_address:
        return {'success': False, 'error': "original_train_data_address字段为空"}
    if not original_val_data_address:
        return {'success': False, 'error': "original_val_data_address字段为空"}
    
    local_train_address = ensure_local_path(original_train_data_address)
    local_val_address = ensure_local_path(original_val_data_address)
    
    def analyze_dataset(address):
        analyzer = DatasetDetailAnalyzer(address)
        return analyzer.analyze_dataset()
    
    def find_sample_images_for_labels(local_address, base_path, tag_num_dict):
        """为每个标签找一张包含该类型的示例图片"""
        label_images = {}
        json_dir = resolve_annotation_dir(local_address)
        images_dir = Path(local_address) / 'images'
        
        if not json_dir.exists() or not images_dir.exists():
            logger.warning(f"json/jsons或images目录不存在: {local_address}")
            return label_images
        
        json_files = [json_file for json_file in sorted(json_dir.glob('*.json')) if json_file.is_file()]
        if not json_files:
            logger.warning(f"未找到JSON标注文件: {json_dir}")
            return label_images
        
        label_to_images = {label: [] for label in tag_num_dict.keys()}
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                labels_in_file = set()
                shapes = data.get('shapes', [])
                for shape in shapes:
                    label = shape.get('label')
                    if label:
                        labels_in_file.add(label)
                
                image_name = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                    potential_image = images_dir / (json_file.stem + ext)
                    if potential_image.exists():
                        image_name = potential_image.name
                        break
                
                if image_name:
                    for label_name in labels_in_file:
                        if label_name in label_to_images:
                            label_to_images[label_name].append(image_name)
            except Exception as e:
                logger.debug(f"处理JSON文件失败 {json_file}: {e}")
                continue
        
        for label_name, images in label_to_images.items():
            if images:
                selected_image = random.choice(images)
                object_key = f"{base_path}/images/{selected_image}"
                label_images[label_name] = object_key
                logger.debug(f"标签 {label_name} 选择图片: {object_key}")
        
        return label_images
    
    train_result = None
    val_result = None
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        train_future = executor.submit(analyze_dataset, local_train_address)
        val_future = executor.submit(analyze_dataset, local_val_address)
        
        train_result = train_future.result()
        val_result = val_future.result()
    
    if not train_result or not val_result:
        return {'success': False, 'error': "数据集分析失败"}
    
    train_tag_sum = train_result.get('tag_sum', 0)
    train_tag_num_dict = train_result.get('tag_num', {})
    train_tag_percentage_dict = train_result.get('tag_percentage', {})
    train_tag_num = json.dumps(train_tag_num_dict, ensure_ascii=False)
    train_tag_percentage = json.dumps(train_tag_percentage_dict, ensure_ascii=False)
    
    val_tag_sum = val_result.get('tag_sum', 0)
    val_tag_num_dict = val_result.get('tag_num', {})
    val_tag_percentage_dict = val_result.get('tag_percentage', {})
    val_tag_num = json.dumps(val_tag_num_dict, ensure_ascii=False)
    val_tag_percentage = json.dumps(val_tag_percentage_dict, ensure_ascii=False)
    
    update_success = db_manager.update_divided_dataset_detail(
        dataset_id=dataset_id,
        train_tag_sum=train_tag_sum,
        val_tag_sum=val_tag_sum,
        train_tag_num=train_tag_num,
        val_tag_num=val_tag_num,
        train_tag_percentage=train_tag_percentage,
        val_tag_percentage=val_tag_percentage
    )
    
    if not update_success:
        return {'success': False, 'error': "数据库更新失败"}
    
    try:
        train_base_path = original_train_data_address
        val_base_path = original_val_data_address
        
        train_label_images = find_sample_images_for_labels(local_train_address, train_base_path, train_tag_num_dict)
        val_label_images = find_sample_images_for_labels(local_val_address, val_base_path, val_tag_num_dict)
        
        label_data_list = []
        
        for label_name, tag_count in train_tag_num_dict.items():
            percentage_str = train_tag_percentage_dict.get(label_name, '0%')
            proportion = float(percentage_str.replace('%', '')) if percentage_str else 0.0
            
            label_data_list.append({
                'label_name': label_name,
                'tag_num': tag_count,
                'proportion': proportion,
                'data_type': 1,
                'object_key': train_label_images.get(label_name)
            })
        
        for label_name, tag_count in val_tag_num_dict.items():
            percentage_str = val_tag_percentage_dict.get(label_name, '0%')
            proportion = float(percentage_str.replace('%', '')) if percentage_str else 0.0
            
            label_data_list.append({
                'label_name': label_name,
                'tag_num': tag_count,
                'proportion': proportion,
                'data_type': 2,
                'object_key': val_label_images.get(label_name)
            })
        
        if label_data_list:
            success_count, fail_count = db_manager.insert_dataset_labels(dataset_id, label_data_list)
            logger.info(f"写入train_dataset_label表完成: dataset_id={dataset_id}, 成功={success_count}, 失败={fail_count}")
        
    except Exception as e:
        logger.error(f"写入train_dataset_label表失败: {str(e)}")
    
    return {
        'success': True,
        'output': {
            'train_set': {
                'samples': train_result.get('sample_num', 0),
                'annotations': train_tag_sum,
                'labels': train_tag_num_dict
            },
            'valid_set': {
                'samples': val_result.get('sample_num', 0),
                'annotations': val_tag_sum,
                'labels': val_tag_num_dict
            }
        }
    }


def _process_json2txt(dataset_id):
    from concurrent.futures import ThreadPoolExecutor

    dataset_info = db_manager.get_dataset_json2txt_info(dataset_id)
    if not dataset_info:
        return {'success': False, 'error': f"未找到dataset_id={dataset_id}的数据集信息"}

    duty_type = str(dataset_info.get('duty_type') or '').strip().lower()
    draw_type = str(dataset_info.get('draw_type') or '').strip().lower()
    original_train_data_address = dataset_info.get('original_train_data_address')
    original_val_data_address = dataset_info.get('original_val_data_address')
    labels_str = dataset_info.get('labels')

    if not duty_type or not labels_str:
        return {'success': False, 'error': "缺少duty_type或labels"}

    if not original_train_data_address and not original_val_data_address:
        return {'success': False, 'error': "original_train_data_address和original_val_data_address都为空"}

    # point 数据集判定与 stats/yaml 阶段保持一致：duty_type=='pose' 或 draw_type=='point'
    # 都按 pose 处理。否则 json2txt 会把矩形当检测框、丢弃关键点，产出 5 列检测标签，
    # 而 yaml 仍写出 kpt_shape，训练前校验必然失败。
    is_pose = duty_type == 'pose' or draw_type == 'point'
    effective_duty_type = 'pose' if is_pose else duty_type

    labels_list = parse_labels(labels_str)

    class_mapping = {label: idx for idx, label in enumerate(labels_list)}
    logger.info(f"标签映射字典: {class_mapping}")

    kpt_labels = None
    kpt_num = None
    if is_pose:
        try:
            kpt_labels, kpt_num = resolve_pose_keypoint_config(
                dataset_id,
                dataset_info,
                split_addresses=[original_train_data_address, original_val_data_address],
                persist=True,
            )
        except ValueError as e:
            return {'success': False, 'error': str(e)}
        logger.info(f"pose 任务关键点标签: {kpt_labels}, kpt_num: {kpt_num}")

    def convert_dataset(address, storage_path):
        split_name = '训练集' if address == original_train_data_address else '验证集'
        local_address = resolve_dataset_dir_for_conversion(address, split_name)

        labels_dir = Path(local_address) / 'labels'
        if labels_dir.exists():
            logger.info(f"删除已存在的labels目录: {labels_dir}")
            safe_rmtree(labels_dir)

        converter = UnifiedJsonConverter(
            duty_type=effective_duty_type,
            original_train_data_address=local_address,
            predefined_labels=labels_list,
            kpt_labels=kpt_labels,
            kpt_num=kpt_num,
        )
        result = converter.convert()

        if PathHandler.is_legacy_remote_uri(storage_path):
            _copy_labels_to_storage(converter.labels_dir, storage_path)

        return result

    train_result = None
    val_result = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if original_train_data_address:
            futures['train'] = executor.submit(convert_dataset, original_train_data_address, original_train_data_address)
        if original_val_data_address:
            futures['val'] = executor.submit(convert_dataset, original_val_data_address, original_val_data_address)

        if 'train' in futures:
            train_result = futures['train'].result()
        if 'val' in futures:
            val_result = futures['val'].result()

    return {
        'success': True,
        'output': {
            'format': 'YOLO TXT',
            'task_type': effective_duty_type,
            'class_mapping': class_mapping,
            'kpt_num': kpt_num,
            'kpt_labels': kpt_labels,
            'train_set': {
                'converted': train_result.get('success', 0) if train_result else 0,
                'failed': train_result.get('failed', 0) if train_result else 0
            },
            'valid_set': {
                'converted': val_result.get('success', 0) if val_result else 0,
                'failed': val_result.get('failed', 0) if val_result else 0
            }
        }
    }


def _collect_keypoint_labels_from_split_dirs(addresses):
    """扫描 split 目录中的 JSON，收集出现过的 point 标签（按出现顺序去重）"""
    seen = []
    for address in addresses:
        if not address:
            continue
        try:
            local_dir = Path(ensure_local_path(address))
        except Exception as e:
            logger.warning(f"解析关键点标签时无法定位本地路径: {address}, error={e}")
            continue
        json_dir = None
        for sub in ('json', 'jsons'):
            candidate = local_dir / sub
            if candidate.exists():
                json_dir = candidate
                break
        if json_dir is None:
            continue
        for json_file in sorted(json_dir.glob('*.json')):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            assignment = assign_points_to_rectangles(data.get('shapes') or [])
            for kp_label in assignment.keypoint_labels:
                if kp_label and kp_label not in seen:
                    seen.append(kp_label)
    return seen


def _process_create_dataset(dataset_id):
    def heartbeat(**kwargs):
        update_dataset_build_heartbeat(dataset_id, **kwargs)

    dataset_info = db_manager.get_dataset_split_info(dataset_id)
    if not dataset_info:
        return {'success': False, 'error': f"未找到dataset_id={dataset_id}的数据集信息"}

    try:
        result = _build_yolo_dataset_structure(dataset_id, dataset_info, heartbeat=heartbeat)
    except FileNotFoundError as e:
        return {'success': False, 'error': f"路径不存在: {str(e)}"}

    if not result['success']:
        return {'success': False, 'error': result['error']}

    dataset_storage_address = result['dataset_storage_address']
    counts = result['counts']

    heartbeat(stage='build', sub_step='update_db', message='更新训练数据集地址到数据库', force_log=True)
    if not db_manager.update_dataset_address(dataset_id, dataset_storage_address):
        return {'success': False, 'error': "数据库更新dataset_address失败"}
    heartbeat(stage='build', sub_step='build_done', message='训练数据集结构构建完成',
              processed_files=0, total_files=0, force_log=True)

    return {
        'success': True,
        'output': {
            'dataset_path': dataset_storage_address,
            'local_dataset_path': result['local_dataset_address'],
            'structure': {
                'train': {'images': counts['train_images'], 'labels': counts['train_labels']},
                'valid': {'images': counts['valid_images'], 'labels': counts['valid_labels']},
            },
            'total_files': sum(counts.values()),
        }
    }


def _process_generate_yaml(dataset_id):
    dataset_info = db_manager.get_dataset_full_info(dataset_id)
    if not dataset_info:
        return {'success': False, 'error': f"未找到dataset_id={dataset_id}的数据集信息"}
    
    dataset_address = dataset_info.get('dataset_address')
    labels = dataset_info.get('labels')
    label_num = dataset_info.get('label_num')
    draw_type = dataset_info.get('draw_type')
    duty_type = str(dataset_info.get('duty_type') or '').strip().lower()
    kpt_labels = None
    kpt_num = dataset_info.get('kpt_num')

    if not dataset_address or not labels or not label_num:
        return {'success': False, 'error': "缺少dataset_address、labels或label_num"}

    duty_type_for_yaml = 'pose' if duty_type == 'pose' or draw_type == 'point' else None
    if duty_type_for_yaml == 'pose':
        try:
            kpt_labels, kpt_num = resolve_pose_keypoint_config(dataset_id, dataset_info, persist=True)
        except ValueError as e:
            return {'success': False, 'error': str(e)}

    update_dataset_build_heartbeat(
        dataset_id,
        stage='configure',
        sub_step='resolve_dataset_path',
        message='解析训练数据集本地路径',
        processed_files=0,
        total_files=0,
        force_log=True
    )
    local_dataset_address = ensure_local_path(dataset_address)
    Path(local_dataset_address).mkdir(parents=True, exist_ok=True)

    from utils import DatasetYamlGenerator

    update_dataset_build_heartbeat(
        dataset_id,
        stage='configure',
        sub_step='generate_yaml',
        message='生成 YOLO dataset.yaml',
        processed_files=0,
        total_files=0,
        force_log=True
    )
    generator = DatasetYamlGenerator()
    yaml_output_path = Path(local_dataset_address) / "dataset.yaml"
    result = generator.generate_silent(
        dataset_address=local_dataset_address,
        labels=labels,
        label_num=label_num,
        output_path=str(yaml_output_path),
        duty_type=duty_type_for_yaml,
        kpt_num=int(kpt_num) if kpt_num else None
    )
    
    if not result['success']:
        return {'success': False, 'error': result.get('error', '未知错误')}
    
    local_yaml_path = result['yaml_path']

    update_dataset_build_heartbeat(
        dataset_id,
        stage='configure',
        sub_step='save_yaml_address',
        message='保存 YAML 地址信息',
        processed_files=0,
        total_files=0,
        force_log=True
    )
    yaml_address = normalize_address_for_db(local_yaml_path)
    if PathHandler.is_legacy_remote_uri(yaml_address):
        upload_file_to_storage(
            Path(local_yaml_path),
            parse_storage_relative_path(yaml_address),
            content_type='application/x-yaml'
        )
        logger.info(f"YAML文件已保存到本地工作区: {yaml_address}")

    update_dataset_build_heartbeat(
        dataset_id,
        stage='configure',
        sub_step='update_db',
        message='更新 YAML 地址到数据库',
        processed_files=0,
        total_files=0,
        force_log=True
    )
    db_manager.update_dataset_yaml_address(dataset_id, yaml_address)
    update_dataset_build_heartbeat(
        dataset_id,
        stage='configure',
        sub_step='configure_done',
        message='训练配置文件生成完成',
        processed_files=0,
        total_files=0,
        force_log=True
    )

    try:
        labels_list = json.loads(labels) if isinstance(labels, str) else labels
    except Exception as e:
        logger.debug(f"解析labels失败，使用空列表: {e}")
        labels_list = []
    
    return {
        'success': True,
        'output': {
            'config_file': yaml_address,
            'local_config_file': local_yaml_path,
            'config_type': 'YOLO dataset.yaml',
            'classes': {
                'count': label_num,
                'names': labels_list
            },
            'kpt_num': int(kpt_num) if duty_type_for_yaml == 'pose' and kpt_num else None,
            'kpt_labels': kpt_labels
        }
    }


@app.route('/algorithm/datasets/build_status', methods=['GET'])
def get_dataset_build_status():
    try:
        dataset_id = request.args.get('dataset_id')
        
        if not dataset_id:
            return jsonify({
                "code": 1,
                "data": {"message": "缺少必需参数 dataset_id"}
            }), 400
        
        logger.info(f"查询数据集构建状态: dataset_id={dataset_id}")
        
        with dataset_build_tasks_lock:
            task_info = dataset_build_tasks.get(str(dataset_id))
            task_info = dict(task_info) if task_info else None

        if not task_info:
            return jsonify({
                "code": 0,
                "data": {
                    "dataset_id": dataset_id,
                    "status": "not_found",
                    "message": "未找到该数据集的构建任务，可能尚未开始或已过期"
                }
            })
        
        # 格式化时间
        start_time_str = format_dataset_build_time(task_info.get('start_time'))
        end_time_str = format_dataset_build_time(task_info.get('end_time'))
        stage_started_at_str = format_dataset_build_time(task_info.get('stage_started_at'))
        last_heartbeat_at_str = format_dataset_build_time(task_info.get('last_heartbeat_at'))
        last_progress_at_str = format_dataset_build_time(task_info.get('last_progress_at'))

        total_stages = task_info.get('total_stages') or 0
        current_stage_index = task_info.get('current_stage_index')
        progress = None
        if total_stages and current_stage_index is not None:
            progress = round(current_stage_index / total_stages * 100, 1)

        return jsonify({
            "code": 0,
            "data": {
                "dataset_id": dataset_id,
                "status": task_info['status'],
                "stage": task_info.get('stage'),
                "current_stage_index": current_stage_index,
                "total_stages": total_stages,
                "progress": progress,
                "sub_step": task_info.get('sub_step'),
                "message": task_info.get('message'),
                "processed_files": task_info.get('processed_files'),
                "total_files": task_info.get('total_files'),
                "watchdog_warnings": task_info.get('watchdog_warnings', 0),
                "stage_started_at": stage_started_at_str,
                "last_heartbeat_at": last_heartbeat_at_str,
                "last_progress_at": last_progress_at_str,
                "error": task_info.get('error'),
                "start_time": start_time_str,
                "end_time": end_time_str
            }
        })
        
    except Exception as e:
        logger.error(f"查询数据集构建状态失败: {str(e)}")
        return jsonify({
            "code": 1,
            "data": {"message": f"服务器内部错误: {str(e)}"}
        }), 500


@app.route('/algorithm/train/log', methods=['POST'])
def get_train_log():
    """获取训练任务的实时日志"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "data": {"message": "请求数据为空"}}), 400
        
        train_task_id = data.get('train_task_id')
        if train_task_id is None:
            return jsonify({"code": 1, "data": {"message": "缺少必需参数 train_task_id"}}), 400

        try:
            train_task_id = validate_positive_id(train_task_id, "train_task_id")
        except ValidationError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400
        
        lines = data.get('lines', 100)
        if not isinstance(lines, int) or lines <= 0:
            lines = 100
        
        logger.info(f"收到获取训练日志请求: train_task_id={train_task_id}, lines={lines}")
        
        local_log_path = None
        
        train_log_path = build_storage_local_path("train_log", require_exists=False)
        default_log_path = train_log_path / str(train_task_id) / 'training.log'
        if default_log_path.exists():
            local_log_path = str(default_log_path)
            logger.info(f"使用本地日志路径: {local_log_path}")
        else:
            train_results_path = build_storage_local_path("train_results", require_exists=False)
            fallback_log_path = train_results_path / str(train_task_id) / 'training.log'
            if fallback_log_path.exists():
                local_log_path = str(fallback_log_path)
                logger.info(f"使用降级日志路径: {local_log_path}")
            else:
                task_dir = train_log_path / str(train_task_id)
                if task_dir.exists():
                    log_files = list(task_dir.glob('*.log'))
                    if log_files:
                        local_log_path = str(log_files[0])
                        logger.info(f"找到日志文件: {local_log_path}")
        
        if not local_log_path or not Path(local_log_path).exists():
            return jsonify({
                "code": 1,
                "data": {
                    "message": f"未找到train_task_id={train_task_id}的日志文件",
                    "train_task_id": train_task_id
                }
            }), 404
        
        try:
            with open(local_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
                total_lines = len(all_lines)
                
                if lines >= total_lines:
                    log_content = ''.join(all_lines)
                else:
                    log_content = ''.join(all_lines[-lines:])
                
                return jsonify({
                    "code": 0,
                    "data": {
                        "message": "获取日志成功",
                        "train_task_id": train_task_id,
                        "log_content": log_content,
                        "total_lines": total_lines,
                        "returned_lines": min(lines, total_lines),
                        "log_path": local_log_path
                    }
                })
                
        except Exception as e:
            logger.error(f"读取日志文件失败: {str(e)}")
            return jsonify({
                "code": 1,
                "data": {
                    "message": f"读取日志文件失败: {str(e)}",
                    "train_task_id": train_task_id
                }
            }), 500
            
    except Exception as e:
        logger.error(f"获取训练日志失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "code": 1,
            "data": {"message": f"服务器内部错误: {str(e)}"}
        }), 500


class PredictRedisManager:
    
    KEY_PREFIX = "predict:status"
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        try:
            import redis
            self.client = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                db=11,
                password=self.config['password'],
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"PredictRedisManager连接成功: {self.config['host']}:{self.config['port']}, db=11")
        except ImportError:
            logger.error("redis模块未安装")
            self.client = None
        except Exception as e:
            logger.error(f"PredictRedisManager连接失败: {str(e)}")
            self.client = None
    
    def update_status(self, model_id, prediction_id, status):
        if not self.client:
            logger.warning("PredictRedisManager未连接，跳过状态更新")
            return False
        
        try:
            key = f"{self.KEY_PREFIX}:{model_id}:{prediction_id}"
            self.client.set(key, status)
            logger.info(f"Redis状态更新: {key} = {status}")
            return True
        except Exception as e:
            logger.error(f"Redis状态更新失败: {str(e)}")
            return False


class AutoLabelRedisManager:
    """自动标注任务 Redis 进行中标记。"""

    KEY_PREFIX = "auto_label"

    def __init__(self, config):
        self.config = config
        self.client = None
        self._connect()

    def _connect(self):
        try:
            import redis
            self.client = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                db=11,
                password=self.config['password'],
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"AutoLabelRedisManager连接成功: {self.config['host']}:{self.config['port']}, db=11")
        except ImportError:
            logger.error("redis模块未安装")
            self.client = None
        except Exception as e:
            logger.error(f"AutoLabelRedisManager连接失败: {str(e)}")
            self.client = None

    def build_key(self, label_task_id):
        return f"{self.KEY_PREFIX}:{label_task_id}"

    def start_task(self, label_task_id, model_id, conf, classes):
        """创建进行中 key。存在则说明该标注任务已有自动标注在跑。"""
        if not self.client:
            logger.warning("AutoLabelRedisManager未连接，无法创建自动标注状态key")
            return False, None, "Redis不可用，无法创建自动标注状态key"

        key = self.build_key(label_task_id)
        value = json.dumps({
            'label_task_id': str(label_task_id),
            'model_id': str(model_id),
            'conf': str(conf),
            'classes': classes,
            'status': 'processing',
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, ensure_ascii=False)

        try:
            created = self.client.set(
                key,
                value,
                nx=True,
                ex=AUTO_LABEL_REDIS_LOCK_TTL_SECONDS
            )
            if created:
                logger.info(f"自动标注状态key创建成功: {key}")
                return True, key, None

            existing_value = self.client.get(key)
            logger.warning(f"自动标注任务已在进行中: key={key}, value={existing_value}")
            return False, key, existing_value or "自动标注任务已在进行中"
        except Exception as e:
            logger.error(f"创建自动标注状态key失败: key={key}, error={str(e)}")
            return False, None, str(e)

    def finish_task(self, label_task_id):
        """任务完成后删除进行中 key。"""
        if not self.client:
            logger.warning("AutoLabelRedisManager未连接，跳过删除自动标注状态key")
            return False

        key = self.build_key(label_task_id)
        try:
            self.client.delete(key)
            logger.info(f"自动标注状态key已删除: {key}")
            return True
        except Exception as e:
            logger.error(f"删除自动标注状态key失败: key={key}, error={str(e)}")
            return False


predict_redis_manager = PredictRedisManager(redis_config)
auto_label_redis_manager = AutoLabelRedisManager(redis_config)

dataset_build_tasks = {}
dataset_build_tasks_lock = threading.Lock()


def update_dataset_build_status(dataset_id, **fields):
    """持锁更新 dataset_build_tasks 中某任务的状态字段。

    async_build_pipeline 过去对该共享字典的失败/完成态写入不加锁，而
    heartbeat / watchdog / 状态查询接口都通过 dataset_build_tasks_lock 读取，
    读写锁不一致存在竞态（脏读、丢失更新）。所有非 heartbeat 的状态变更统一
    走这里，保证与读取端使用同一把锁。
    """
    dataset_id_str = str(dataset_id)
    with dataset_build_tasks_lock:
        task_info = dataset_build_tasks.get(dataset_id_str)
        if not task_info:
            return False
        task_info.update(fields)
    return True


def update_dataset_build_heartbeat(
    dataset_id,
    stage=None,
    sub_step=None,
    message=None,
    processed_files=None,
    total_files=None,
    force_log=False,
):
    dataset_id_str = str(dataset_id)
    now = datetime.now()
    with dataset_build_tasks_lock:
        task_info = dataset_build_tasks.get(dataset_id_str)
        if not task_info:
            return False

        previous_stage = task_info.get('stage')
        if stage:
            task_info['stage'] = stage
            if previous_stage != stage:
                task_info['stage_started_at'] = now
        if sub_step is not None:
            task_info['sub_step'] = sub_step
        if message is not None:
            task_info['message'] = message
        if processed_files is not None:
            task_info['processed_files'] = processed_files
            task_info['last_progress_at'] = now
        if total_files is not None:
            task_info['total_files'] = total_files
        task_info['last_heartbeat_at'] = now

        log_stage = task_info.get('stage')
        log_sub_step = task_info.get('sub_step')
        log_message = task_info.get('message')
        log_processed = task_info.get('processed_files')
        log_total = task_info.get('total_files')

    if force_log:
        progress_text = ''
        if log_processed is not None and log_total is not None:
            progress_text = f", progress={log_processed}/{log_total}"
        logger.info(
            f"[异步][监测] dataset_id={dataset_id}, stage={log_stage}, "
            f"sub_step={log_sub_step}{progress_text}, message={log_message}"
        )
    return True


def start_dataset_build_watchdog(dataset_id, stop_event):
    dataset_id_str = str(dataset_id)

    def watchdog_loop():
        while not stop_event.wait(DATASET_BUILD_WATCHDOG_INTERVAL):
            with dataset_build_tasks_lock:
                task_info = dataset_build_tasks.get(dataset_id_str)
                if not task_info or task_info.get('status') != 'processing':
                    return
                stage = task_info.get('stage')
                last_heartbeat_at = task_info.get('last_heartbeat_at') or task_info.get('start_time')
                sub_step = task_info.get('sub_step')
                message = task_info.get('message')
                processed_files = task_info.get('processed_files')
                total_files = task_info.get('total_files')
                watchdog_warnings = task_info.get('watchdog_warnings', 0)

            if not last_heartbeat_at:
                continue

            idle_seconds = (datetime.now() - last_heartbeat_at).total_seconds()
            idle_threshold = DATASET_BUILD_WATCHDOG_IDLE_SECONDS.get(stage, 600)
            if idle_seconds < idle_threshold:
                continue

            with dataset_build_tasks_lock:
                task_info = dataset_build_tasks.get(dataset_id_str)
                if not task_info or task_info.get('status') != 'processing':
                    return
                task_info['watchdog_warnings'] = task_info.get('watchdog_warnings', 0) + 1
                watchdog_warnings = task_info['watchdog_warnings']

            logger.warning(
                f"[异步][监测] 数据集构建长时间无心跳: dataset_id={dataset_id}, "
                f"stage={stage}, sub_step={sub_step}, idle_seconds={int(idle_seconds)}, "
                f"warnings={watchdog_warnings}, progress={processed_files}/{total_files}, message={message}"
            )

    watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
    watchdog_thread.start()
    return watchdog_thread


def format_dataset_build_time(value):
    return value.strftime('%Y-%m-%d %H:%M:%S') if value else None


dataset_process_locks = {}
dataset_process_locks_lock = threading.Lock()

def get_dataset_lock(dataset_id):
    """获取指定数据集的处理锁"""
    dataset_id_str = str(dataset_id)
    with dataset_process_locks_lock:
        if dataset_id_str not in dataset_process_locks:
            dataset_process_locks[dataset_id_str] = threading.Lock()
        return dataset_process_locks[dataset_id_str]


def dataset_processing_response(dataset_id, action_name):
    """同一个 dataset_id 的耗时任务未完成时，拒绝重复提交。"""
    return jsonify({
        "code": 1,
        "data": {
            "message": f"{action_name}正在处理中，请勿重复提交",
            "dataset_id": dataset_id,
            "status": "processing"
        }
    }), 409


def _read_ffmpeg_stderr(stderr_file) -> str:
    """从 ffmpeg 的 stderr 临时文件读取全部内容（解码为文本）。"""
    if stderr_file is None:
        return ''
    try:
        stderr_file.seek(0)
        return stderr_file.read().decode('utf-8', errors='replace')
    except Exception:
        return ''


def normalize_posix_path(path):
    return str(path or '').strip().replace('\\', '/')


def map_label_task_path_to_local(path):
    path_str = normalize_posix_path(path)
    db_prefix = normalize_posix_path(LABEL_TASK_STORAGE_DB_PREFIX).rstrip('/')
    local_prefix = normalize_posix_path(LABEL_TASK_STORAGE_LOCAL_PREFIX).rstrip('/')
    if db_prefix and local_prefix and (path_str == db_prefix or path_str.startswith(f"{db_prefix}/")):
        suffix = path_str[len(db_prefix):].lstrip('/')
        return str(Path(local_prefix) / suffix)
    return path_str


def map_label_task_path_to_db(path):
    path_str = normalize_posix_path(path)
    db_prefix = normalize_posix_path(LABEL_TASK_STORAGE_DB_PREFIX).rstrip('/')
    local_prefix = normalize_posix_path(LABEL_TASK_STORAGE_LOCAL_PREFIX).rstrip('/')
    if db_prefix and local_prefix and (path_str == local_prefix or path_str.startswith(f"{local_prefix}/")):
        suffix = path_str[len(local_prefix):].lstrip('/')
        return f"{db_prefix}/{suffix}" if suffix else db_prefix
    return path_str


def scan_auto_label_image_dir(image_root: Path):
    image_paths = []
    existing_json_keys = set()

    for path in image_root.rglob('*'):
        if not path.is_file():
            continue

        relative_key = path.relative_to(image_root).with_suffix('').as_posix()
        suffix = path.suffix.lower()
        if suffix in DATASET_IMAGE_EXTENSIONS:
            image_paths.append(path)
        elif suffix == '.json':
            existing_json_keys.add(relative_key)

    return sorted(image_paths), existing_json_keys


def has_matching_labelme_json(img_path: Path, image_root: Path, existing_json_keys) -> bool:
    return img_path.relative_to(image_root).with_suffix('').as_posix() in existing_json_keys


def parse_trained_class_mapping(value):
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}

    mapping = {}
    if isinstance(value, dict):
        for key, raw_id in value.items():
            label = str(key).strip()
            try:
                cls_id = int(raw_id)
                if label:
                    mapping[label] = cls_id
                continue
            except (TypeError, ValueError):
                pass

            try:
                cls_id = int(key)
                label = str(raw_id).strip()
                if label:
                    mapping[label] = cls_id
            except (TypeError, ValueError):
                continue
    elif isinstance(value, list):
        for cls_id, label in enumerate(value):
            label = str(label).strip()
            if label:
                mapping[label] = cls_id

    return mapping


def normalize_auto_label_classes(classes_value):
    if classes_value is None:
        return []
    if isinstance(classes_value, bytes):
        classes_value = classes_value.decode('utf-8', errors='ignore')
    if isinstance(classes_value, str):
        raw_value = classes_value.strip()
        if not raw_value:
            return []
        try:
            parsed = json.loads(raw_value)
            return normalize_auto_label_classes(parsed)
        except json.JSONDecodeError:
            return [item.strip() for item in raw_value.split(',') if item.strip()]
    if isinstance(classes_value, dict):
        return [str(key).strip() for key in classes_value.keys() if str(key).strip()]
    if isinstance(classes_value, (list, tuple, set)):
        return [str(item).strip() for item in classes_value if str(item).strip()]
    return [str(classes_value).strip()] if str(classes_value).strip() else []


def build_model_class_mapping_from_yolo(model):
    names = getattr(model, 'names', {}) or {}
    if isinstance(names, dict):
        return {str(name).strip(): int(cls_id) for cls_id, name in names.items() if str(name).strip()}
    if isinstance(names, (list, tuple)):
        return {str(name).strip(): idx for idx, name in enumerate(names) if str(name).strip()}
    return {}


def resolve_auto_label_target_class_ids(classes_value, class_mapping):
    requested_classes = normalize_auto_label_classes(classes_value)
    if not requested_classes:
        raise ValueError("classes不能为空")

    target_class_ids = set()
    missing_classes = []
    for class_name in requested_classes:
        if class_name in class_mapping:
            target_class_ids.add(int(class_mapping[class_name]))
            continue
        try:
            target_class_ids.add(int(class_name))
        except (TypeError, ValueError):
            missing_classes.append(class_name)

    if missing_classes:
        available_classes = sorted(class_mapping.keys())
        raise ValueError(
            f"classes中存在未匹配的类别: {missing_classes}, 可用类别: {available_classes}"
        )
    if not target_class_ids:
        raise ValueError("classes未匹配到任何模型类别ID")

    return target_class_ids, requested_classes


def read_image_for_auto_label(img_path: Path):
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"自动标注依赖缺失: {e}")

    try:
        data = np.fromfile(str(img_path), dtype=np.uint8)
    except OSError:
        return None

    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def get_auto_label_name(model, cls_id, id_to_label):
    if cls_id in id_to_label:
        return id_to_label[cls_id]

    names = getattr(model, 'names', {})
    if isinstance(names, dict):
        return str(names.get(cls_id, names.get(str(cls_id), cls_id)))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def build_auto_labelme_json(img_path: Path, image_width, image_height, shapes):
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def run_auto_label_for_task_dirs(model, task_dirs, conf, target_class_ids, class_mapping):
    id_to_label = {int(cls_id): label for label, cls_id in class_mapping.items()}
    summary = {
        'task_dir_count': len(task_dirs),
        'total_images': 0,
        'pending_images': 0,
        'generated_count': 0,
        'skipped_existing_count': 0,
        'skipped_empty_count': 0,
        'unreadable_count': 0,
        'failed_count': 0,
        'generated_records': [],
        'failed_images': [],
        'task_dirs': [],
    }

    for task_dir in task_dirs:
        db_task_dir = normalize_posix_path(task_dir)
        local_task_dir = Path(map_label_task_path_to_local(db_task_dir))
        dir_result = {
            'task_dir': db_task_dir,
            'local_task_dir': str(local_task_dir),
            'total_images': 0,
            'pending_images': 0,
            'generated_count': 0,
            'skipped_existing_count': 0,
            'skipped_empty_count': 0,
            'unreadable_count': 0,
            'failed_count': 0,
        }
        summary['task_dirs'].append(dir_result)

        if not local_task_dir.exists():
            raise FileNotFoundError(f"标注任务目录不存在: {local_task_dir}")
        if not local_task_dir.is_dir():
            raise NotADirectoryError(f"标注任务路径不是目录: {local_task_dir}")

        image_paths, existing_json_keys = scan_auto_label_image_dir(local_task_dir)
        pending_count = sum(
            1 for img_path in image_paths
            if not has_matching_labelme_json(img_path, local_task_dir, existing_json_keys)
        )
        dir_result['total_images'] = len(image_paths)
        dir_result['pending_images'] = pending_count
        summary['total_images'] += len(image_paths)
        summary['pending_images'] += pending_count

        logger.info(
            f"[自动标注] 扫描目录完成: task_dir={db_task_dir}, local={local_task_dir}, "
            f"images={len(image_paths)}, existing_json={len(existing_json_keys)}, pending={pending_count}"
        )

        for img_path in image_paths:
            if has_matching_labelme_json(img_path, local_task_dir, existing_json_keys):
                dir_result['skipped_existing_count'] += 1
                summary['skipped_existing_count'] += 1
                continue

            img = read_image_for_auto_label(img_path)
            if img is None:
                dir_result['unreadable_count'] += 1
                summary['unreadable_count'] += 1
                continue

            image_height, image_width = img.shape[:2]
            try:
                results = model(img, conf=conf, verbose=False)
                shapes = []
                for result in results:
                    boxes = getattr(result, 'boxes', None)
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        if cls_id not in target_class_ids:
                            continue

                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
                        x1 = max(0.0, min(float(x1), float(image_width - 1)))
                        y1 = max(0.0, min(float(y1), float(image_height - 1)))
                        x2 = max(0.0, min(float(x2), float(image_width - 1)))
                        y2 = max(0.0, min(float(y2), float(image_height - 1)))
                        shapes.append({
                            "label": get_auto_label_name(model, cls_id, id_to_label),
                            "points": [
                                [round(x1, 2), round(y1, 2)],
                                [round(x2, 2), round(y2, 2)],
                            ],
                            "group_id": None,
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {},
                        })

                if not shapes:
                    dir_result['skipped_empty_count'] += 1
                    summary['skipped_empty_count'] += 1
                    continue

                json_path = img_path.with_suffix('.json')
                labelme_data = build_auto_labelme_json(img_path, image_width, image_height, shapes)
                json_path.write_text(
                    json.dumps(labelme_data, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )

                existing_json_keys.add(img_path.relative_to(local_task_dir).with_suffix('').as_posix())
                image_db_path = map_label_task_path_to_db(str(img_path))
                json_db_path = map_label_task_path_to_db(str(json_path))
                record = {
                    'image_name': img_path.name,
                    'image_path': str(img_path),
                    'image_db_path': image_db_path,
                    'json_path': str(json_path),
                    'json_db_path': json_db_path,
                    'shape_count': len(shapes),
                }
                summary['generated_records'].append(record)
                dir_result['generated_count'] += 1
                summary['generated_count'] += 1
            except Exception as e:
                dir_result['failed_count'] += 1
                summary['failed_count'] += 1
                summary['failed_images'].append({'image_path': str(img_path), 'error': str(e)})
                logger.error(f"[自动标注] 图片处理失败: image={img_path}, error={str(e)}")

    return summary


@app.route('/algorithm/dataset/auto-label', methods=['POST'])
def auto_label_dataset():
    """对标注任务目录中尚无 JSON 的图片执行模型预标注。"""
    lock_acquired = False
    label_task_id = None
    try:
        data = request.get_json(silent=True) or {}
        required_fields = ('label_task_id', 'model_id', 'conf', 'classes')
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "code": 1,
                "data": {"message": f"缺少必需参数: {', '.join(missing_fields)}"}
            }), 400

        try:
            label_task_id = validate_positive_id(data.get('label_task_id'), "label_task_id")
            model_id = validate_positive_id(data.get('model_id'), "model_id")
        except ValidationError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400

        try:
            conf = float(data.get('conf'))
        except (TypeError, ValueError):
            return jsonify({"code": 1, "data": {"message": "conf必须是数字"}}), 400
        if conf <= 0 or conf > 1:
            return jsonify({"code": 1, "data": {"message": "conf必须大于0且小于等于1"}}), 400

        lock_created, redis_key, redis_message = auto_label_redis_manager.start_task(
            label_task_id=label_task_id,
            model_id=model_id,
            conf=conf,
            classes=data.get('classes')
        )
        if not lock_created:
            if redis_key:
                return jsonify({
                    "code": 1,
                    "data": {
                        "message": "自动标注任务正在进行中，请勿重复提交",
                        "label_task_id": label_task_id,
                        "redis_key": redis_key,
                        "current": redis_message
                    }
                }), 409
            return jsonify({
                "code": 1,
                "data": {
                    "message": redis_message or "创建自动标注状态key失败",
                    "label_task_id": label_task_id
                }
            }), 500
        lock_acquired = True

        assignee_rows = db_manager.get_label_task_assignee_dirs(label_task_id)
        if assignee_rows is None:
            return jsonify({"code": 1, "data": {"message": "查询标注任务分配目录失败"}}), 500

        task_dirs = []
        seen_dirs = set()
        for row in assignee_rows:
            task_dir = normalize_posix_path(row.get('task_dir'))
            if task_dir and task_dir not in seen_dirs:
                task_dirs.append(task_dir)
                seen_dirs.add(task_dir)
        if not task_dirs:
            return jsonify({
                "code": 1,
                "data": {"message": f"未找到label_task_id={label_task_id}的task_dir"}
            }), 404

        weight_info = db_manager.get_trained_weight_for_auto_label(model_id)
        if not weight_info:
            return jsonify({"code": 1, "data": {"message": f"未找到model_id={model_id}的训练权重记录"}}), 404

        model_storage_address = weight_info.get('bestmodel_address')
        if not model_storage_address:
            return jsonify({"code": 1, "data": {"message": f"model_id={model_id}缺少bestmodel_address"}}), 400

        local_model_path = Path(ensure_local_path(model_storage_address))
        if not local_model_path.exists():
            return jsonify({"code": 1, "data": {"message": f"模型文件不存在: {local_model_path}"}}), 400

        logger.info(
            f"[自动标注] 收到请求: label_task_id={label_task_id}, model_id={model_id}, "
            f"conf={conf}, task_dir_count={len(task_dirs)}, model={local_model_path}"
        )

        model = YOLO(str(local_model_path))
        class_mapping = parse_trained_class_mapping(weight_info.get('class'))
        if not class_mapping:
            class_mapping = build_model_class_mapping_from_yolo(model)
        try:
            target_class_ids, requested_classes = resolve_auto_label_target_class_ids(
                data.get('classes'),
                class_mapping
            )
        except ValueError as e:
            return jsonify({"code": 1, "data": {"message": str(e)}}), 400

        result = run_auto_label_for_task_dirs(
            model=model,
            task_dirs=task_dirs,
            conf=conf,
            target_class_ids=target_class_ids,
            class_mapping=class_mapping
        )

        update_result = db_manager.update_auto_label_items(
            label_task_id,
            result['generated_records']
        )
        if update_result is None:
            return jsonify({"code": 1, "data": {"message": "自动标注完成，但写回label_task_item失败"}}), 500

        response_data = {
            "message": "自动标注完成",
            "label_task_id": label_task_id,
            "model_id": model_id,
            "conf": conf,
            "classes": requested_classes,
            "target_class_ids": sorted(target_class_ids),
            "task_dir_count": result['task_dir_count'],
            "total_images": result['total_images'],
            "pending_images": result['pending_images'],
            "generated_count": result['generated_count'],
            "updated_count": update_result['updated_count'],
            "skipped_existing_count": result['skipped_existing_count'],
            "skipped_empty_count": result['skipped_empty_count'],
            "unreadable_count": result['unreadable_count'],
            "failed_count": result['failed_count'],
            "task_dirs": result['task_dirs'],
            "missing_items": update_result['missing_items'][:50],
            "failed_images": result['failed_images'][:50],
            "generated_records": result['generated_records'][:50],
            "redis_key": redis_key,
        }
        return jsonify({"code": 0, "data": response_data})
    except FileNotFoundError as e:
        logger.error(f"[自动标注] 路径不存在: {str(e)}")
        return jsonify({"code": 1, "data": {"message": f"路径不存在: {str(e)}"}}), 404
    except Exception as e:
        return handle_api_exception(e, "自动标注")
    finally:
        if lock_acquired and label_task_id is not None:
            auto_label_redis_manager.finish_task(label_task_id)


@app.route('/algorithm/train/predict', methods=['POST'])
def predict_with_model():
    """使用训练好的模型进行推理预测"""
    try:
        data = request.get_json()
        
        model_id = data.get('modelId')
        prediction_id = data.get('predictionId')
        confidence = data.get('confidence', 0.25)

        if not model_id or prediction_id is None:
            return jsonify({
                "code": 1,
                "data": "缺少必要参数: modelId 或 predictionId"
            }), 400

        try:
            model_id = validate_positive_id(model_id, "modelId")
            prediction_id = validate_positive_id(prediction_id, "predictionId")
        except ValidationError as e:
            return jsonify({"code": 1, "data": str(e)}), 400

        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.25
        
        logger.info(f"收到推理请求: modelId={model_id}, predictionId={prediction_id}, confidence={confidence}")
        
        predict_redis_manager.update_status(model_id, prediction_id, 'pending')
        connection = None
        try:
            connection = db_manager.get_connection()
            if connection:
                with connection.cursor() as cursor:
                    sql = "UPDATE train_prediction_record SET status = %s, error_message = NULL WHERE id = %s"
                    cursor.execute(sql, ('pending', prediction_id))
                    connection.commit()
                logger.info(f"已更新预测记录初始状态: prediction_id={prediction_id}, status=pending")
        except Exception as e:
            logger.error(f"更新预测记录初始状态失败: {str(e)}")
        finally:
            if connection:
                connection.close()
        
        # 启动异步推理线程
        def async_predict(model_id, prediction_id, confidence):
            import cv2

            def update_prediction_record(prediction_id, status, error_msg=None):
                connection = None
                try:
                    connection = db_manager.get_connection()
                    if connection:
                        with connection.cursor() as cursor:
                            if error_msg:
                                sql = "UPDATE train_prediction_record SET status = %s, error_message = %s WHERE id = %s"
                                cursor.execute(sql, (status, error_msg, prediction_id))
                            else:
                                sql = "UPDATE train_prediction_record SET status = %s, error_message = NULL WHERE id = %s"
                                cursor.execute(sql, (status, prediction_id))
                            connection.commit()
                        logger.info(f"已更新预测记录: prediction_id={prediction_id}, status={status}")
                except Exception as e:
                    logger.error(f"更新预测记录失败: {str(e)}")
                finally:
                    if connection:
                        connection.close()
            
            try:
                logger.info(f"[异步推理] 开始: modelId={model_id}, predictionId={prediction_id}")
                
                predict_redis_manager.update_status(model_id, prediction_id, 'processing')
                update_prediction_record(prediction_id, 'processing')
                
                connection = db_manager.get_connection()
                if not connection:
                    error_msg = "数据库连接失败"
                    logger.error(f"[异步推理] {error_msg}")
                    predict_redis_manager.update_status(model_id, prediction_id, 'failed')
                    update_prediction_record(prediction_id, 'failed', error_msg)
                    return
                
                try:
                    with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                        sql = "SELECT bestmodel_address FROM trained_weights WHERE id = %s"
                        cursor.execute(sql, (model_id,))
                        result = cursor.fetchone()
                        
                        if not result or not result.get('bestmodel_address'):
                            error_msg = f"未找到modelId={model_id}的模型地址"
                            logger.error(f"[异步推理] {error_msg}")
                            predict_redis_manager.update_status(model_id, prediction_id, 'failed')
                            update_prediction_record(prediction_id, 'failed', error_msg)
                            return
                        
                        model_storage_address = result['bestmodel_address']
                        logger.info(f"[异步推理] 模型存储地址: {model_storage_address}")
                finally:
                    connection.close()
                
                local_base_path = build_storage_local_path("predict", model_id, prediction_id, require_exists=False)
                local_original_path = build_storage_local_path("predict", model_id, prediction_id, "original", require_exists=False)
                local_results_path = local_base_path / "results"

                if local_results_path.exists():
                    safe_rmtree(local_results_path)
                local_original_path.mkdir(parents=True, exist_ok=True)
                local_results_path.mkdir(parents=True, exist_ok=True)

                original_object_prefix = build_storage_object_key("predict", model_id, prediction_id, "original")
                downloaded_count = download_storage_prefix(
                    original_object_prefix,
                    local_original_path,
                    suffixes=DATASET_IMAGE_EXTENSIONS | DATASET_VIDEO_EXTENSIONS
                )
                logger.info(
                    f"[异步推理] 原始素材已定位到本地工作区: "
                    f"{original_object_prefix} -> {local_original_path}, count={downloaded_count}"
                )

                local_model_path = ensure_local_path(model_storage_address)
                logger.info(f"[异步推理] 模型本地路径: {local_model_path}")

                if not Path(local_model_path).exists():
                    error_msg = f"模型文件不存在: {local_model_path}"
                    logger.error(f"[异步推理] {error_msg}")
                    predict_redis_manager.update_status(model_id, prediction_id, 'failed')
                    update_prediction_record(prediction_id, 'failed', error_msg)
                    return

                logger.info(f"[异步推理] 加载模型并开始推理，置信度: {confidence}")

                model = YOLO(local_model_path)

                image_files = []
                video_files = []
                for entry in local_original_path.iterdir():
                    if not entry.is_file():
                        continue
                    suffix = entry.suffix.lower()
                    if suffix in DATASET_IMAGE_EXTENSIONS:
                        image_files.append(entry)
                    elif suffix in DATASET_VIDEO_EXTENSIONS:
                        video_files.append(entry)

                image_files = list(set(image_files))
                video_files = list(set(video_files))

                logger.info(f"[异步推理] 待推理图片数量: {len(image_files)}, 视频数量: {len(video_files)}")
                logger.info(f"[异步推理] 图片列表: {[f.name for f in image_files]}")
                logger.info(f"[异步推理] 视频列表: {[f.name for f in video_files]}")

                if not image_files and not video_files:
                    error_msg = f"本地原始素材目录中未找到任何图片或视频: {local_original_path}"
                    logger.error(f"[异步推理] {error_msg}")
                    predict_redis_manager.update_status(model_id, prediction_id, 'failed')
                    update_prediction_record(prediction_id, 'failed', error_msg)
                    return

                if image_files:
                    image_only_dir = local_results_path.parent / f"{local_results_path.name}_imgsrc"
                    if image_only_dir.exists():
                        safe_rmtree(image_only_dir)
                    image_only_dir.mkdir(parents=True, exist_ok=True)
                    for img_path in image_files:
                        shutil.copy2(str(img_path), str(image_only_dir / img_path.name))

                    try:
                        results = model.predict(
                            source=str(image_only_dir),
                            conf=confidence,
                            save=True,
                            project=str(local_results_path.parent),
                            name=local_results_path.name,
                            exist_ok=True
                        )
                        logger.info(f"[异步推理] 图片批量推理完成，处理了 {len(results)} 张图片")

                        stem_to_files = {}
                        for img_path in image_files:
                            stem = img_path.stem
                            if stem not in stem_to_files:
                                stem_to_files[stem] = []
                            stem_to_files[stem].append(img_path)

                        for stem, files in stem_to_files.items():
                            if len(files) > 1:
                                logger.info(f"[异步推理] 检测到同名文件冲突: {[f.name for f in files]}，重新单独推理")
                                for img_path in files:
                                    try:
                                        result = model.predict(
                                            source=str(img_path),
                                            conf=confidence,
                                            save=False
                                        )
                                        if result and len(result) > 0:
                                            result_img = result[0].plot()
                                            result_file = local_results_path / img_path.name
                                            cv2.imwrite(str(result_file), result_img)
                                            logger.debug(f"[异步推理] 单独保存结果: {img_path.name}")
                                    except Exception as e:
                                        logger.error(f"[异步推理] 单独推理失败 {img_path.name}: {str(e)}")
                            else:
                                img_path = files[0]
                                if img_path.suffix.lower() == '.png':
                                    jpg_result = local_results_path / f"{stem}.jpg"
                                    png_result = local_results_path / f"{stem}.png"
                                    if jpg_result.exists() and not png_result.exists():
                                        jpg_result.rename(png_result)
                                        logger.debug(f"[异步推理] 重命名结果: {jpg_result.name} -> {png_result.name}")

                    except Exception as e:
                        logger.error(f"[异步推理] 批量推理失败: {str(e)}")
                        logger.info(f"[异步推理] 降级为逐张处理模式")
                        for img_path in image_files:
                            try:
                                result = model.predict(
                                    source=str(img_path),
                                    conf=confidence,
                                    save=False
                                )
                                if result and len(result) > 0:
                                    result_img = result[0].plot()
                                    result_file = local_results_path / img_path.name
                                    cv2.imwrite(str(result_file), result_img)
                                logger.debug(f"[异步推理] 完成: {img_path.name}")
                            except Exception as e2:
                                logger.error(f"[异步推理] 推理失败 {img_path.name}: {str(e2)}")
                    finally:
                        if image_only_dir.exists():
                            safe_rmtree(image_only_dir)

                seen_video_outputs = set()
                ffmpeg_bin = shutil.which('ffmpeg')
                if not ffmpeg_bin:
                    logger.warning("[异步推理] 未检测到系统 ffmpeg，mp4 输出将回退到 OpenCV (mp4v/XVID)，无法生成 H.264")
                for video_path in video_files:
                    cap = None
                    writer = None
                    ffmpeg_proc = None
                    ffmpeg_stderr_file = None
                    try:
                        logger.info(f"[异步推理] 开始处理视频: {video_path.name}")
                        cap = cv2.VideoCapture(str(video_path))
                        if not cap.isOpened():
                            logger.error(f"[异步推理] 无法打开视频: {video_path.name}")
                            continue

                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        cap = None

                        out_suffix = video_path.suffix.lower()
                        if out_suffix in {'.mp4', '.m4v', '.mov'}:
                            base_name = f"{video_path.stem}.mp4"
                        else:
                            base_name = f"{video_path.stem}.avi"

                        output_name = base_name
                        if output_name in seen_video_outputs:
                            stem, ext = base_name.rsplit('.', 1)
                            disambig = video_path.suffix.lstrip('.')
                            candidate = f"{stem}_{disambig}.{ext}"
                            counter = 1
                            while candidate in seen_video_outputs:
                                candidate = f"{stem}_{disambig}_{counter}.{ext}"
                                counter += 1
                            output_name = candidate
                        seen_video_outputs.add(output_name)

                        result_video_path = local_results_path / output_name

                        is_mp4_family = out_suffix in {'.mp4', '.m4v', '.mov'}
                        use_ffmpeg = bool(ffmpeg_bin) and is_mp4_family

                        if use_ffmpeg:
                            target_w = width - (width % 2)
                            target_h = height - (height % 2)
                            # stderr 落临时文件而非 PIPE：逐帧写 stdin 的同时若不持续读 stderr，
                            # ffmpeg 的 stderr 写满管道缓冲后会阻塞，与我们等待写 stdin 形成死锁。
                            ffmpeg_stderr_file = tempfile.TemporaryFile()
                            ffmpeg_proc = subprocess.Popen(
                                [
                                    ffmpeg_bin, '-y', '-loglevel', 'error',
                                    '-f', 'rawvideo', '-vcodec', 'rawvideo',
                                    '-s', f'{target_w}x{target_h}',
                                    '-pix_fmt', 'bgr24',
                                    '-r', f'{fps}',
                                    '-i', '-',
                                    '-c:v', 'libx264',
                                    '-pix_fmt', 'yuv420p',
                                    '-preset', 'medium',
                                    '-crf', '23',
                                    '-movflags', '+faststart',
                                    str(result_video_path),
                                ],
                                stdin=subprocess.PIPE,
                                stderr=ffmpeg_stderr_file,
                            )
                        else:
                            target_w, target_h = width, height
                            if is_mp4_family:
                                writer = cv2.VideoWriter(str(result_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                            else:
                                writer = cv2.VideoWriter(str(result_video_path), cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
                            if not writer.isOpened():
                                logger.error(f"[异步推理] 无法创建输出视频文件: {result_video_path}")
                                continue

                        frame_idx = 0
                        for result in model.predict(source=str(video_path), conf=confidence, stream=True, save=False, verbose=False):
                            annotated = result.plot()
                            if annotated.shape[1] != target_w or annotated.shape[0] != target_h:
                                annotated = cv2.resize(annotated, (target_w, target_h))
                            if use_ffmpeg:
                                try:
                                    ffmpeg_proc.stdin.write(annotated.tobytes())
                                except BrokenPipeError:
                                    err_msg = _read_ffmpeg_stderr(ffmpeg_stderr_file)
                                    logger.error(f"[异步推理] ffmpeg 编码中断: {err_msg}")
                                    break
                            else:
                                writer.write(annotated)
                            frame_idx += 1
                            if total_frames and frame_idx % 100 == 0:
                                logger.debug(f"[异步推理] {video_path.name} 已处理 {frame_idx}/{total_frames} 帧")

                        logger.info(f"[异步推理] 视频处理完成: {video_path.name} -> {output_name}, 帧数={frame_idx}")
                    except Exception as e:
                        logger.error(f"[异步推理] 视频处理异常 {video_path.name}: {str(e)}")
                    finally:
                        if cap is not None:
                            cap.release()
                        if writer is not None:
                            writer.release()
                        if ffmpeg_proc is not None:
                            try:
                                if ffmpeg_proc.stdin:
                                    ffmpeg_proc.stdin.close()
                            except Exception:
                                pass
                            try:
                                returncode = ffmpeg_proc.wait(timeout=60)
                                if returncode != 0:
                                    err_msg = _read_ffmpeg_stderr(ffmpeg_stderr_file)
                                    logger.error(f"[异步推理] ffmpeg 退出码 {returncode}: {err_msg.strip()}")
                            except subprocess.TimeoutExpired:
                                ffmpeg_proc.kill()
                                ffmpeg_proc.wait()
                                logger.error(f"[异步推理] ffmpeg 超时被强制终止: {video_path.name}")
                        if ffmpeg_stderr_file is not None:
                            try:
                                ffmpeg_stderr_file.close()
                            except Exception:
                                pass

                logger.info(f"[异步推理] 推理完成，结果保存到: {local_results_path}")

                results_storage_dir = build_storage_path('predict', model_id, prediction_id, 'results')

                actual_results_path = local_results_path
                if actual_results_path.exists():
                    has_result_files = any(
                        f.is_file() and f.suffix.lower() in (DATASET_IMAGE_EXTENSIONS | DATASET_VIDEO_EXTENSIONS)
                        for f in actual_results_path.iterdir()
                    )
                    if not has_result_files:
                        subdirs = [d for d in actual_results_path.iterdir() if d.is_dir()]
                        if subdirs:
                            actual_results_path = subdirs[0]

                results_object_prefix = parse_storage_relative_path(results_storage_dir)
                success_count, fail_count, failed_files = persist_directory_to_storage(
                    actual_results_path,
                    results_object_prefix
                )
                if fail_count:
                    error_msg = f"保存推理结果到本地工作区失败: fail_count={fail_count}, failed_files={failed_files[:5]}"
                    logger.error(f"[异步推理] {error_msg}")
                    predict_redis_manager.update_status(model_id, prediction_id, 'failed')
                    update_prediction_record(prediction_id, 'failed', error_msg)
                    return

                logger.info(f"[异步推理] 结果已保存到本地工作区: {results_storage_dir}, count={success_count}")
                
                predict_redis_manager.update_status(model_id, prediction_id, 'completed')
                update_prediction_record(prediction_id, 'completed')
                logger.info(f"[异步推理] 任务完成: modelId={model_id}, predictionId={prediction_id}")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[异步推理] 异常: {error_msg}")
                logger.error(traceback.format_exc())
                predict_redis_manager.update_status(model_id, prediction_id, 'failed')
                update_prediction_record(prediction_id, 'failed', error_msg)
        
        thread = threading.Thread(
            target=async_predict,
            args=(model_id, prediction_id, confidence),
            daemon=True
        )
        thread.start()

        return jsonify({
            "code": 0,
            "data": "开始推理"
        })
            
    except Exception as e:
        logger.error(f"预测接口错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "code": 1,
            "data": f"服务器内部错误: {str(e)}"
        }), 500


if __name__ == '__main__':
    logger.info("训练任务管理API启动")
    logger.info(f"数据库配置: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    logger.info(f"训练脚本路径: {Path(__file__).parent / 'train_method' / 'trainer.py'}")
    
    app.run(host='0.0.0.0', port=12241, debug=False)
