# -*- coding:utf-8 -*-
"""
输入验证模块
提供统一的输入验证函数
"""
import re
from typing import Any, Optional

# 使用 services.exceptions 中的 ValidationError
try:
    from services.exceptions import ValidationError
except ImportError:
    class ValidationError(Exception):
        """输入验证异常"""

        def __init__(self, message: str):
            super().__init__(message)
            self.message = message


def validate_dataset_id(dataset_id: Any) -> int:
    """
    验证并转换 dataset_id 为整数

    Args:
        dataset_id: 要验证的 dataset_id

    Returns:
        验证后的整数值

    Raises:
        ValidationError: 验证失败时抛出
    """
    if dataset_id is None:
        raise ValidationError("dataset_id 不能为空")

    try:
        id_value = int(dataset_id)
    except (TypeError, ValueError):
        raise ValidationError(f"dataset_id 必须是整数，收到: {dataset_id}")

    if id_value <= 0:
        raise ValidationError(f"dataset_id 必须是正整数，收到: {id_value}")

    if id_value > 2147483647:  # MySQL INT 上限
        raise ValidationError(f"dataset_id 超出范围: {id_value}")

    return id_value


def validate_task_id(task_id: Any) -> int:
    """
    验证并转换 task_id 为整数

    Args:
        task_id: 要验证的 task_id

    Returns:
        验证后的整数值

    Raises:
        ValidationError: 验证失败时抛出
    """
    if task_id is None:
        raise ValidationError("task_id 不能为空")

    try:
        id_value = int(task_id)
    except (TypeError, ValueError):
        raise ValidationError(f"task_id 必须是整数，收到: {task_id}")

    if id_value <= 0:
        raise ValidationError(f"task_id 必须是正整数，收到: {id_value}")

    return id_value


def validate_status(status: Any, valid_values: tuple = (0, 1)) -> int:
    """
    验证 status 值

    Args:
        status: 要验证的 status 值
        valid_values: 有效的 status 值元组

    Returns:
        验证后的 status 值

    Raises:
        ValidationError: 验证失败时抛出
    """
    if status is None:
        raise ValidationError("status 不能为空")

    if not isinstance(status, int):
        try:
            status = int(status)
        except (TypeError, ValueError):
            raise ValidationError(f"status 必须是整数，收到: {status}")

    if status not in valid_values:
        raise ValidationError(f"status 必须是 {valid_values} 之一，收到: {status}")

    return status


def validate_string(value: Any, field_name: str, max_length: int = 255, required: bool = True) -> Optional[str]:
    """
    验证字符串输入

    Args:
        value: 要验证的值
        field_name: 字段名称（用于错误信息）
        max_length: 最大长度
        required: 是否必填

    Returns:
        验证后的字符串，如果非必填且为空则返回 None

    Raises:
        ValidationError: 验证失败时抛出
    """
    if value is None:
        if required:
            raise ValidationError(f"{field_name} 不能为空")
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if not value:
        if required:
            raise ValidationError(f"{field_name} 不能为空")
        return None

    if len(value) > max_length:
        raise ValidationError(f"{field_name} 长度不能超过 {max_length}，收到: {len(value)}")

    return value


def validate_integer(value: Any, field_name: str, min_value: int = None, max_value: int = None, required: bool = True) -> Optional[int]:
    """
    验证整数输入

    Args:
        value: 要验证的值
        field_name: 字段名称（用于错误信息）
        min_value: 最小值
        max_value: 最大值
        required: 是否必填

    Returns:
        验证后的整数值，如果非必填且为空则返回 None

    Raises:
        ValidationError: 验证失败时抛出
    """
    if value is None:
        if required:
            raise ValidationError(f"{field_name} 不能为空")
        return None

    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} 必须是整数，收到: {value}")

    if min_value is not None and int_value < min_value:
        raise ValidationError(f"{field_name} 不能小于 {min_value}，收到: {int_value}")

    if max_value is not None and int_value > max_value:
        raise ValidationError(f"{field_name} 不能大于 {max_value}，收到: {int_value}")

    return int_value


def sanitize_string(value: str, max_length: int = 255) -> str:
    """
    清理字符串输入，移除潜在危险字符

    Args:
        value: 要清理的字符串
        max_length: 最大长度

    Returns:
        清理后的字符串
    """
    if not isinstance(value, str):
        return str(value)[:max_length]

    # 移除路径遍历字符
    cleaned = value.replace('..', '').replace('\\', '/')

    # 移除空字节
    cleaned = cleaned.replace('\x00', '')

    # 移除控制字符（保留换行和制表符）
    cleaned = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)

    return cleaned[:max_length]


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不安全字符

    Args:
        filename: 要清理的文件名

    Returns:
        清理后的文件名
    """
    if not isinstance(filename, str):
        filename = str(filename)

    # 只保留字母、数字、下划线、连字符、点号
    cleaned = re.sub(r'[^\w\-.]', '_', filename)

    # 移除连续的点号（防止路径遍历）
    cleaned = re.sub(r'\.{2,}', '.', cleaned)

    # 移除开头和结尾的点号和下划线
    cleaned = cleaned.strip('._')

    return cleaned or 'unnamed'


def validate_json_body(data: Any, required_fields: list = None) -> dict:
    """
    验证 JSON 请求体

    Args:
        data: 请求体数据
        required_fields: 必填字段列表

    Returns:
        验证后的数据字典

    Raises:
        ValidationError: 验证失败时抛出
    """
    if data is None:
        raise ValidationError("请求数据为空")

    if not isinstance(data, dict):
        raise ValidationError("请求数据必须是 JSON 对象")

    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(f"缺少必需参数: {', '.join(missing_fields)}")

    return data
