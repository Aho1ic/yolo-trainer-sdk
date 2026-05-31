#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pose 标注共享工具。

统一处理 LabelMe JSON 中 rectangle/point 的关系，避免统计阶段和
json2txt 转换阶段使用不同规则。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RectangleEntry:
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    shape_index: int


@dataclass(frozen=True)
class PointEntry:
    label: str
    x: float
    y: float
    shape_index: int


@dataclass
class PoseShapeAssignment:
    rectangles: List[RectangleEntry]
    points: List[PointEntry]
    keypoints_by_rect: List[Dict[str, Tuple[float, float]]]
    keypoint_labels: List[str]
    shape_types: List[str]
    unassigned_points: List[PointEntry]
    duplicate_points: List[PointEntry]


def infer_shape_type(shape: dict) -> Optional[str]:
    """推断 shape 类型，兼容缺失 shape_type 的旧 JSON。"""
    shape_type = str(shape.get('shape_type') or '').strip().lower()
    if shape_type:
        return shape_type

    points = shape.get('points') or []
    if len(points) == 1:
        return 'point'
    if len(points) == 2:
        return 'rectangle'
    if len(points) >= 3:
        return 'polygon'
    return None


def rectangle_bbox_from_points(points) -> Optional[Tuple[float, float, float, float]]:
    if not points or len(points) < 2:
        return None
    try:
        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])
    except (TypeError, ValueError, IndexError):
        return None

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def rectangle_from_shape(shape: dict, shape_index: int) -> Optional[RectangleEntry]:
    label = str(shape.get('label') or '').strip()
    if not label:
        return None

    bbox = rectangle_bbox_from_points(shape.get('points') or [])
    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox
    return RectangleEntry(
        label=label,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        shape_index=shape_index,
    )


def point_from_shape(shape: dict, shape_index: int) -> Optional[PointEntry]:
    label = str(shape.get('label') or '').strip()
    points = shape.get('points') or []
    if not label or not points:
        return None
    try:
        px, py = float(points[0][0]), float(points[0][1])
    except (TypeError, ValueError, IndexError):
        return None
    return PointEntry(label=label, x=px, y=py, shape_index=shape_index)


def point_in_rectangle(point: PointEntry, rect: RectangleEntry) -> bool:
    return rect.x_min <= point.x <= rect.x_max and rect.y_min <= point.y <= rect.y_max


def sort_keypoint_labels(labels: Iterable[str]) -> List[str]:
    """关键点标签顺序：全数字时按数值排序，否则保持首次出现顺序。"""
    ordered = []
    for label in labels:
        label = str(label).strip()
        if label and label not in ordered:
            ordered.append(label)

    if ordered and all(label.isdigit() for label in ordered):
        ordered.sort(key=lambda value: int(value))
    return ordered


def assign_points_to_rectangles(shapes: Iterable[dict]) -> PoseShapeAssignment:
    """把 point 分配给第一个包含它的 rectangle。

    返回的 keypoint_labels 只包含已成功归属到 rectangle 的 point 标签。
    框外 point 不参与 kpt_num/kpt_labels 统计，也不会进入 YOLO pose TXT。
    """
    rectangles: List[RectangleEntry] = []
    points: List[PointEntry] = []
    shape_types: List[str] = []

    for shape_index, shape in enumerate(shapes or []):
        if not isinstance(shape, dict):
            continue
        shape_type = infer_shape_type(shape)
        if shape_type:
            shape_types.append(shape_type)

        if shape_type == 'rectangle':
            rect = rectangle_from_shape(shape, shape_index)
            if rect is not None:
                rectangles.append(rect)
        elif shape_type == 'point':
            point = point_from_shape(shape, shape_index)
            if point is not None:
                points.append(point)

    keypoints_by_rect: List[Dict[str, Tuple[float, float]]] = [dict() for _ in rectangles]
    keypoint_labels: List[str] = []
    unassigned_points: List[PointEntry] = []
    duplicate_points: List[PointEntry] = []

    for point in points:
        assigned = False
        for rect_index, rect in enumerate(rectangles):
            if not point_in_rectangle(point, rect):
                continue

            assigned = True
            rect_kpts = keypoints_by_rect[rect_index]
            if point.label in rect_kpts:
                duplicate_points.append(point)
            else:
                rect_kpts[point.label] = (point.x, point.y)
                if point.label not in keypoint_labels:
                    keypoint_labels.append(point.label)
            break

        if not assigned:
            unassigned_points.append(point)

    return PoseShapeAssignment(
        rectangles=rectangles,
        points=points,
        keypoints_by_rect=keypoints_by_rect,
        keypoint_labels=keypoint_labels,
        shape_types=shape_types,
        unassigned_points=unassigned_points,
        duplicate_points=duplicate_points,
    )
