#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON文件转换为YOLO格式的TXT
支持检测(detect)和分割(segment)两种模式
python json2txt.py --duty_type detect --original_train_data_address "/path/to/data" --labels duckweed ship bird
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .pose_labels import assign_points_to_rectangles
    from .annotation_paths import resolve_annotation_dir
except ImportError:
    from pose_labels import assign_points_to_rectangles
    from annotation_paths import resolve_annotation_dir

logger = logging.getLogger(__name__)

# 并行处理的最大线程数
MAX_WORKERS = 16

class DutyType(Enum):
    DETECT = "detect"
    SEGMENT = "segment"
    POSE = "pose"


class UnknownLabelError(ValueError):
    """预定义标签模式下出现了未在预定义列表中的标签"""

    def __init__(self, label: str):
        super().__init__(f"标签 '{label}' 不在预定义标签列表中")
        self.label = label


@dataclass
class BoundingBox:
    """边界框数据结构"""
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    class_name: str


@dataclass
class PoseAnnotation:
    """Pose 标注数据结构（一个矩形 + 其关联的关键点序列）"""
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    class_name: str
    keypoints: List[Tuple[float, float, int]]  # [(x_norm, y_norm, visibility), ...]，长度等于 K


class UnifiedJsonConverter:
    """JSON转换器"""
    
    def __init__(self, duty_type: str, original_train_data_address: str,
                 class_mapping: Optional[Dict[str, int]] = None,
                 predefined_labels: Optional[List[str]] = None,
                 kpt_labels: Optional[List[str]] = None,
                 kpt_num: Optional[int] = None):
        """
        初始化转换器

        Args:
            duty_type: 任务类型 'detect' / 'segment' / 'pose'
            original_train_data_address: 原始训练数据地址
            class_mapping: 类别名称到ID的映射，如果为None则自动生成
            predefined_labels: 预定义的矩形类别标签列表，按顺序分配ID
            kpt_labels: pose 模式下关键点标签的规范顺序（长度即 K）
            kpt_num: pose 模式下关键点数量 K（kpt_labels 已传时可省略）
        """
        valid_types = {DutyType.DETECT.value, DutyType.SEGMENT.value, DutyType.POSE.value}
        if duty_type not in valid_types:
            raise ValueError(f"不支持的duty_type: {duty_type}，必须是 'detect' / 'segment' / 'pose'")

        self.duty_type = duty_type
        self.original_train_data_address = Path(original_train_data_address)
        self.images_dir = self.original_train_data_address / 'images'
        self.json_dir = resolve_annotation_dir(self.original_train_data_address)
        self.labels_dir = self.original_train_data_address / 'labels'

        # 处理预定义标签和类别映射
        self.predefined_labels = predefined_labels or []
        self.predefined_label_map = {label: idx for idx, label in enumerate(self.predefined_labels)} if self.predefined_labels else {}
        if class_mapping:
            self.class_mapping = class_mapping.copy()
        elif self.predefined_labels:
            # 根据预定义标签创建类别映射
            self.class_mapping = self.predefined_label_map.copy()
            logger.debug(f"使用预定义标签顺序: {self.predefined_labels}")
            logger.debug(f"生成类别映射: {self.class_mapping}")
        else:
            self.class_mapping = {}

        # 设置自动分配ID的起始值
        self.auto_class_id = len(self.class_mapping)

        # get_class_id 在并行转换线程中会修改 class_mapping/auto_class_id，
        # 用锁保护共享状态，避免 race。
        self._class_mapping_lock = threading.Lock()

        # pose 模式相关：关键点标签顺序与数量
        self.kpt_labels = list(kpt_labels) if kpt_labels else []
        self.kpt_label_index = {label: idx for idx, label in enumerate(self.kpt_labels)}
        if kpt_num is not None:
            self.kpt_num = int(kpt_num)
        else:
            self.kpt_num = len(self.kpt_labels)
        if self.duty_type == DutyType.POSE.value:
            if self.kpt_num <= 0:
                raise ValueError("pose 模式必须提供 kpt_labels 或正整数的 kpt_num")
            if self.kpt_labels and len(self.kpt_labels) != self.kpt_num:
                raise ValueError(
                    f"kpt_labels 长度({len(self.kpt_labels)}) 与 kpt_num({self.kpt_num}) 不一致"
                )

        # 验证路径
        self._validate_paths()

        logger.debug(f"初始化转换器: duty_type={duty_type}")
        logger.debug(f"原始数据路径: {self.original_train_data_address}")
        logger.debug(f"图片目录: {self.images_dir}")
        logger.debug(f"JSON目录: {self.json_dir}")
        logger.debug(f"输出标签目录: {self.labels_dir}")
        if self.duty_type == DutyType.POSE.value:
            logger.debug(f"关键点数量(K): {self.kpt_num}, 关键点顺序: {self.kpt_labels}")
    
    def _validate_paths(self):
        """验证路径存在性"""
        if not self.original_train_data_address.exists():
            raise FileNotFoundError(f"原始训练数据路径不存在: {self.original_train_data_address}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {self.images_dir}")
        
        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON目录不存在: {self.original_train_data_address} (需包含 json 或 jsons)")
    
    def get_class_id(self, class_name: str) -> int:
        """获取类别ID

        - 预定义标签模式：标签必须在预定义列表中，否则抛出 UnknownLabelError。
          （此前会自动分配新 ID，导致生成 class_id >= nc 的 TXT，
          而 YAML 的 nc/names 仍按预定义 labels 生成，训练前检查必然失败。）
        - 无预定义标签模式：按首次出现顺序自动分配 ID（持锁，线程安全）。
        """
        # 预定义标签模式：只接受预定义列表中的标签
        if self.predefined_label_map:
            if class_name in self.predefined_label_map:
                return self.predefined_label_map[class_name]
            logger.error(
                f"标签 '{class_name}' 不在预定义标签列表中: {self.predefined_labels}"
            )
            raise UnknownLabelError(class_name)

        # 无预定义标签：自动分配新 ID（持锁保护共享状态）
        with self._class_mapping_lock:
            if class_name in self.class_mapping:
                return self.class_mapping[class_name]
            self.class_mapping[class_name] = self.auto_class_id
            logger.debug(f"新增类别映射: {class_name} -> {self.auto_class_id}")
            self.auto_class_id += 1
            return self.class_mapping[class_name]
    
    def polygon_to_bbox(self, points: List[List[float]]) -> Tuple[float, float, float, float]:
        """
        将多边形转换为边界框
        
        Args:
            points: 多边形点坐标列表 [[x1,y1], [x2,y2], ...]
        
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        if not points:
            raise ValueError("多边形点列表不能为空")
        
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return x_min, y_min, x_max, y_max
    
    def rectangle_to_bbox(self, points: List[List[float]]) -> Tuple[float, float, float, float]:
        """
        将矩形点转换为边界框
        
        Args:
            points: 矩形的两个对角点 [[x1,y1], [x2,y2]]
        
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        if len(points) != 2:
            # 如果不是两个点，按多边形处理
            return self.polygon_to_bbox(points)
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        return x_min, y_min, x_max, y_max
    
    def bbox_to_yolo_format(self, x_min: float, y_min: float, x_max: float, y_max: float,
                            img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        将边界框转换为YOLO格式（相对坐标）
        
        Args:
            x_min, y_min, x_max, y_max: 绝对坐标边界框
            img_width, img_height: 图像尺寸
        
        Returns:
            (x_center, y_center, width, height) - 相对坐标
        """
        # 计算中心点和宽高
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # 转换为相对坐标
        x_center_rel = x_center / img_width
        y_center_rel = y_center / img_height
        width_rel = width / img_width
        height_rel = height / img_height
        
        # 确保坐标在[0,1]范围内
        x_center_rel = max(0.0, min(1.0, x_center_rel))
        y_center_rel = max(0.0, min(1.0, y_center_rel))
        width_rel = max(0.0, min(1.0, width_rel))
        height_rel = max(0.0, min(1.0, height_rel))
        
        return x_center_rel, y_center_rel, width_rel, height_rel
    
    def convert_json_for_detect(self, json_path: str) -> List[BoundingBox]:
        """
        检测模式：将JSON转换为边界框列表
        
        Args:
            json_path: JSON文件路径
        
        Returns:
            边界框列表
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件失败 {json_path}: {e}")
            return []
        
        # 获取图像尺寸
        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')
        
        if not img_width or not img_height:
            logger.error(f"JSON文件缺少图像尺寸信息: {json_path}")
            return []
        
        bboxes = []
        shapes = data.get('shapes', [])
        
        for shape in shapes:
            try:
                label = shape.get('label', 'unknown')
                shape_type = shape.get('shape_type', 'polygon')
                points = shape.get('points', [])
                
                if not points:
                    logger.warning(f"跳过空的标注: {label}")
                    continue
                
                # 根据标注类型处理
                if shape_type == 'rectangle':
                    x_min, y_min, x_max, y_max = self.rectangle_to_bbox(points)
                elif shape_type == 'polygon':
                    x_min, y_min, x_max, y_max = self.polygon_to_bbox(points)
                else:
                    logger.warning(f"不支持的标注类型: {shape_type}，尝试按多边形处理")
                    x_min, y_min, x_max, y_max = self.polygon_to_bbox(points)
                
                # 检查边界框有效性
                if x_max <= x_min or y_max <= y_min:
                    logger.warning(f"无效的边界框: {label} - ({x_min}, {y_min}, {x_max}, {y_max})")
                    continue
                
                # 转换为YOLO格式
                x_center, y_center, width, height = self.bbox_to_yolo_format(
                    x_min, y_min, x_max, y_max, img_width, img_height
                )
                
                # 获取类别ID
                class_id = self.get_class_id(label)
                
                bbox = BoundingBox(
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    class_id=class_id,
                    class_name=label
                )
                
                bboxes.append(bbox)

            except UnknownLabelError:
                raise
            except Exception as e:
                logger.error(f"处理标注时出错: {e}")
                continue

        return bboxes
    
    def convert_json_for_segment(self, json_path: str) -> List[str]:
        """
        分割：将JSON转换为多边形分割txt标注
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件失败 {json_path}: {e}")
            return []
        
        # 获取图像尺寸
        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')
        
        if not img_width or not img_height:
            logger.error(f"JSON文件缺少图像尺寸信息: {json_path}")
            return []
        
        annotations = []
        shapes = data.get('shapes', [])
        
        for shape in shapes:
            try:
                label = shape.get('label', 'unknown')
                shape_type = shape.get('shape_type', '')
                points = shape.get('points', [])
                
                # 只处理多边形
                if shape_type == 'polygon' and len(points) >= 3:
                    # 将像素坐标转换为归一化坐标
                    normalized_points = []
                    for point in points:
                        x, y = point
                        # 归一化到[0,1]范围
                        norm_x = x / img_width
                        norm_y = y / img_height
                        normalized_points.extend([norm_x, norm_y])
                    
                    # YOLO分割格式: class_id x1 y1 x2 y2 ... xn yn
                    class_id = self.get_class_id(label)
                    annotation_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
                    annotations.append(annotation_line)
                    
                elif shape_type == 'rectangle':
                    logger.warning(f"分割模式下跳过矩形标注: {label}")
                else:
                    logger.warning(f"分割模式下跳过不支持的标注类型: {shape_type}")

            except UnknownLabelError:
                raise
            except Exception as e:
                logger.error(f"处理分割标注时出错: {e}")
                continue

        return annotations
    
    def convert_json_for_pose(self, json_path: str) -> List[PoseAnnotation]:
        """
        Pose 模式：将 JSON 转换为 (矩形框 + K 个关键点) 列表。

        逻辑：
        - 矩形 shape 视为目标实例，决定 class_id 与 bbox
        - 点 shape 通过坐标"是否落在矩形内"归属到矩形
        - 同一矩形内同名 keypoint 多次出现时取第一个
        - 缺失的 keypoint 槽位填 (0, 0, 0)；存在则填 (x_norm, y_norm, 2)
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件失败 {json_path}: {e}")
            return []

        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')
        if not img_width or not img_height:
            logger.error(f"JSON文件缺少图像尺寸信息: {json_path}")
            return []

        if self.kpt_num <= 0:
            logger.error("pose 转换缺少 kpt_num/kpt_labels 配置")
            return []

        shapes = data.get('shapes', [])
        assignment = assign_points_to_rectangles(shapes)
        if assignment.unassigned_points:
            for point in assignment.unassigned_points[:10]:
                logger.warning(
                    f"point '{point.label}' ({point.x},{point.y}) 未落在任何矩形内，已丢弃"
                )
        if assignment.duplicate_points:
            for point in assignment.duplicate_points[:10]:
                logger.warning(
                    f"point '{point.label}' 在同一矩形内重复出现，已保留首个点: ({point.x},{point.y})"
                )

        if not assignment.rectangles:
            return []

        annotations: List[PoseAnnotation] = []
        for rect, kp_dict in zip(assignment.rectangles, assignment.keypoints_by_rect):
            x_center, y_center, width, height = self.bbox_to_yolo_format(
                rect.x_min, rect.y_min, rect.x_max, rect.y_max, img_width, img_height
            )
            class_id = self.get_class_id(rect.label)

            keypoints: List[Tuple[float, float, int]] = []
            for slot_idx in range(self.kpt_num):
                # 关键点 label 顺序由 self.kpt_labels 决定
                if self.kpt_labels:
                    kp_label = self.kpt_labels[slot_idx]
                    coord = kp_dict.get(kp_label)
                else:
                    # 没有给定 kpt_labels 时按 dict 插入顺序的前 K 个填入
                    coord = list(kp_dict.values())[slot_idx] if slot_idx < len(kp_dict) else None

                if coord is None:
                    keypoints.append((0.0, 0.0, 0))
                else:
                    nx = max(0.0, min(1.0, coord[0] / img_width))
                    ny = max(0.0, min(1.0, coord[1] / img_height))
                    keypoints.append((nx, ny, 2))

            annotations.append(PoseAnnotation(
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                class_id=class_id,
                class_name=rect.label,
                keypoints=keypoints,
            ))

        return annotations

    def save_detect_txt(self, bboxes: List[BoundingBox], output_path: str) -> bool:
        """
        保存检测模式的YOLO格式TXT文件
        
        Args:
            bboxes: 边界框列表
            output_path: 输出文件路径
        
        Returns:
            是否保存成功
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for bbox in bboxes:
                    # YOLO检测格式: class_id x_center y_center width height
                    line = f"{bbox.class_id} {bbox.x_center:.6f} {bbox.y_center:.6f} {bbox.width:.6f} {bbox.height:.6f}\n"
                    f.write(line)
            return True
        except Exception as e:
            logger.error(f"保存检测TXT文件失败 {output_path}: {e}")
            return False
    
    def save_segment_txt(self, annotations: List[str], output_path: str) -> bool:
        """
        保存分割模式的TXT文件

        Args:
            annotations: 标注行列表
            output_path: 输出文件路径

        Returns:
            是否保存成功
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for annotation in annotations:
                    f.write(annotation + '\n')
            return True
        except Exception as e:
            logger.error(f"保存分割TXT文件失败 {output_path}: {e}")
            return False

    def save_pose_txt(self, annotations: List[PoseAnnotation], output_path: str) -> bool:
        """
        保存 pose 模式 YOLO 格式 TXT。
        每行格式：class_id cx cy w h x1 y1 v1 x2 y2 v2 ... xK yK vK
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for ann in annotations:
                    parts = [
                        str(ann.class_id),
                        f"{ann.x_center:.6f}",
                        f"{ann.y_center:.6f}",
                        f"{ann.width:.6f}",
                        f"{ann.height:.6f}",
                    ]
                    for kx, ky, kv in ann.keypoints:
                        parts.append(f"{kx:.6f}")
                        parts.append(f"{ky:.6f}")
                        parts.append(str(int(kv)))
                    f.write(" ".join(parts) + "\n")
            return True
        except Exception as e:
            logger.error(f"保存 pose TXT 文件失败 {output_path}: {e}")
            return False
    
    def save_class_mapping(self) -> bool:
        """
        保存类别映射文件到labels目录
        
        Returns:
            是否保存成功
        """
        try:
            self.labels_dir.mkdir(parents=True, exist_ok=True)
            classes_file = self.original_train_data_address / 'classes.txt'
            with open(classes_file, 'w', encoding='utf-8') as f:
                sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
                for class_name, class_id in sorted_classes:
                    f.write(f"{class_name}\n")
            
            mapping_file = self.original_train_data_address / 'class_mapping.json'
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.class_mapping, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"类别映射已保存到: {classes_file} 和 {mapping_file}")
            return True
        except Exception as e:
            logger.error(f"保存类别映射失败: {e}")
            return False
    
    def _convert_single_file(self, json_file: Path) -> Tuple[bool, str]:
        """
        转换单个JSON文件（用于并行处理）
        
        Args:
            json_file: JSON文件路径
        
        Returns:
            (是否成功, 文件名)
        """
        try:
            output_filename = json_file.stem + '.txt'
            output_path = self.labels_dir / output_filename
            
            if self.duty_type == DutyType.DETECT.value:
                bboxes = self.convert_json_for_detect(str(json_file))
                if self.save_detect_txt(bboxes, str(output_path)):
                    logger.debug(f"检测转换成功: {json_file.name} -> {output_filename} ({len(bboxes)} 个标注)")
                    return (True, json_file.name)
                else:
                    logger.error(f"检测转换失败: {json_file.name}")
                    return (False, json_file.name)
            
            elif self.duty_type == DutyType.SEGMENT.value:
                annotations = self.convert_json_for_segment(str(json_file))
                if self.save_segment_txt(annotations, str(output_path)):
                    logger.debug(f"分割转换成功: {json_file.name} -> {output_filename} ({len(annotations)} 个标注)")
                    return (True, json_file.name)
                else:
                    logger.error(f"分割转换失败: {json_file.name}")
                    return (False, json_file.name)

            elif self.duty_type == DutyType.POSE.value:
                pose_anns = self.convert_json_for_pose(str(json_file))
                if self.save_pose_txt(pose_anns, str(output_path)):
                    logger.debug(f"pose 转换成功: {json_file.name} -> {output_filename} ({len(pose_anns)} 个实例)")
                    return (True, json_file.name)
                else:
                    logger.error(f"pose 转换失败: {json_file.name}")
                    return (False, json_file.name)

            return (False, json_file.name)
        except UnknownLabelError:
            # 预定义标签模式下出现未知标签：让其向上传播，由 convert() 终止整个转换
            raise
        except Exception as e:
            logger.error(f"转换文件时出错 {json_file.name}: {e}")
            return (False, json_file.name)
    
    def convert(self) -> Dict[str, int]:
        """
        执行转换任务（并行处理）
        
        Returns:
            统计信息字典
        """
        # 创建输出目录
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有JSON文件
        json_files = [json_file for json_file in sorted(self.json_dir.glob('*.json')) if json_file.is_file()]
        
        if not json_files:
            logger.warning(f"在JSON目录中未找到JSON文件: {self.json_dir}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        logger.info(f"找到 {len(json_files)} 个JSON文件，开始并行转换...")
        logger.debug(f"转换模式: {self.duty_type}")
        
        success_count = 0
        failed_count = 0
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self._convert_single_file, json_file): json_file
                      for json_file in json_files}
            for future in as_completed(futures):
                try:
                    success, filename = future.result()
                except UnknownLabelError as e:
                    # 预定义标签模式下出现未知标签：直接终止整个转换，
                    # 避免生成 class_id >= nc 的 TXT 让训练前检查失败。
                    for pending in futures:
                        pending.cancel()
                    logger.error(
                        f"转换终止：{e}。预定义标签: {self.predefined_labels}"
                    )
                    raise
                if success:
                    success_count += 1
                else:
                    failed_count += 1
        
        # 保存类别映射
        self.save_class_mapping()
        
        # 输出统计信息
        total_count = len(json_files)
        logger.info(f"转换完成: {success_count}/{total_count} 个文件转换成功")
        
        if self.class_mapping:
            logger.debug("发现的类别:")
            for class_name, class_id in sorted(self.class_mapping.items(), key=lambda x: x[1]):
                logger.debug(f"  {class_id}: {class_name}")
        
        # 文件复制由调用方处理，这里只返回本地路径
        
        return {
            'success': success_count,
            'failed': failed_count,
            'total': total_count
        }


def main():
    """主函数 - 命令行接口"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='JSON到TXT转换器 - 支持检测/分割/pose 任务')
    parser.add_argument('--duty_type', required=True, choices=['detect', 'segment', 'pose'],
                      help='任务类型: detect(检测) / segment(分割) / pose(关键点)')
    parser.add_argument('--original_train_data_address', required=True,
                      help='原始训练数据地址')
    parser.add_argument('--labels', nargs='+', default=None,
                      help='预定义类别标签列表，按顺序分配类别ID。例如: --labels duckweed ship')
    parser.add_argument('--kpt_labels', nargs='+', default=None,
                      help='pose 模式下关键点标签的规范顺序，长度即 K')
    parser.add_argument('--kpt_num', type=int, default=None,
                      help='pose 模式下关键点数量 K（与 --kpt_labels 二选一）')

    args = parser.parse_args()

    try:
        # 创建转换器
        converter = UnifiedJsonConverter(
            duty_type=args.duty_type,
            original_train_data_address=args.original_train_data_address,
            predefined_labels=args.labels,
            kpt_labels=args.kpt_labels,
            kpt_num=args.kpt_num,
        )

        # 执行转换
        result = converter.convert()

        # 输出结果
        if result['success'] > 0:
            print(f"\n✅ 转换成功: {result['success']}/{result['total']} 个文件")
            print(f"输出目录: {converter.labels_dir}")
            if args.labels:
                print(f"使用预定义标签顺序: {args.labels}")
            if args.duty_type == 'pose':
                print(f"关键点数量(K): {converter.kpt_num}, 关键点顺序: {converter.kpt_labels}")
        else:
            print(f"\n❌ 转换失败: 没有成功转换任何文件")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
