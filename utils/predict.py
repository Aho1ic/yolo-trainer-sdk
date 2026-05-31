#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO模型推理预测类
支持目标检测、实例分割和关键点检测
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO
import cv2
import numpy as np


class Predict:
    """YOLO模型推理类，支持目标检测、实例分割和关键点检测"""
    
    def __init__(
        self,
        dataset_address: str,
        model_address: str,
        duty_type: str = "detect",
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = None
    ):
        """
        初始化Predict类
        
        Args:
            dataset_address: 待预测图片的文件夹路径
            model_address: YOLO模型路径
            duty_type: 任务类型，"detect"表示目标检测，"segment"表示实例分割，"pose"表示关键点检测
            img_size: 输入图像尺寸，对应YOLO的imgsz参数
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            device: 推理设备，None时自动选择
        """
        self.dataset_address = Path(dataset_address)
        self.model_address = model_address
        self.duty_type = duty_type.lower()
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 验证参数
        self._validate_params()
        
        # 设置输出目录
        self.predicted_result_address = self._setup_output_dir()
        
        # 加载模型
        self.model = self._load_model()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"utils.predict.{Path(self.model_address).stem}")
        logger.setLevel(logging.INFO)
        
        # 添加控制台处理器
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_params(self):
        """验证输入参数"""
        # 验证数据集路径
        if not self.dataset_address.exists():
            raise ValueError(f"数据集路径不存在: {self.dataset_address}")
        
        if not self.dataset_address.is_dir():
            raise ValueError(f"数据集路径不是文件夹: {self.dataset_address}")
        
        # 验证模型路径
        if not Path(self.model_address).exists():
            raise ValueError(f"模型文件不存在: {self.model_address}")
        
        # 验证任务类型
        if self.duty_type not in ["detect", "segment", "pose"]:
            raise ValueError(f"不支持的任务类型: {self.duty_type}，仅支持'detect'、'segment'或'pose'")
        
        # 验证图像尺寸
        if self.img_size <= 0 or self.img_size % 32 != 0:
            raise ValueError(f"图像尺寸必须是32的倍数且大于0: {self.img_size}")
        
        self.logger.info(f"参数验证通过 - 任务类型: {self.duty_type}, 图像尺寸: {self.img_size}")
    
    def _setup_output_dir(self) -> Path:
        """设置输出目录"""
        # 获取dataset_address的父目录
        parent_dir = self.dataset_address.parent
        
        # 创建predicted_result文件夹路径
        output_dir = parent_dir / "predicted_result"
        
        # 如果不存在则创建
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"创建输出目录: {output_dir}")
        else:
            self.logger.info(f"使用已存在的输出目录: {output_dir}")
        
        return output_dir
    
    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        try:
            self.logger.info(f"正在加载模型: {self.model_address}")
            model = YOLO(self.model_address)
            
            # 设置设备
            if self.device:
                model.to(self.device)
            
            self.logger.info(f"模型加载成功，任务类型: {self.duty_type}")
            return model
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def _get_image_files(self) -> list:
        """获取所有图像文件"""
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # 收集所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.dataset_address.glob(f"*{ext}"))
            image_files.extend(self.dataset_address.glob(f"*{ext.upper()}"))
        
        # 排序以确保顺序一致
        image_files = sorted(image_files)
        
        if not image_files:
            raise ValueError(f"在 {self.dataset_address} 中没有找到图像文件")
        
        self.logger.info(f"找到 {len(image_files)} 个图像文件")
        return image_files
    
    def predict(self) -> Dict[str, Any]:
        """
        执行推理预测
        
        Returns:
            包含预测结果信息的字典，包括predicted_result_address
        """
        try:
            # 获取所有图像文件
            image_files = self._get_image_files()
            
            # 设置推理参数
            predict_args = {
                'imgsz': self.img_size,
                'conf': self.conf_threshold,
                'iou': self.iou_threshold,
                'save': True,  # 保存预测结果
                'save_txt': False,  # 不保存文本结果
                'save_conf': False,  # 不保存置信度
                'project': str(self.predicted_result_address),  # 输出目录
                'name': '.',  # 直接保存到project目录，不创建子目录
                'exist_ok': True,  # 如果目录存在也继续
                'verbose': False  # 减少输出
            }
            
            self.logger.info(f"开始推理，共 {len(image_files)} 张图片...")
            
            # 批量推理
            for i, img_path in enumerate(image_files, 1):
                try:
                    self.logger.info(f"正在处理 [{i}/{len(image_files)}]: {img_path.name}")
                    
                    # 执行推理
                    results = self.model.predict(
                        source=str(img_path),
                        **predict_args
                    )
                    
                    # 如果需要自定义保存逻辑，可以在这里处理results
                    # 默认情况下，YOLO会自动保存标注后的图片
                    
                except Exception as e:
                    self.logger.error(f"处理图片 {img_path.name} 时出错: {str(e)}")
                    continue
            
            self.logger.info(f"推理完成！结果保存在: {self.predicted_result_address}")
            
            # 结果文件复制由调用方（train_api.py）处理
            
            # 返回结果
            result = {
                'predicted_result_address': str(self.predicted_result_address),
                'total_images': len(image_files),
                'duty_type': self.duty_type,
                'img_size': self.img_size,
                'model_address': self.model_address
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"推理过程出错: {str(e)}")
            raise
    
    def predict_single_image(self, image_path: str) -> Any:
        """
        对单张图片进行推理
        
        Args:
            image_path: 图片路径
            
        Returns:
            YOLO推理结果
        """
        try:
            results = self.model.predict(
                source=image_path,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            return results
        except Exception as e:
            self.logger.error(f"单张图片推理失败: {str(e)}")
            raise


def main():
    """主函数，用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO模型推理预测')
    parser.add_argument('--dataset_address', type=str, required=True,
                        help='待预测图片的文件夹路径')
    parser.add_argument('--model_address', type=str, required=True,
                        help='YOLO模型路径')
    parser.add_argument('--duty_type', type=str, default='detect',
                        choices=['detect', 'segment', 'pose'],
                        help='任务类型：detect(目标检测) / segment(实例分割) / pose(关键点检测)')
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图像尺寸（对应YOLO的imgsz参数）')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                        help='IOU阈值')
    parser.add_argument('--device', type=str, default=None,
                        help='推理设备，如cuda:0或cpu')
    
    args = parser.parse_args()
    
    # 创建Predict实例
    predictor = Predict(
        dataset_address=args.dataset_address,
        model_address=args.model_address,
        duty_type=args.duty_type,
        img_size=args.img_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    
    # 执行推理
    result = predictor.predict()
    
    # 打印结果
    print("\n" + "="*60)
    print("推理完成！")
    print(f"结果保存路径: {result['predicted_result_address']}")
    print(f"处理图片数量: {result['total_images']}")
    print(f"任务类型: {result['duty_type']}")
    print(f"图像尺寸: {result['img_size']}")
    print(f"使用模型: {result['model_address']}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    main()
