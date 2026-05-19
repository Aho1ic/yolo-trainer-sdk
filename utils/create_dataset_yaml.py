#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成YOLO数据集配置文件(dataset.yaml)的工具类
仅支持程序化调用
"""

import json
import sys
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# 尝试导入MinIO客户端
try:
    from minio_client import get_minio_manager
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False



def parse_labels(labels_str):
    try:
        labels = json.loads(labels_str)
        if not isinstance(labels, list):
            raise ValueError("标签必须是列表格式")
        return labels
    except json.JSONDecodeError:
        labels = [label.strip().strip('"\'') for label in labels_str.strip('[]').split(',')]
        return labels


def validate_arguments(args):
    # 验证数据集路径
    dataset_path = Path(args.dataset_address)
    if not dataset_path.exists():
        return False
    
    # 设置默认输出路径
    if not args.output:
        args.output = str(dataset_path / "dataset.yaml")
    else:
        # 如果指定了输出路径，转换为绝对路径
        output_path = Path(args.output)
        if not output_path.is_absolute():
            args.output = str(Path.cwd() / output_path)
    
    # 解析标签
    try:
        labels = parse_labels(args.labels)
        args.parsed_labels = labels
    except Exception as e:
        return False
    
    # 验证标签数量
    if len(args.parsed_labels) != args.label_num:
        return False
    
    return True


def generate_yaml(args):
    """生成YAML配置文件（静默版本）"""
    dataset_path = Path(args.dataset_address)
    
    # 构建训练和验证路径
    train_path = dataset_path / args.train_subdir
    val_path = dataset_path / args.val_subdir
    
    # 生成YAML内容
    yaml_content = f"""train: {train_path}
val: {val_path}
nc: {args.label_num}
names: {args.parsed_labels}
"""
    
    # 写入文件
    output_file = Path(args.output)
    try:
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        # 获取绝对路径
        absolute_path = output_file.absolute()
        
        # 返回绝对路径（供其他代码使用）
        return str(absolute_path)
        
    except Exception as e:
        return None


class DatasetYamlGenerator:
    """数据集YAML配置文件生成器类"""
    
    def __init__(self, train_subdir="datasets/train/images", val_subdir="datasets/valid/images"):
        """
        初始化生成器
        
        Args:
            train_subdir: 训练集子目录
            val_subdir: 验证集子目录
        """
        self.train_subdir = train_subdir
        self.val_subdir = val_subdir
    
    def generate(self, dataset_address, labels, label_num, output_path=None):
        """
        生成YAML配置文件
        
        Args:
            dataset_address: 数据集根目录路径
            labels: 标签列表
            label_num: 标签数量
            output_path: 输出路径，如果为None则默认在dataset_address下
        
        Returns:
            str: 生成的YAML文件的绝对路径，失败时返回None
        """
        try:
            # 创建类似命令行参数的对象
            class Args:
                def __init__(self):
                    self.dataset_address = dataset_address
                    self.labels = str(labels) if isinstance(labels, list) else labels
                    self.label_num = label_num
                    self.output = output_path or ""
                    self.train_subdir = train_subdir
                    self.val_subdir = val_subdir
            
            args = Args()
            args.train_subdir = self.train_subdir
            args.val_subdir = self.val_subdir
            
            # 验证参数
            if not validate_arguments(args):
                return None
            
            # 生成YAML文件
            yaml_path = generate_yaml(args)
            return yaml_path
            
        except Exception as e:
            return None
    
    def generate_silent(self, dataset_address, labels, label_num, output_path=None):
        """
        静默生成YAML配置文件（不打印信息）
        
        Args:
            dataset_address: 数据集根目录路径
            labels: 标签列表
            label_num: 标签数量
            output_path: 输出路径，如果为None则默认在dataset_address下
        
        Returns:
            dict: 包含结果信息的字典 {'success': bool, 'yaml_path': str, 'error': str}
        """
        try:
            dataset_path = Path(dataset_address)
            # 如果路径不存在，创建它（对于MinIO数据集，我们只需要生成YAML）
            if not dataset_path.exists():
                logger.warning(f'数据集路径不存在，创建目录: {dataset_path}')
                dataset_path.mkdir(parents=True, exist_ok=True)
            
            # 设置输出路径
            if not output_path:
                output_path = str(dataset_path / "dataset.yaml")
            else:
                output_path_obj = Path(output_path)
                if not output_path_obj.is_absolute():
                    output_path = str(Path.cwd() / output_path_obj)
            
            # 解析标签
            if isinstance(labels, str):
                try:
                    parsed_labels = json.loads(labels)
                except json.JSONDecodeError:
                    parsed_labels = [label.strip().strip('"\'') for label in labels.strip('[]').split(',')]
            else:
                parsed_labels = labels
            
            # 验证标签数量
            if len(parsed_labels) != label_num:
                return {'success': False, 'yaml_path': None, 'error': f'标签数量不匹配 - 提供了{len(parsed_labels)}个标签，但指定数量为{label_num}'}
            
            # 构建训练和验证路径
            train_path = dataset_path / self.train_subdir
            val_path = dataset_path / self.val_subdir
            
            # 生成YAML内容
            yaml_content = f"""train: {train_path}
val: {val_path}
nc: {label_num}
names: {parsed_labels}
"""
            
            # 写入文件
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            absolute_path = str(output_file.absolute())
            
            # MinIO上传由调用方（train_api.py）处理
            
            return {'success': True, 'yaml_path': absolute_path, 'error': None}
            
        except Exception as e:
            return {'success': False, 'yaml_path': None, 'error': str(e)}


def create_dataset_yaml(dataset_address, labels, label_num, output_path=None, train_subdir="datasets/train/images", val_subdir="datasets/valid/images"):
    """
    程序化调用接口，用于其他脚本调用（保持向后兼容）
    
    Args:
        dataset_address: 数据集根目录路径
        labels: 标签列表
        label_num: 标签数量
        output_path: 输出路径，如果为None则默认在dataset_address下
        train_subdir: 训练集子目录
        val_subdir: 验证集子目录
    
    Returns:
        str: 生成的YAML文件的绝对路径，失败时返回None
    """
    generator = DatasetYamlGenerator(train_subdir, val_subdir)
    return generator.generate(dataset_address, labels, label_num, output_path)