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
import yaml

logger = logging.getLogger(__name__)


class _FlowList(list):
    """YAML 中需要保持行内表示的列表。"""


class _DatasetYamlDumper(yaml.SafeDumper):
    pass


def _represent_flow_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


_DatasetYamlDumper.add_representer(_FlowList, _represent_flow_list)


def _dump_dataset_yaml(train_path, val_path, label_num, names, duty_type=None, kpt_num=None):
    yaml_data = {
        'train': str(train_path),
        'val': str(val_path),
        'nc': int(label_num),
        'names': list(names),
    }
    if duty_type == 'pose' and kpt_num:
        yaml_data['kpt_shape'] = _FlowList([int(kpt_num), 3])
    return yaml.dump(
        yaml_data,
        Dumper=_DatasetYamlDumper,
        sort_keys=False,
        allow_unicode=True,
    )


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

    yaml_content = _dump_dataset_yaml(
        train_path,
        val_path,
        args.label_num,
        args.parsed_labels,
        duty_type=getattr(args, 'duty_type', None),
        kpt_num=getattr(args, 'kpt_num', None),
    )

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

    def generate(self, dataset_address, labels, label_num, output_path=None,
                 duty_type=None, kpt_num=None):
        """
        生成YAML配置文件

        Args:
            dataset_address: 数据集根目录路径
            labels: 标签列表
            label_num: 标签数量
            output_path: 输出路径，如果为None则默认在dataset_address下
            duty_type: 任务类型（detect/segment/pose），pose 时需要 kpt_num
            kpt_num: pose 模式下的关键点数量 K

        Returns:
            str: 生成的YAML文件的绝对路径，失败时返回None
        """
        try:
            # 创建类似命令行参数的对象
            class Args:
                pass

            args = Args()
            args.dataset_address = dataset_address
            args.labels = str(labels) if isinstance(labels, list) else labels
            args.label_num = label_num
            args.output = output_path or ""
            args.train_subdir = self.train_subdir
            args.val_subdir = self.val_subdir
            args.duty_type = duty_type
            args.kpt_num = kpt_num

            # 验证参数
            if not validate_arguments(args):
                return None

            # 生成YAML文件
            yaml_path = generate_yaml(args)
            return yaml_path

        except Exception as e:
            return None

    def generate_silent(self, dataset_address, labels, label_num, output_path=None,
                        duty_type=None, kpt_num=None):
        """
        静默生成YAML配置文件（不打印信息）

        Args:
            dataset_address: 数据集根目录路径
            labels: 标签列表
            label_num: 标签数量
            output_path: 输出路径，如果为None则默认在dataset_address下
            duty_type: 任务类型（detect/segment/pose），pose 时需要 kpt_num
            kpt_num: pose 模式下的关键点数量 K

        Returns:
            dict: 包含结果信息的字典 {'success': bool, 'yaml_path': str, 'error': str}
        """
        try:
            dataset_path = Path(dataset_address)
            # 如果路径不存在，创建它（仅用于生成 YAML）
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

            # pose 模式校验
            if duty_type == 'pose':
                if not kpt_num or int(kpt_num) <= 0:
                    return {'success': False, 'yaml_path': None, 'error': 'pose 模式必须提供正整数的 kpt_num'}

            # 构建训练和验证路径
            train_path = dataset_path / self.train_subdir
            val_path = dataset_path / self.val_subdir

            yaml_content = _dump_dataset_yaml(
                train_path,
                val_path,
                label_num,
                parsed_labels,
                duty_type=duty_type,
                kpt_num=kpt_num,
            )

            # 写入文件
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(yaml_content)

            absolute_path = str(output_file.absolute())

            # 文件复制由调用方（train_api.py）处理

            return {'success': True, 'yaml_path': absolute_path, 'error': None}

        except Exception as e:
            return {'success': False, 'yaml_path': None, 'error': str(e)}


def create_dataset_yaml(dataset_address, labels, label_num, output_path=None,
                        train_subdir="datasets/train/images",
                        val_subdir="datasets/valid/images",
                        duty_type=None, kpt_num=None):
    """
    程序化调用接口，用于其他脚本调用（保持向后兼容）

    Args:
        dataset_address: 数据集根目录路径
        labels: 标签列表
        label_num: 标签数量
        output_path: 输出路径，如果为None则默认在dataset_address下
        train_subdir: 训练集子目录
        val_subdir: 验证集子目录
        duty_type: 任务类型（detect/segment/pose）
        kpt_num: pose 模式的关键点数量 K

    Returns:
        str: 生成的YAML文件的绝对路径，失败时返回None
    """
    generator = DatasetYamlGenerator(train_subdir, val_subdir)
    return generator.generate(dataset_address, labels, label_num, output_path,
                              duty_type=duty_type, kpt_num=kpt_num)
