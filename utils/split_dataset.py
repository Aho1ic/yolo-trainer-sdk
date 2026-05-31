#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON文件转换为YOLO格式的TXT
支持检测(detect)和分割(segment)两种模式
python split_dataset.py --dataset_id 250309300626038784 --original_train_data_address "/Users/macbook/Downloads/同步空间/trainer/original_dataset/250309300626038784/original_train_data" --train_val_ratio "8:2"
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
import threading


class DatasetSplitter:
    def __init__(self, original_train_data_address, train_val_ratio, dataset_id):
        self.original_train_data_address = Path(original_train_data_address)
        self.images_dir = self.original_train_data_address / "images"
        self.labels_dir = self.original_train_data_address / "labels"
        self.train_val_ratio = str(train_val_ratio)
        self.dataset_id = str(dataset_id)

        base_path = Path(__file__).parent.parent
        self.train_data_root = base_path / "train_data"
        self.output_root = self.train_data_root / self.dataset_id
        self.datasets_root = self.output_root / "datasets"

    def _validate_paths(self):
        if not self.original_train_data_address.exists():
            raise FileNotFoundError(f"原始训练数据路径不存在: {self.original_train_data_address}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {self.labels_dir}")

    def _parse_ratio(self):
        parts = self.train_val_ratio.split(":")
        if len(parts) != 2:
            raise ValueError(f"train_val_ratio 格式错误，应为 '8:2' 这种形式: {self.train_val_ratio}")
        try:
            train_part = int(parts[0])
            val_part = int(parts[1])
        except ValueError:
            raise ValueError(f"train_val_ratio 必须是整数比例: {self.train_val_ratio}")
        if train_part <= 0 or val_part <= 0:
            raise ValueError(f"train_val_ratio 中的值必须大于 0: {self.train_val_ratio}")
        return train_part, val_part

    def _collect_images(self):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = []
        if not self.images_dir.exists():
            return images
        for p in self.images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                images.append(p)
        return images

    def _prepare_dirs(self):
        if self.datasets_root.exists():
            for _ in self.datasets_root.iterdir():
                raise FileExistsError(f"目标数据集目录已存在且非空: {self.datasets_root}")

        train_images = self.datasets_root / "train" / "images"
        train_labels = self.datasets_root / "train" / "labels"
        val_images = self.datasets_root / "valid" / "images"
        val_labels = self.datasets_root / "valid" / "labels"

        train_images.mkdir(parents=True, exist_ok=True)
        train_labels.mkdir(parents=True, exist_ok=True)
        val_images.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)

        return train_images, train_labels, val_images, val_labels

    def _copy_pair(self, image_path, dst_images_dir, dst_labels_dir):
        dst_image = dst_images_dir / image_path.name
        shutil.copy2(image_path, dst_image)

        label_path = self.labels_dir / (image_path.stem + ".txt")
        if label_path.exists():
            dst_label = dst_labels_dir / label_path.name
            shutil.copy2(label_path, dst_label)

    def split(self):
        self._validate_paths()
        train_part, val_part = self._parse_ratio()

        images = self._collect_images()
        if not images:
            raise ValueError("在 images 目录中未找到任何图片文件")

        random.shuffle(images)

        total = len(images)
        total_ratio = train_part + val_part
        train_count = int(total * train_part / total_ratio)

        if train_count <= 0:
            train_count = 1
        if train_count >= total:
            train_count = total - 1

        train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = self._prepare_dirs()

        for idx, img in enumerate(images):
            if idx < train_count:
                self._copy_pair(img, train_images_dir, train_labels_dir)
            else:
                self._copy_pair(img, val_images_dir, val_labels_dir)
        
        # 数据复制由 train_api.py 处理
        # 只返回本地路径，让调用方决定如何处理
        return str(self.output_root.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_train_data_address", required=True)
    parser.add_argument("--train_val_ratio", required=True)
    parser.add_argument("--dataset_id", required=True)

    args = parser.parse_args()

    try:
        splitter = DatasetSplitter(
            original_train_data_address=args.original_train_data_address,
            train_val_ratio=args.train_val_ratio,
            dataset_id=args.dataset_id,
        )
        output_path = splitter.split()
        print(output_path)
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

