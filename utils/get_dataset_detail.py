#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集详情分析脚本
分析original_train_data文件夹中的图片和标注文件，生成统计信息
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .annotation_paths import resolve_annotation_dir
except ImportError:
    from annotation_paths import resolve_annotation_dir

logger = logging.getLogger(__name__)

# 并行处理的最大线程数
MAX_WORKERS = 16

class DatasetDetailAnalyzer:
    def __init__(self, original_train_data_address):
        self.original_train_data_address = Path(original_train_data_address)
        self.images_dir = self.original_train_data_address / 'images'
        self.json_dir = resolve_annotation_dir(self.original_train_data_address)
        
        # 支持的图片格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # 验证路径存在性
        self._validate_paths()
    
    def _validate_paths(self):
        """验证路径存在性"""
        if not self.original_train_data_address.exists():
            raise FileNotFoundError(f"路径不存在: {self.original_train_data_address}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images文件夹不存在: {self.images_dir}")
        
        if not self.json_dir.exists():
            raise FileNotFoundError(f"json/jsons文件夹不存在: {self.original_train_data_address}")
    
    def get_image_files(self):
        """获取所有图片文件"""
        image_files = []
        for file_path in self.images_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)
        return image_files
    
    def get_json_files(self):
        """获取所有JSON标注文件"""
        json_files = []
        for file_path in self.json_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.json':
                json_files.append(file_path)
        return json_files
    
    def get_sample_num(self):
        """计算图片数量"""
        image_files = self.get_image_files()
        sample_num = len(image_files)
        logger.info(f"发现图片文件: {sample_num} 个")
        return sample_num
    
    def get_annotation_num(self):
        """计算标注文件数量"""
        json_files = self.get_json_files()
        annotation_num = len(json_files)
        logger.info(f"发现标注文件: {annotation_num} 个")
        return annotation_num
    
    def check_correspondence(self):
        """检查JSON文件名和图片文件名的对应关系"""
        image_files = self.get_image_files()
        json_files = self.get_json_files()
        
        # 获取文件名（不含扩展名）
        image_names = {file_path.stem for file_path in image_files}
        json_names = {file_path.stem for file_path in json_files}
        
        # 检查每个JSON是否有对应的图片
        missing_images = json_names - image_names
        missing_annotations = image_names - json_names
        
        correspondence_info = {
            'json_without_image': list(missing_images),
            'image_without_json': list(missing_annotations),
            'perfect_match': len(missing_images) == 0 and len(missing_annotations) == 0
        }
        
        if correspondence_info['perfect_match']:
            logger.info("✅ 所有JSON文件都有对应的图片文件")
        else:
            if missing_images:
                logger.warning(f"⚠️  没有对应图片的JSON文件: {missing_images}")
            if missing_annotations:
                logger.warning(f"⚠️  没有对应标注的图片文件: {missing_annotations}")
        
        return correspondence_info
    
    def extract_labels_from_json(self, json_file):
        """从单个JSON文件中提取标签"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            labels = []
            if 'shapes' in data:
                for shape in data['shapes']:
                    if 'label' in shape and shape['label']:
                        labels.append(shape['label'])
            
            return labels
        
        except Exception as e:
            logger.error(f"读取JSON文件失败 {json_file}: {str(e)}")
            return []
    
    def extract_shape_types_from_json(self, json_file):
        """从单个JSON文件中提取形状类型"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            shape_types = []
            if 'shapes' in data:
                for shape in data['shapes']:
                    if 'shape_type' in shape and shape['shape_type']:
                        shape_types.append(shape['shape_type'])
            
            return shape_types
        
        except Exception as e:
            logger.error(f"读取JSON文件失败 {json_file}: {str(e)}")
            return []
    
    def get_labels_and_statistics(self):
        """获取所有标签和统计信息（并行处理）"""
        json_files = self.get_json_files()
        all_labels = []
        
        logger.info("开始提取标签信息（并行处理）...")
        
        # 使用线程池并行读取JSON文件
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.extract_labels_from_json, json_file): json_file 
                      for json_file in json_files}
            for future in as_completed(futures):
                labels = future.result()
                all_labels.extend(labels)
        
        # 统计标签
        label_counter = Counter(all_labels)
        unique_labels = list(label_counter.keys())
        label_num = len(unique_labels)
        
        # 计算标注框总数（tag_sum）
        tag_sum = len(all_labels)
        
        # 计算各标签占比（tag_percentage）
        tag_percentage = {}
        if tag_sum > 0:
            for label, count in label_counter.items():
                percentage = round((count / tag_sum) * 100, 2)
                tag_percentage[label] = f"{percentage}%"
        
        # 格式化结果
        labels_result = json.dumps(unique_labels, ensure_ascii=False)
        tag_num = dict(label_counter)
        
        logger.info(f"标签统计完成:")
        logger.info(f"  - 唯一标签数量: {label_num}")
        logger.info(f"  - 标签列表: {unique_labels}")
        logger.info(f"  - 各标签数量: {tag_num}")
        logger.info(f"  - 标注框总数 (tag_sum): {tag_sum}")
        logger.info(f"  - 各标签占比 (tag_percentage): {tag_percentage}")
        
        return {
            'labels': labels_result,
            'label_num': label_num,
            'tag_num': tag_num,
            'unique_labels': unique_labels,
            'tag_sum': tag_sum,
            'tag_percentage': tag_percentage
        }
    
    def detect_draw_type(self):
        """检测数据集的标注类型（矩形、多边形或混合）- 并行处理"""
        json_files = self.get_json_files()
        all_shape_types = []
        
        logger.info("开始分析标注类型（并行处理）...")
        
        # 使用线程池并行读取JSON文件
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.extract_shape_types_from_json, json_file): json_file 
                      for json_file in json_files}
            for future in as_completed(futures):
                shape_types = future.result()
                all_shape_types.extend(shape_types)
        
        if not all_shape_types:
            logger.warning("未找到任何标注形状类型")
            return "rectangle"  # 默认返回矩形
        
        # 统计形状类型
        shape_type_counter = Counter(all_shape_types)
        unique_shape_types = set(all_shape_types)
        
        # 判断标注类型
        if 'polygon' in unique_shape_types and 'rectangle' in unique_shape_types:
            # 既有多边形又有矩形
            draw_type = "mix"
            logger.info(f"检测到混合标注类型，draw_type: {draw_type}")
        elif 'polygon' in unique_shape_types:
            # 只有多边形
            draw_type = "polygon"
            logger.info(f"检测到多边形标注，draw_type: {draw_type}")
        else:
            # 只有矩形或其他类型（默认为矩形）
            draw_type = "rectangle"
            logger.info(f"检测到矩形标注，draw_type: {draw_type}")
        
        logger.info(f"形状类型统计: {dict(shape_type_counter)}")
        return draw_type
    
    def analyze_dataset(self):
        """完整的数据集分析"""
        logger.info(f"开始分析数据集: {self.original_train_data_address}")
        
        # 获取基本统计信息
        sample_num = self.get_sample_num()
        annotation_num = self.get_annotation_num()
        
        # 检查文件对应关系
        correspondence_info = self.check_correspondence()
        
        # 获取标签统计
        label_stats = self.get_labels_and_statistics()
        
        # 检测标注类型
        draw_type = self.detect_draw_type()
        
        # 汇总结果
        result = {
            'dataset_path': str(self.original_train_data_address),
            'sample_num': sample_num,
            'annotation_num': annotation_num,
            'labels': label_stats['labels'],
            'label_num': label_stats['label_num'],
            'tag_num': label_stats['tag_num'],
            'tag_sum': label_stats['tag_sum'],
            'tag_percentage': label_stats['tag_percentage'],
            'draw_type': draw_type,
            'correspondence_check': correspondence_info
        }
        
        # 保存分析结果到JSON文件
        result_file = self.original_train_data_address / 'dataset_analysis.json'
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存到: {result_file}")
            # 文件复制由调用方处理
        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}")
        
        return result


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='数据集详情分析工具')
    parser.add_argument(
        '--original_train_data_address',
        required=True,
        help='original_train_data文件夹的本地路径'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建分析器实例
        analyzer = DatasetDetailAnalyzer(args.original_train_data_address)
        
        # 执行分析
        result = analyzer.analyze_dataset()
        
        # 输出结果
        print("\n" + "="*50)
        print("数据集分析结果")
        print("="*50)
        print(f"数据集路径: {result['dataset_path']}")
        print(f"图片数量 (sample_num): {result['sample_num']}")
        print(f"标注文件数量 (annotation_num): {result['annotation_num']}")
        print(f"标签类别数量 (label_num): {result['label_num']}")
        print(f"标签列表 (labels): {result['labels']}")
        print(f"各标签数量 (tag_num): {json.dumps(result['tag_num'], ensure_ascii=False, indent=2)}")
        print(f"标注类型 (draw_type): {result['draw_type']}")
        
        if not result['correspondence_check']['perfect_match']:
            print("\n⚠️  文件对应关系检查:")
            if result['correspondence_check']['json_without_image']:
                print(f"  - 没有对应图片的JSON文件: {result['correspondence_check']['json_without_image']}")
            if result['correspondence_check']['image_without_json']:
                print(f"  - 没有对应标注的图片文件: {result['correspondence_check']['image_without_json']}")
        else:
            print("\n✅ 所有JSON文件都有对应的图片文件")
        
        print("="*50)
        
        # 返回结构化数据供其他程序使用
        return result
        
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        print(f"❌ 分析失败: {str(e)}")
        return None


if __name__ == '__main__':
    main()
