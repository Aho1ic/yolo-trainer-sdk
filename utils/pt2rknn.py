#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PT模型转换工具类
用于将YOLOv8/YOLOv11训练的pt权重转换为onnx格式，以及onnx转换为rknn格式
python pt2rknn.py --model_address /path/to/model.pt --duty_type detect
"""

import os
import sys
import subprocess
import yaml
import shutil
from pathlib import Path


class PT2ONNX:
    """
    将YOLOv8/YOLOv11的pt权重转换为onnx格式
    
    使用方法:
        converter = PT2ONNX(model_address="/path/to/your/model.pt")
        onnx_path = converter.convert()
    """
    
    def __init__(self, model_address: str):
        """
        初始化PT2ONNX转换器
        
        Args:
            model_address: pt模型的完整路径
        """
        self.model_path = Path(model_address).resolve()
        
        # 验证模型文件是否存在
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        if not self.model_path.suffix == '.pt':
            raise ValueError(f"模型文件必须是.pt格式: {self.model_path}")
        
        # 获取pt2onnx目录路径
        self.base_dir = Path(__file__).parent.parent.resolve()
        self.pt2onnx_dir = self.base_dir / "pt2onnx"
        self.default_yaml_path = self.pt2onnx_dir / "ultralytics" / "cfg" / "default.yaml"
        self.exporter_path = self.pt2onnx_dir / "ultralytics" / "engine" / "exporter.py"
        
        # 验证必要文件是否存在
        if not self.pt2onnx_dir.exists():
            raise FileNotFoundError(f"pt2onnx目录不存在: {self.pt2onnx_dir}")
        
        if not self.default_yaml_path.exists():
            raise FileNotFoundError(f"default.yaml文件不存在: {self.default_yaml_path}")
        
        if not self.exporter_path.exists():
            raise FileNotFoundError(f"exporter.py文件不存在: {self.exporter_path}")
        
        # 输出onnx路径（与原模型在同一目录下）
        self.onnx_path = self.model_path.with_suffix('.onnx')
    
    def _backup_config(self) -> str:
        """
        备份原始配置文件
        
        Returns:
            备份文件路径
        """
        backup_path = str(self.default_yaml_path) + ".backup"
        shutil.copy2(self.default_yaml_path, backup_path)
        return backup_path
    
    def _restore_config(self, backup_path: str):
        """
        恢复原始配置文件
        
        Args:
            backup_path: 备份文件路径
        """
        if os.path.exists(backup_path):
            shutil.move(backup_path, self.default_yaml_path)
    
    def _modify_config(self):
        """
        修改default.yaml中的model路径
        """
        # 读取yaml文件
        with open(self.default_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 修改model路径
        config['model'] = str(self.model_path)
        
        # 确保mode为export
        config['mode'] = 'export'
        
        # 写回yaml文件
        with open(self.default_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"[PT2ONNX] 已更新配置文件，model路径设置为: {self.model_path}")
    
    def convert(self) -> str:
        """
        执行pt到onnx的转换
        
        Returns:
            生成的onnx文件路径
        """
        print(f"[PT2ONNX] 开始转换: {self.model_path}")
        
        # 备份原始配置
        backup_path = self._backup_config()
        
        try:
            # 修改配置文件
            self._modify_config()
            
            # 设置环境变量并运行exporter.py
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.pt2onnx_dir)
            
            # 构建命令
            cmd = [sys.executable, str(self.exporter_path)]
            
            print(f"[PT2ONNX] 执行命令: cd {self.pt2onnx_dir} && PYTHONPATH=./ python {self.exporter_path}")
            
            # 运行转换命令
            result = subprocess.run(
                cmd,
                cwd=str(self.pt2onnx_dir),
                env=env,
                capture_output=True,
                text=True
            )
            
            # 输出执行日志
            if result.stdout:
                print(f"[PT2ONNX] 输出:\n{result.stdout}")
            
            if result.returncode != 0:
                print(f"[PT2ONNX] 错误:\n{result.stderr}")
                raise RuntimeError(f"转换失败，返回码: {result.returncode}")
            
            # 检查onnx文件是否生成
            if self.onnx_path.exists():
                print(f"[PT2ONNX] 转换成功，onnx文件路径: {self.onnx_path}")
                return str(self.onnx_path)
            else:
                # 尝试在pt模型目录下查找onnx文件
                possible_onnx = self.model_path.parent / (self.model_path.stem + ".onnx")
                if possible_onnx.exists():
                    print(f"[PT2ONNX] 转换成功，onnx文件路径: {possible_onnx}")
                    return str(possible_onnx)
                else:
                    raise FileNotFoundError(f"转换完成但未找到onnx文件: {self.onnx_path}")
        
        finally:
            # 恢复原始配置
            self._restore_config(backup_path)
            print("[PT2ONNX] 已恢复原始配置文件")
    
    def get_onnx_path(self) -> str:
        """
        获取预期的onnx输出路径
        
        Returns:
            onnx文件路径
        """
        return str(self.onnx_path)


class ONNX2RKNN:
    """
    将ONNX模型转换为RKNN格式
    
    使用方法:
        converter = ONNX2RKNN(model_address="/path/to/your/model.onnx", duty_type="detect")
        rknn_path = converter.convert()
    """
    
    def __init__(self, model_address: str, duty_type: str = "detect"):
        """
        初始化ONNX2RKNN转换器
        
        Args:
            model_address: onnx模型的完整路径
            duty_type: 任务类型，'detect' 或 'segment'
        """
        self.model_path = Path(model_address).resolve()
        self.duty_type = duty_type.lower()
        
        # 验证模型文件是否存在
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        if not self.model_path.suffix == '.onnx':
            raise ValueError(f"模型文件必须是.onnx格式: {self.model_path}")
        
        # 验证duty_type参数
        if self.duty_type not in ['detect', 'segment']:
            raise ValueError(f"duty_type必须是 'detect' 或 'segment'，当前值: {self.duty_type}")
        
        # 获取rknn_model_zoo目录路径
        self.base_dir = Path(__file__).parent.parent.resolve()
        
        # 根据duty_type选择不同的convert.py路径
        if self.duty_type == 'detect':
            self.convert_script_path = self.base_dir / "rknn_model_zoo" / "examples" / "yolov8" / "python" / "convert.py"
        else:  # segment
            self.convert_script_path = self.base_dir / "rknn_model_zoo" / "examples" / "yolov8_seg" / "python" / "convert.py"
        
        # 验证convert.py是否存在
        if not self.convert_script_path.exists():
            raise FileNotFoundError(f"convert.py脚本不存在: {self.convert_script_path}")
        
        # 输出rknn路径（与原模型在同一目录下）
        self.rknn_path = self.model_path.with_suffix('.rknn')
    
    def convert(self, platform: str = "rk3588", dtype: str = "i8") -> str:
        """
        执行onnx到rknn的转换
        
        Args:
            platform: 目标平台，默认 'rk3588'
            dtype: 数据类型，默认 'i8'
        
        Returns:
            生成的rknn文件路径
        """
        print(f"[ONNX2RKNN] 开始转换: {self.model_path}")
        print(f"[ONNX2RKNN] 任务类型: {self.duty_type}")
        print(f"[ONNX2RKNN] 目标平台: {platform}, 数据类型: {dtype}")
        
        # 获取convert.py所在目录
        convert_dir = self.convert_script_path.parent
        
        # 构建命令: python convert.py 'onnx_path' rk3588 i8 'output_path'
        cmd = [
            sys.executable,
            str(self.convert_script_path),
            str(self.model_path),
            platform,
            dtype,
            str(self.rknn_path)
        ]
        
        print(f"[ONNX2RKNN] 执行命令: cd {convert_dir} && python convert.py {self.model_path} {platform} {dtype} {self.rknn_path}")
        
        # 运行转换命令
        result = subprocess.run(
            cmd,
            cwd=str(convert_dir),
            capture_output=True,
            text=True
        )
        
        # 输出执行日志
        if result.stdout:
            print(f"[ONNX2RKNN] 输出:\n{result.stdout}")
        
        if result.returncode != 0:
            print(f"[ONNX2RKNN] 错误:\n{result.stderr}")
            raise RuntimeError(f"转换失败，返回码: {result.returncode}")
        
        # 检查rknn文件是否生成
        if self.rknn_path.exists():
            print(f"[ONNX2RKNN] 转换成功，rknn文件路径: {self.rknn_path}")
            return str(self.rknn_path)
        else:
            raise FileNotFoundError(f"转换完成但未找到rknn文件: {self.rknn_path}")
    
    def get_rknn_path(self) -> str:
        """
        获取预期的rknn输出路径
        
        Returns:
            rknn文件路径
        """
        return str(self.rknn_path)


class PT2RKNN:
    """
    将PT模型一键转换为RKNN格式（PT -> ONNX -> RKNN）
    
    使用方法:
        converter = PT2RKNN(model_address="/path/to/your/model.pt", duty_type="detect")
        rknn_path = converter.convert()
    """
    
    def __init__(self, model_address: str, duty_type: str = "detect"):
        """
        初始化PT2RKNN转换器
        
        Args:
            model_address: pt模型的完整路径
            duty_type: 任务类型，'detect' 或 'segment'
        """
        self.model_address = model_address
        self.duty_type = duty_type
        
        # 初始化PT2ONNX转换器
        self.pt2onnx_converter = PT2ONNX(model_address=model_address)
    
    def convert(self, platform: str = "rk3588", dtype: str = "i8") -> str:
        """
        执行完整的PT到RKNN转换流程
        
        Args:
            platform: 目标平台，默认 'rk3588'
            dtype: 数据类型，默认 'i8'
        
        Returns:
            生成的rknn文件路径
        """
        print(f"[PT2RKNN] 开始完整转换流程: PT -> ONNX -> RKNN")
        print(f"[PT2RKNN] 输入模型: {self.model_address}")
        print(f"[PT2RKNN] 任务类型: {self.duty_type}")
        
        # Step 1: PT -> ONNX
        print(f"\n{'='*50}")
        print(f"[PT2RKNN] 步骤1: PT -> ONNX")
        print(f"{'='*50}")
        onnx_path = self.pt2onnx_converter.convert()
        
        # Step 2: ONNX -> RKNN
        print(f"\n{'='*50}")
        print(f"[PT2RKNN] 步骤2: ONNX -> RKNN")
        print(f"{'='*50}")
        onnx2rknn_converter = ONNX2RKNN(model_address=onnx_path, duty_type=self.duty_type)
        rknn_path = onnx2rknn_converter.convert(platform=platform, dtype=dtype)
        
        print(f"\n{'='*50}")
        print(f"[PT2RKNN] 完整转换完成！")
        print(f"[PT2RKNN] RKNN模型路径: {rknn_path}")
        print(f"{'='*50}")
        
        return rknn_path


def main():
    """
    命令行入口函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='将YOLOv8/YOLOv11的pt权重转换为onnx和rknn格式')
    parser.add_argument('--model_address', type=str, required=True, 
                        help='pt模型的完整路径')
    parser.add_argument('--duty_type', type=str, choices=['detect', 'segment'], default='detect',
                        help='任务类型: detect(目标检测) 或 segment(实例分割)')
    parser.add_argument('--platform', type=str, default='rk3588',
                        help='目标平台，默认 rk3588')
    parser.add_argument('--dtype', type=str, default='i8', choices=['i8', 'u8', 'fp'],
                        help='数据类型，默认 i8')
    parser.add_argument('--onnx_only', action='store_true',
                        help='仅转换为ONNX格式，不转换为RKNN')
    
    args = parser.parse_args()
    
    try:
        if args.onnx_only:
            # 仅转换为ONNX
            converter = PT2ONNX(model_address=args.model_address)
            onnx_path = converter.convert()
            print(f"\n转换完成！ONNX模型路径: {onnx_path}")
        else:
            # 完整转换: PT -> ONNX -> RKNN
            converter = PT2RKNN(model_address=args.model_address, duty_type=args.duty_type)
            rknn_path = converter.convert(platform=args.platform, dtype=args.dtype)
            print(f"\n转换完成！RKNN模型路径: {rknn_path}")
    except Exception as e:
        print(f"\n转换失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
