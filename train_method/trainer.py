#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8训练类
"""

import os
import sys
import logging
import threading
from pathlib import Path
from ultralytics import YOLO
import torch
from ultralytics.utils.callbacks.base import default_callbacks
import io
import re
from typing import Optional, Dict, Any
import yaml
from utils.path_handler import PathHandler

# 配置matplotlib支持中文显示
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
# 使用WenQuanYi Micro Hei中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class TrainingStoppedException(Exception):
    """训练被主动停止的异常"""
    pass

# 训练日志写入器
from utils.training_log_writer import TrainingLogWriter


class EmojiFilter(logging.Filter):
    """过滤有表情和ANSI转义序列的日志"""
    def __init__(self):
        self.emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # 表情符号范围
            "\U0001F300-\U0001F5FF"  # 符号和象形文字
            "\U0001F680-\U0001F6FF"  # 运输和地图符号
            "\U0001F700-\U0001F77F"  # Alchemical Symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed characters
            "]+", flags=re.UNICODE
        )
        # ANSI转义序列正则
        self.ansi_escape_pattern = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

    def filter(self, record):
        message = record.getMessage()
        # 检查是否包含表情符号或ANSI转义序列
        has_emoji = bool(self.emoji_pattern.search(message))
        has_ansi = bool(self.ansi_escape_pattern.search(message))
        # 如果包含任一内容则返回False，进行过滤
        return not (has_emoji or has_ansi)


class BaseLogHandler:
    """日志处理器基类，提供通用的日志处理逻辑"""

    def __init__(self, terminal, log_writer=None):
        self.terminal = terminal
        self.log_writer = log_writer
        self.line_buffer = ""
        self._written_keys = set()
        self._epoch_lines = {}

    def _clean_ansi(self, message: str) -> str:
        """清理 ANSI 转义序列"""
        return re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', message)

    def _process_buffer_lines(self):
        """处理缓冲区中的完整行"""
        while '\n' in self.line_buffer:
            line, self.line_buffer = self.line_buffer.split('\n', 1)
            if '\r' in line:
                line = line.split('\r')[-1]
            self._process_line(line)

        # 处理 \r（进度条覆盖）
        if '\r' in self.line_buffer:
            parts = self.line_buffer.split('\r')
            for part in parts[:-1]:
                part = part.strip()
                if part:
                    if '100%' in part and '|' in part:
                        self._process_line(part)
                    elif part.startswith('Epoch') and 'GPU_mem' in part:
                        self._process_line(part)
                    elif re.match(r'\s*all\s+\d+', part):
                        self._process_line(part)
            self.line_buffer = parts[-1]

    def _process_line(self, line: str):
        """处理单行日志（子类实现）"""
        raise NotImplementedError

    def _get_line_key(self, line: str):
        """生成行的去重 key"""
        if 'Scanning' in line and ('images' in line or 'cache' in line):
            if 'train' in line.lower() or line.strip().startswith('train:'):
                return 'scan_train'
            elif 'val' in line.lower() or line.strip().startswith('val:'):
                return 'scan_val'

        epoch_match = re.match(r'\s*(\d+)/(\d+)\s+[\d.]+G', line)
        if epoch_match:
            return f"epoch_{epoch_match.group(1)}"

        if line.strip().startswith('Class') and 'Images' in line:
            last_epoch = max(self._epoch_lines.keys()) if self._epoch_lines else 0
            return f"class_header_{last_epoch}"

        return None

    def _beautify_progress_bar(self, line: str) -> str:
        """美化进度条"""
        def replace_progress(match):
            return '100%|██████████|'
        return re.sub(r'100%\|[#█\d\s]+\|', replace_progress, line)

    def _should_filter(self, line: str) -> bool:
        """判断是否应该过滤该行"""
        if 'UserWarning' in line and 'Glyph' in line:
            return True
        if 'missing from font' in line:
            return True
        if 'plt.savefig' in line:
            return True
        if '/site-packages/' in line and 'Warning' in line:
            return True
        return False

    def _deduplicate_and_write(self, line: str):
        """去重并写入日志"""
        line_key = self._get_line_key(line)
        if line_key:
            if line_key in self._written_keys:
                return
            self._written_keys.add(line_key)
            if line_key.startswith('epoch_'):
                epoch_num = int(line_key.split('_')[1])
                self._epoch_lines[epoch_num] = line

        if self.log_writer:
            self.log_writer.write(line + '\n')

    def flush(self):
        """刷新缓冲区"""
        self.terminal.flush()

    def isatty(self):
        """判断是否为终端"""
        return self.terminal.isatty()


class StderrPassthrough(BaseLogHandler):
    """stderr处理类，同时输出到终端和日志（过滤警告信息，保留训练进度）"""

    def __init__(self, terminal, log_writer=None):
        super().__init__(terminal, log_writer)

    def write(self, message):
        """写入终端，如果有log_uploader也写入日志"""
        self.terminal.write(message)

        if self.log_writer and message:
            clean_msg = self._clean_ansi(message)
            if not clean_msg:
                return

            self.line_buffer += clean_msg
            self._process_buffer_lines()

    def _process_line(self, line):
        """处理单行日志"""
        line = line.strip()
        if not line:
            return

        if self._should_filter(line):
            return
        
        # 对于包含进度条的行，只保留100%完成的
        if '|' in line and '%' in line and '100%' not in line:
            progress_match = re.search(r'\|\s*(\d+)/(\d+)\s*\[', line)
            if progress_match:
                current = int(progress_match.group(1))
                total = int(progress_match.group(2))
                if current != total:
                    return
            else:
                return

        line = self._beautify_progress_bar(line)
        self._deduplicate_and_write(line)


class DualOutput(BaseLogHandler):
    """同时输出到日志写入器和终端的类，保留YOLO默认日志格式"""

    def __init__(self, log_writer, terminal):
        super().__init__(terminal, log_writer)
        self.emoji_filter = EmojiFilter()
        self._last_flush_time = 0

    def write(self, message):
        """写入消息到日志写入器和终端"""
        self.terminal.write(message)

        if not message:
            return

        clean_message = self._clean_message(message)
        if not clean_message:
            return

        self.line_buffer += clean_message
        self._process_buffer_lines()

    def _process_line(self, line):
        """处理单行日志，保留YOLO默认格式"""
        if '\r' in line:
            parts = line.split('\r')
            line = parts[-1]

        line = line.rstrip()
        if not line:
            return
        
        # 过滤规则
        if 'UserWarning: Glyph' in line or 'plt.savefig' in line:
            return
        if line.strip().startswith('plt.') or 'missing from font' in line:
            return

        if re.match(r'^\s*\d+%\s*$', line):
            return

        # 对于包含进度条的行，只保留完成的
        if '|' in line and '%' in line:
            if '100%' not in line:
                progress_match = re.search(r'\|\s*(\d+)/(\d+)\s*\[', line)
                if progress_match:
                    current = int(progress_match.group(1))
                    total = int(progress_match.group(2))
                    if current != total:
                        return

        line = self._beautify_progress_bar(line)
        self._deduplicate_and_write(line)

    def _clean_message(self, message):
        """清理消息，去除表情和ANSI转义序列"""
        if not message:
            return message
        message = self.emoji_filter.ansi_escape_pattern.sub('', message)
        message = self.emoji_filter.emoji_pattern.sub('', message)
        return message

    def flush(self):
        """刷新缓冲区"""
        if self.line_buffer.strip():
            if '\r' in self.line_buffer:
                parts = self.line_buffer.split('\r')
                self.line_buffer = parts[-1]
            line = self.line_buffer.strip()
            if line and '100%' in line:
                line = self._beautify_progress_bar(line)
                self.log_writer.write(line + '\n')
            self.line_buffer = ""
        self.terminal.flush()
        self.log_writer.flush()


class YOLOv8Trainer:
    """YOLOv8训练器类"""
    IMAGE_SUFFIXES = {".bmp", ".dng", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".pfm"}
    
    def __init__(self, task_id: str = None, db_manager=None, redis_manager=None):
        """
        初始化训练器
        
        Args:
            task_id: 训练任务ID，用于训练日志写入路径
            db_manager: 数据库管理器实例（用于epoch回调更新数据库）
            redis_manager: Redis管理器实例（用于实时更新训练进度）
        """
        self.task_id = task_id
        self.db_manager = db_manager
        self.redis_manager = redis_manager
        self.log_writer: Optional[TrainingLogWriter] = None
        self.epoch_callback: Optional['TrainingEpochCallback'] = None
        self.logger = self._setup_logging()
        self.default_params = self._get_default_params()
        self.stop_event = threading.Event()
        self.training_stopped = False
        self.model_trainer = None
        self.total_epochs: Optional[int] = None
    
    def _setup_logging(self):
        """设置日志记录（仅控制台输出，日志写入本地数据目录）"""
        # 创建专属logger
        logger_name = f"trainer_{os.getpid()}_{threading.get_ident()}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # 清除之前的handlers
        logger.handlers = []
        
        # 只添加控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(EmojiFilter())
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # 配置 ultralytics 库的日志记录器
        self._setup_ultralytics_logger()
        
        return logger
    
    def _setup_ultralytics_logger(self):
        """配置ultralytics库的日志记录器（保留原始格式）"""
        # 获取YOLO库的记录器
        yolo_logger = logging.getLogger('ultralytics')
        yolo_logger.setLevel(logging.INFO)
        
        # 清空以前的处理器，防止重复日志
        yolo_logger.handlers.clear()
        
        # 只添加控制台处理器，不添加格式化器以保留YOLO原始格式
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(EmojiFilter())
        # 使用简单格式，只输出消息内容，不添加时间戳
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        yolo_logger.addHandler(console_handler)
    
    def _setup_output_redirection(self, device='0'):
        """设置输出重定向，捕获所有 stdout 和 stderr 输出到日志写入器
        
        Args:
            device: 训练设备，如 '0', '1', '0,1' 等
        """
        if not self.task_id:
            self.logger.warning("未设置task_id，跳过输出重定向")
            return
        
        # 检测是否是多GPU模式（DDP）
        self.is_multi_gpu = ',' in str(device)
        
        if self.is_multi_gpu:
            # 多GPU模式：DDP会spawn子进程，stdout重定向在子进程中无效
            # 创建日志上传器，但不重定向stdout/stderr
            # 日志将通过监控YOLO输出目录中的results.csv文件来生成
            self.log_writer = TrainingLogWriter(self.task_id, flush_interval=0.5)
            self.log_writer.start()
            self.logger.info(f"多GPU模式(device={device})：使用CSV监控方式生成日志")
            
            # 启动CSV监控线程来生成日志（而不是监控training.log）
            self._start_csv_log_monitor()
        else:
            # 单GPU模式：使用stdout重定向
            self.log_writer = TrainingLogWriter(self.task_id, flush_interval=0.5)
            self.log_writer.start()
            
            # 保存原始的stdout和stderr
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            
            # 创建双重输出对象（stdout和stderr都写入日志）
            self.dual_stdout = DualOutput(self.log_writer, self.original_stdout)
            self.dual_stderr = StderrPassthrough(self.original_stderr, self.log_writer)
            
            # 重定向stdout和stderr
            sys.stdout = self.dual_stdout
            sys.stderr = self.dual_stderr
            self.logger.info(f"单GPU模式(device={device})：使用stdout重定向捕获日志")
    
    def _start_csv_log_monitor(self):
        """启动CSV监控线程（用于多GPU模式）
        
        监控YOLO训练输出目录中的results.csv文件，生成与单GPU模式类似的训练日志
        每轮训练生成4行日志：
        1. Epoch表头行
        2. epoch训练数据行（含进度条）
        3. Class验证表头行（含进度条）
        4. all验证结果行
        """
        import threading
        
        self._log_monitor_stop_event = threading.Event()
        
        def monitor_csv_files():
            """监控CSV文件的线程函数"""
            import time
            import csv
            from pathlib import Path
            from io import StringIO
            from datetime import datetime
            
            # 监控results.csv文件
            results_csv_path = Path(
                PathHandler.resolve_local_path('train_results', self.task_id, 'results.csv', require_exists=False)
            )
            last_line_count = 0
            header_written = False
            # 优先使用训练参数中传入的真实总轮数，避免被 CSV 推断逻辑污染
            total_epochs = self.total_epochs if self.total_epochs else None
            last_epoch_time = None
            epoch_start_time = time.time()
            
            while not self._log_monitor_stop_event.is_set():
                try:
                    # 写入训练开始信息（只写一次）
                    if not header_written:
                        self.log_writer.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - YOLOv8训练开始\n")
                        self.log_writer.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 多GPU训练模式 (DDP)\n")
                        self.log_writer.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Task ID: {self.task_id}\n")
                        self.log_writer.write("\n")
                        header_written = True
                    
                    if results_csv_path.exists():
                        with open(results_csv_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        current_line_count = len(lines)
                        
                        # 处理新行（跳过CSV表头）
                        if current_line_count > last_line_count and current_line_count > 1:
                            for i in range(max(1, last_line_count), current_line_count):
                                try:
                                    # 解析CSV行
                                    reader = csv.DictReader(StringIO(lines[0] + lines[i]))
                                    for row in reader:
                                        epoch = int(row.get('epoch', 0))
                                        epoch_time = float(row.get('time', 0)) if row.get('time') else 0
                                        
                                        # 优先使用训练参数中的真实总轮数，避免日志一直显示 **/100
                                        if self.total_epochs:
                                            total_epochs = self.total_epochs
                                        elif total_epochs is None or epoch > total_epochs:
                                            # 兜底：仅在拿不到真实参数时，按当前 epoch 推断
                                            total_epochs = epoch
                                        
                                        # 计算每个epoch的耗时
                                        if last_epoch_time is not None and epoch_time > 0:
                                            epoch_duration = epoch_time - last_epoch_time
                                        else:
                                            epoch_duration = epoch_time if epoch_time > 0 else (time.time() - epoch_start_time)
                                        last_epoch_time = epoch_time
                                        epoch_start_time = time.time()
                                        
                                        # 格式化时间显示
                                        if epoch_duration > 0:
                                            minutes = int(epoch_duration // 60)
                                            seconds = int(epoch_duration % 60)
                                            time_str = f"{minutes:02d}:{seconds:02d}"
                                            speed = f"{1/max(epoch_duration, 0.1):.2f}it/s" if epoch_duration < 10 else f"{epoch_duration:.1f}s/it"
                                        else:
                                            time_str = "00:00"
                                            speed = "-it/s"
                                        
                                        # 训练损失
                                        box_loss = row.get('train/box_loss', '0')
                                        cls_loss = row.get('train/cls_loss', '0')
                                        dfl_loss = row.get('train/dfl_loss', '0')
                                        
                                        # 验证指标
                                        precision = row.get('metrics/precision(B)', '0')
                                        recall = row.get('metrics/recall(B)', '0')
                                        mAP50 = row.get('metrics/mAP50(B)', '0')
                                        mAP50_95 = row.get('metrics/mAP50-95(B)', '0')
                                        
                                        # 获取images和instances数量（如果CSV中有的话）
                                        images_count = row.get('val/images', '-')
                                        instances_count = row.get('val/instances', '-')
                                        
                                        try:
                                            box_loss_f = float(box_loss)
                                            cls_loss_f = float(cls_loss)
                                            dfl_loss_f = float(dfl_loss)
                                            precision_f = float(precision)
                                            recall_f = float(recall)
                                            mAP50_f = float(mAP50)
                                            mAP50_95_f = float(mAP50_95)
                                            
                                            # 第1行: Epoch表头（每轮都输出）
                                            self.log_writer.write("      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n")
                                            
                                            # 第2行: epoch训练数据行（含进度条）
                                            epoch_line = f"{epoch}/{total_epochs}      3.35G    {box_loss_f:.4f}    {cls_loss_f:.4f}    {dfl_loss_f:.4f}        -      640: 100%|██████████| [{time_str}, {speed}]\n"
                                            self.log_writer.write(epoch_line)
                                            
                                            # 第3行: Class验证表头行（含进度条）
                                            class_line = f"Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| [{time_str}, {speed}]\n"
                                            self.log_writer.write(class_line)
                                            
                                            # 第4行: all验证结果行
                                            val_line = f"                   all          {images_count}          {instances_count}    {precision_f:.3f}          {recall_f:.3f}      {mAP50_f:.3f}      {mAP50_95_f:.3f}\n"
                                            self.log_writer.write(val_line)
                                            
                                        except (ValueError, TypeError) as e:
                                            # 数值转换失败时记录错误
                                            self.log_writer.write(f"# Epoch {epoch} 数据解析错误: {str(e)}\n")
                                        
                                except Exception as e:
                                    self.logger.debug(f"CSV日志监控解析行失败: {e}")
                            
                            last_line_count = current_line_count
                
                except Exception as e:
                    self.logger.debug(f"CSV日志监控循环出错: {e}")
                
                # 每2秒检查一次
                self._log_monitor_stop_event.wait(2)
            
            # 监控结束时写入完成信息
            if header_written:
                self.log_writer.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 训练完成\n")
        
        self._log_monitor_thread = threading.Thread(
            target=monitor_csv_files,
            name=f"CSVLogMonitor-{self.task_id}",
            daemon=True
        )
        self._log_monitor_thread.start()
        self.logger.info(f"CSV日志监控线程已启动: task_id={self.task_id}")
    
    def stop_training(self):
        """停止当前正在进行的训练"""
        self.logger.info("收到停止训练信号")
        self.stop_event.set()
        self.training_stopped = True
        
        # 如果有trainer实例，尝试停止它
        if self.model_trainer and hasattr(self.model_trainer, 'stop_training'):
            try:
                self.model_trainer.stop_training = True
                self.logger.info("已设置trainer停止标志")
            except Exception as e:
                self.logger.warning(f"停止训练失败: {str(e)}")
    
    def _create_stop_callback(self):
        """创建自定义停止回调函数
        
        在检测到停止信号时中断训练
        """
        def on_train_epoch_end(trainer):
            if self.stop_event.is_set():
                self.logger.info("检测到停止信号，强制停止训练...")
                trainer.stop_training = True
                # 抛出异常强制中断训练循环
                raise TrainingStoppedException("训练已被用户主动停止")
        
        def on_train_batch_end(trainer):
            if self.stop_event.is_set():
                self.logger.info("检测到停止信号（batch结束），强制停止训练...")
                trainer.stop_training = True
                # 抛出异常强制中断训练循环
                raise TrainingStoppedException("训练已被用户主动停止")
                
        return {
            'on_train_epoch_end': on_train_epoch_end,
            'on_train_batch_end': on_train_batch_end
        }
    
    def _get_default_params(self):
        """获取默认参数"""
        return {
            'dutyType': 'detect',
            'model_type': 0,
            'model_size': 'n',
            'incremental_model_address': '',
            'batch': 8,
            'device': '-1',
            'imgsz': 640,
            'epoch': 100,
            'yaml': '',
            'model': '',
            'project': 'train_results',
            'name': 'task_id',
            'lr0': 0.0001,
            'lrf': 0.001,
            'cos_lr': True,
            'warmup_epochs': 5,
            'warmup_bias_lr': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'fliplr': 0.5,
            'amp': False,
            'patience': 0,
            'save_period': -1,
            'workers': 8,
            'resume': '',
            'kpt_num': 0,
        }

    def _normalize_dataset_entries(self, entry):
        """将 YAML 中的 train/val 配置统一转换为路径列表"""
        if entry is None:
            return []
        if isinstance(entry, (list, tuple)):
            return [str(item) for item in entry if str(item).strip()]
        entry_str = str(entry).strip()
        return [entry_str] if entry_str else []

    def _resolve_dataset_entry(self, dataset_root: Path, entry: str) -> Path:
        """解析数据集路径，兼容绝对路径和相对 path 根目录"""
        entry_path = Path(entry).expanduser()
        if entry_path.is_absolute():
            return entry_path
        return (dataset_root / entry_path).resolve()

    def _scan_label_file(self, label_path: Path, task_type: str, class_count: Optional[int],
                         kpt_num: Optional[int] = None):
        """检查单个标签文件是否包含至少一条有效标注"""
        if task_type == 'detect':
            min_fields = 5
        elif task_type == 'pose':
            # class + bbox(4) + K * (x,y,v)
            expected = 5 + 3 * int(kpt_num or 0)
            min_fields = expected if expected > 5 else 5
        else:
            min_fields = 7
        issues = []
        has_valid_annotation = False

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_no, raw_line in enumerate(f, 1):
                    line = raw_line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < min_fields:
                        issues.append(f"{label_path.name}:{line_no} 字段数不足")
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        coords = [float(value) for value in parts[1:]]
                    except ValueError:
                        issues.append(f"{label_path.name}:{line_no} 存在非数字内容")
                        continue

                    if class_count is not None and not (0 <= class_id < class_count):
                        issues.append(f"{label_path.name}:{line_no} 类别ID越界: {class_id}")
                        continue

                    if task_type == 'detect':
                        if len(coords) != 4:
                            issues.append(f"{label_path.name}:{line_no} 检测标签应为 5 列")
                            continue
                    elif task_type == 'pose':
                        if not kpt_num or kpt_num <= 0:
                            issues.append(f"{label_path.name}:{line_no} pose 任务缺少 kpt_num 配置")
                            continue
                        expected_kpt_cols = 4 + 3 * int(kpt_num)
                        if len(coords) != expected_kpt_cols:
                            issues.append(
                                f"{label_path.name}:{line_no} pose 标签列数应为 {1 + expected_kpt_cols}"
                            )
                            continue
                    else:
                        if len(coords) < 6 or len(coords) % 2 != 0:
                            issues.append(f"{label_path.name}:{line_no} 分割标签点数非法")
                            continue

                    has_valid_annotation = True

        except Exception as e:
            issues.append(f"{label_path.name} 读取失败: {e}")

        return has_valid_annotation, issues

    def _inspect_dataset_split(self, split_name: str, entries, dataset_root: Path, task_type: str,
                               class_count: Optional[int], kpt_num: Optional[int] = None):
        """检查单个数据集切分的目录、图片和标签情况"""
        errors = []
        image_count = 0
        label_count = 0
        matched_pairs = 0
        valid_label_files = 0
        invalid_issue_samples = []
        missing_label_samples = []

        if not entries:
            return [f"{split_name} 未在 dataset.yaml 中配置路径"]

        for entry in entries:
            image_dir = self._resolve_dataset_entry(dataset_root, entry)
            if not image_dir.exists():
                errors.append(f"{split_name} 图片目录不存在: {image_dir}")
                continue
            if not image_dir.is_dir():
                errors.append(f"{split_name} 配置不是目录: {image_dir}")
                continue

            label_dir = image_dir.parent / "labels"
            if not label_dir.exists():
                errors.append(f"{split_name} 标签目录不存在: {label_dir}")
                continue
            if not label_dir.is_dir():
                errors.append(f"{split_name} 标签路径不是目录: {label_dir}")
                continue

            images = [
                path for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            ]
            labels = [path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"]

            image_count += len(images)
            label_count += len(labels)

            for image_path in images:
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    if len(missing_label_samples) < 3:
                        missing_label_samples.append(image_path.name)
                    continue

                matched_pairs += 1
                has_valid_annotation, issues = self._scan_label_file(
                    label_path, task_type, class_count, kpt_num=kpt_num
                )
                if has_valid_annotation:
                    valid_label_files += 1
                elif issues and len(invalid_issue_samples) < 3:
                    invalid_issue_samples.append(issues[0])

        if image_count == 0:
            errors.append(f"{split_name} 没有找到任何图片")
        if label_count == 0:
            errors.append(f"{split_name} 没有找到任何标签文件")
        if matched_pairs == 0:
            errors.append(f"{split_name} 图片与标签未能成功配对")
        if valid_label_files == 0:
            errors.append(f"{split_name} 没有任何有效标注，训练会被视为全背景数据")
        if missing_label_samples:
            errors.append(f"{split_name} 缺少对应标签示例: {', '.join(missing_label_samples)}")
        if invalid_issue_samples:
            errors.append(f"{split_name} 无效标签示例: {', '.join(invalid_issue_samples)}")

        return errors

    def _validate_dataset_config(self, yaml_path: str, task_type: str):
        """训练前检查 dataset.yaml 与目录结构，提前暴露数据集问题"""
        yaml_file = Path(yaml_path)

        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"读取数据集配置失败: {yaml_file} - {e}")
            return False

        dataset_root = data_config.get('path')
        if dataset_root:
            dataset_root = self._resolve_dataset_entry(yaml_file.parent, str(dataset_root))
        else:
            dataset_root = yaml_file.parent

        names = data_config.get('names')
        class_count = None
        if isinstance(names, list):
            class_count = len(names)
        elif isinstance(names, dict):
            class_count = len(names)

        # pose 任务从 yaml 中读取 kpt_shape，作为 _scan_label_file 的列数校验依据
        kpt_num = None
        if task_type == 'pose':
            kpt_shape = data_config.get('kpt_shape')
            if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
                try:
                    kpt_num = int(kpt_shape[0])
                except (TypeError, ValueError):
                    kpt_num = None
            if not kpt_num or kpt_num <= 0:
                self.logger.error("pose 任务的 dataset.yaml 缺少有效的 kpt_shape")
                return False

        dataset_errors = []
        dataset_errors.extend(
            self._inspect_dataset_split(
                split_name='train',
                entries=self._normalize_dataset_entries(data_config.get('train')),
                dataset_root=dataset_root,
                task_type=task_type,
                class_count=class_count,
                kpt_num=kpt_num,
            )
        )
        dataset_errors.extend(
            self._inspect_dataset_split(
                split_name='val',
                entries=self._normalize_dataset_entries(data_config.get('val')),
                dataset_root=dataset_root,
                task_type=task_type,
                class_count=class_count,
                kpt_num=kpt_num,
            )
        )

        if dataset_errors:
            self.logger.error("数据集预检查失败:")
            for error in dataset_errors:
                self.logger.error(f"  - {error}")
            return False

        return True
    
    def validate_arguments(self, args):
        """验证参数"""
        # 使用默认值填充缺失的参数
        merged_args = self.default_params.copy()
        merged_args.update(args)
        args = merged_args
        
        if not os.path.exists(args['yaml']):
            self.logger.error(f"数据集配置文件不存在: {args['yaml']}")
            return False

        if not self._validate_dataset_config(args['yaml'], args['dutyType']):
            return False
        
        if args.get('model_type', 0) == 1:
            if not args.get('incremental_model_address', ''):
                self.logger.error("增量训练模式下必须提供incremental_model_address参数")
                return False
            if not os.path.exists(args['incremental_model_address']):
                self.logger.error(f"增量训练模型文件不存在: {args['incremental_model_address']}")
                return False
        
        device = str(args.get('device', '-1'))
        if torch.cuda.is_available():
            visible_devices = []
            for idx in range(torch.cuda.device_count()):
                try:
                    visible_devices.append(f"{idx}:{torch.cuda.get_device_name(idx)}")
                except Exception:
                    visible_devices.append(f"{idx}:unknown")
            self.logger.info(
                f"CUDA环境: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                f"device_count={torch.cuda.device_count()}, visible={visible_devices}"
            )

        if device != 'auto' and device != 'cpu' and device != '-1':
            # 支持多GPU格式，如 '0,1' 或单GPU '0'
            device_ids = device.split(',')
            for dev_id in device_ids:
                dev_id = dev_id.strip()
                try:
                    device_id = int(dev_id)
                    if device_id >= 0 and torch.cuda.is_available() and device_id >= torch.cuda.device_count():
                        self.logger.error(f"GPU设备 {device_id} 不可用")
                        return False
                except ValueError:
                    self.logger.error(f"无效的设备参数: {device}")
                    return False
        
        if args.get('batch', 1) <= 0:
            self.logger.error("批次大小必须大于0")
            return False
        
        if args.get('epoch', 1) <= 0:
            self.logger.error("训练轮数必须大于0")
            return False
        
        if args.get('imgsz', 640) <= 0:
            self.logger.error("图像尺寸必须大于0")
            return False
        
        return True
    
    def select_model_path(self, args):
        """根据参数选择模型路径"""
        try:
            if args['model_type'] == 1:
                model_path = args['incremental_model_address']
                self.logger.info(f"增量训练模式，使用模型: {model_path}")
                return model_path

            script_dir = Path(__file__).parent.parent  # trainer目录
            pre_model_dir = script_dir / "pre_model"

            if args['dutyType'] == 'detect':
                model_name = f"yolov8{args['model_size']}.pt"
                fallback_name = "yolov8n.pt"
            elif args['dutyType'] == 'segment':
                model_name = f"yolov8{args['model_size']}-seg.pt"
                fallback_name = "yolov8n-seg.pt"
            elif args['dutyType'] == 'pose':
                # pose 模型采用 yolo11{size}-pose.pt
                model_name = f"yolo11{args['model_size']}-pose.pt"
                fallback_name = "yolo11n-pose.pt"
            else:
                raise ValueError(f"不支持的dutyType: {args['dutyType']}")

            model_path = pre_model_dir / model_name

            if not model_path.exists():
                self.logger.error(f"预训练模型不存在: {model_path}")
                fallback_model = pre_model_dir / fallback_name
                if fallback_model.exists():
                    self.logger.warning(f"使用备用模型: {fallback_model}")
                    return str(fallback_model)
                else:
                    raise FileNotFoundError(f"模型文件不存在且无备用模型: {model_path}")

            self.logger.info(f"自动选择模型: {model_path}")
            return str(model_path)

        except Exception as e:
            self.logger.error(f"模型选择失败: {str(e)}")
            raise
    
    def _run_training(self, args, task_label: str, ultra_task: Optional[str] = None):
        """通用训练执行方法，detect/segment/pose 共用

        Args:
            args: 训练参数
            task_label: 任务中文标签（用于日志）
            ultra_task: 显式传入 Ultralytics 的 task 名称（pose 必填）
        """
        model = None
        try:
            self.logger.info(f"开始初始化YOLOv8{task_label}模型...")

            if args['resume']:
                self.logger.info(f"从检查点恢复训练: {args['resume']}")
                model = YOLO(args['resume'])
            else:
                selected_model = self.select_model_path(args)
                self.logger.info(f"加载预训练模型: {selected_model}")
                if ultra_task:
                    model = YOLO(selected_model, task=ultra_task)
                else:
                    model = YOLO(selected_model)

            train_args = {
                'data': args['yaml'],
                'epochs': args['epoch'],
                'batch': args['batch'],
                'imgsz': args['imgsz'],
                'device': args['device'],
                'project': args['project'],
                'name': args['name'],
                'lr0': args['lr0'],
                'lrf': args['lrf'],
                'cos_lr': args['cos_lr'],
                'warmup_epochs': args['warmup_epochs'],
                'warmup_bias_lr': args['warmup_bias_lr'],
                'momentum': args['momentum'],
                'weight_decay': args['weight_decay'],
                'fliplr': args['fliplr'],
                'amp': args['amp'],
                'patience': args['patience'],
                'save_period': args['save_period'],
                'workers': args['workers'],
                'verbose': False,
                'save': True,
                'plots': True,
                'val': True
            }

            # pose 任务显式声明 task，避免被预训练权重的默认 task 误判
            if ultra_task:
                train_args['task'] = ultra_task

            self.logger.info(f"{task_label}配置: 数据集={train_args['data']}, Epochs={train_args['epochs']}, Batch={train_args['batch']}")

            # 添加停止回调
            for cb_name, cb_func in self._create_stop_callback().items():
                model.add_callback(cb_name, cb_func)

            # 添加 Epoch 训练回调
            if self.task_id:
                try:
                    from utils.epoch_callback import TrainingEpochCallback
                    output_dir = Path(train_args['project']) / train_args['name']
                    self.epoch_callback = TrainingEpochCallback(self.task_id, output_dir, self.db_manager, self.redis_manager)
                    model.add_callback('on_train_epoch_end', self.epoch_callback.on_train_epoch_end)
                    self.logger.info(f"已添加 Epoch 训练回调: {output_dir}")
                except Exception as e:
                    self.logger.warning(f"添加Epoch回调失败: {e}")

            self.logger.info(f"开始{task_label}训练...")
            results = model.train(**train_args)

            self.model_trainer = model.trainer if hasattr(model, 'trainer') else None

            if self.training_stopped and self.stop_event.is_set():
                self.logger.info(f"{task_label}训练已被主动停止!")
                return 'stopped'
            else:
                self.logger.info(f"{task_label}训练完成!")

            if hasattr(model, 'trainer') and model.trainer:
                self.logger.info(f"最佳模型保存路径: {model.trainer.best}")
                self.logger.info(f"最后模型保存路径: {model.trainer.last}")
                self.best_model_path = model.trainer.best
                self.last_model_path = model.trainer.last

            if hasattr(results, 'results_dict'):
                self.logger.info("训练结果:")
                for metric, value in results.results_dict.items():
                    self.logger.info(f"  {metric}: {value}")

            return True

        except TrainingStoppedException as e:
            self.logger.info(f"{task_label}训练已被主动停止: {str(e)}")
            self.training_stopped = True
            if model and hasattr(model, 'trainer') and model.trainer:
                self.best_model_path = getattr(model.trainer, 'best', None)
                self.last_model_path = getattr(model.trainer, 'last', None)
                if self.best_model_path:
                    self.logger.info(f"停止时保存的最佳模型: {self.best_model_path}")
            return 'stopped'
        except Exception as e:
            self.logger.error(f"{task_label}训练过程中发生错误: {str(e)}")
            return False

    def train_detection(self, args):
        """执行目标检测训练"""
        return self._run_training(args, "目标检测")

    def train_segmentation(self, args):
        """执行实例分割训练"""
        return self._run_training(args, "实例分割")

    def train_pose(self, args):
        """执行关键点(pose)训练"""
        return self._run_training(args, "关键点", ultra_task='pose')
    
    def train(self, **kwargs):
        """
        程序化训练接口
        
        Args:
            **kwargs: 训练参数，会与默认参数合并
            
        Returns:
            bool: 训练是否成功
        """
        # 合并参数
        args = self.default_params.copy()
        args.update(kwargs)

        # 记录训练总轮数，供日志监控线程使用，避免被硬编码下限覆盖
        try:
            self.total_epochs = int(args.get('epoch', 0)) or None
        except (TypeError, ValueError):
            self.total_epochs = None

        # 设置输出重定向（启动日志写入），传入device参数以检测多GPU模式
        self._setup_output_redirection(device=args.get('device', '0'))
        
        self.logger.info("YOLOv8训练开始")
        # 只输出关键参数
        self.logger.info(f"任务类型: {args['dutyType']}, 模型: {args['model_size']}, Epochs: {args['epoch']}, Batch: {args['batch']}, Device: {args['device']}")
        
        # 验证参数
        if not self.validate_arguments(args):
            self.logger.error("参数验证失败")
            self._cleanup_output_redirection()
            return False
        
        # 让YOLO自己创建输出目录，避免目录冲突
        output_dir = Path(args['project']) / args['name']
        self.logger.info(f"预期输出目录: {output_dir}")
        
        # 执行训练
        self.logger.info(f"训练任务类型: {args['dutyType']}")
        
        try:
            if args['dutyType'] == 'detect':
                result = self.train_detection(args)
            elif args['dutyType'] == 'segment':
                result = self.train_segmentation(args)
            elif args['dutyType'] == 'pose':
                result = self.train_pose(args)
            else:
                self.logger.error(f"不支持的训练任务类型: {args['dutyType']}")
                return False
            
            # 处理返回值：True=成功, False=失败, 'stopped'=被主动停止
            if result == 'stopped':
                self.logger.info("训练已被主动停止!")
                return 'stopped'
            elif result:
                self.logger.info("训练成功完成!")
                return True
            else:
                self.logger.error("训练失败!")
                return False
                
        finally:
            # 无论成功还是失败都清理输出重定向（停止日志写入）
            self._cleanup_output_redirection()
    
    def _cleanup_output_redirection(self):
        """清理输出重定向，恢复原始stdout/stderr"""
        try:
            # 停止日志文件监控线程（多GPU模式）
            if hasattr(self, '_log_monitor_stop_event'):
                self._log_monitor_stop_event.set()
                if hasattr(self, '_log_monitor_thread') and self._log_monitor_thread.is_alive():
                    self._log_monitor_thread.join(timeout=3)
            
            # 恢复原始的stdout和stderr（单GPU模式）
            if hasattr(self, 'original_stdout'):
                sys.stdout = self.original_stdout
            if hasattr(self, 'original_stderr'):
                sys.stderr = self.original_stderr
            
            # 停止训练日志写入器
            if hasattr(self, 'log_writer') and self.log_writer:
                self.log_writer.stop()
        except Exception as e:
            # 即使清理失败也不影响训练结果
            pass
    
    def get_log_path(self) -> Optional[str]:
        """获取日志的本地完整路径"""
        if self.log_writer:
            return self.log_writer.get_log_path()
        return None
