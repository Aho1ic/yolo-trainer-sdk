#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO训练Epoch回调函数
在每轮训练结束时自动上传results.csv和日志到MinIO
"""

import logging
import time
from pathlib import Path

from utils.path_handler import PathHandler

logger = logging.getLogger(__name__)


class EpochMinIOUploader:
    """每个Epoch结束时登记对象路径并更新训练进度的回调类"""
    
    def __init__(self, task_id: str, local_save_dir: Path, db_manager=None, redis_manager=None):
        """
        初始化Epoch MinIO上传器
        
        Args:
            task_id: 训练任务ID
            local_save_dir: 本地保存目录（训练结果目录）
            db_manager: 数据库管理器实例（用于更新log_path、chart_path）
            redis_manager: Redis管理器实例（用于实时更新训练进度）
        """
        self.task_id = task_id
        self.local_save_dir = Path(local_save_dir)
        self.db_manager = db_manager
        self.redis_manager = redis_manager
        self._last_upload_epoch = -1
        # 标记是否已经更新过数据库路径
        self._log_path_updated = False
        self._chart_path_updated = False
        # 训练进度跟踪
        self._training_start_time = None
        self._epoch_times = []  # 记录每个epoch的耗时
        
    def on_train_epoch_end(self, trainer):
        """
        每个Epoch结束时的回调函数
        
        Args:
            trainer: YOLO训练器对象
        """
        try:
            current_epoch = trainer.epoch
            total_epochs = trainer.epochs
            
            # 避免重复上传同一个epoch
            if current_epoch == self._last_upload_epoch:
                return
            
            # 第一次回调时，从trainer获取实际的保存目录
            if hasattr(trainer, 'save_dir') and trainer.save_dir:
                actual_save_dir = Path(trainer.save_dir)
                if actual_save_dir != self.local_save_dir:
                    logger.info(f"更新保存目录: {self.local_save_dir} -> {actual_save_dir}")
                    self.local_save_dir = actual_save_dir
            
            # 记录训练开始时间（第一个epoch）
            current_time = time.time()
            if self._training_start_time is None:
                self._training_start_time = current_time
            
            # 计算并更新训练进度
            self._update_training_progress(current_epoch, total_epochs, current_time)
            
            logger.info(f"Epoch {current_epoch} 结束，开始登记训练结果对象路径...")
            logger.info(f"本地保存目录: {self.local_save_dir}")
            
            # 上传results.csv
            results_csv = self.local_save_dir / 'results.csv'
            logger.info(f"检查 results.csv 路径: {results_csv}, 存在: {results_csv.exists()}")
            if results_csv.exists():
                minio_path = f'{PathHandler.get_storage_base_path()}/train_results/{self.task_id}/results.csv'
                success = self._upload_file(results_csv, minio_path)
                # 首次上传成功后更新数据库chart_path
                if success and not self._chart_path_updated and self.db_manager:
                    full_minio_path = self._get_full_minio_path(minio_path)
                    if self.db_manager.update_task_chart_path(self.task_id, full_minio_path):
                        self._chart_path_updated = True
                        logger.info(f"数据库chart_path已更新: {full_minio_path}")
            else:
                logger.warning(f"results.csv 不存在: {results_csv}")
            
            # 上传训练日志
            log_file = Path(
                PathHandler.resolve_storage_local_path('train_log', self.task_id, 'training.log', require_exists=False)
            )
            if log_file.exists():
                minio_path = f'{PathHandler.get_storage_base_path()}/train_log/{self.task_id}/training.log'
                success = self._upload_file(log_file, minio_path)
                # 首次上传成功后更新数据库log_path
                if success and not self._log_path_updated and self.db_manager:
                    full_minio_path = self._get_full_minio_path(minio_path)
                    if self.db_manager.update_task_log_path(self.task_id, full_minio_path):
                        self._log_path_updated = True
                        logger.info(f"数据库log_path已更新: {full_minio_path}")
            else:
                logger.debug(f"训练日志文件不存在: {log_file}")
            
            self._last_upload_epoch = current_epoch
            logger.info(f"Epoch {current_epoch} 路径登记完成")
                
        except Exception as e:
            # 上传失败不应中断训练
            logger.error(f"Epoch回调上传失败: {e}")
    
    def _upload_file(self, local_path: Path, minio_path: str) -> bool:
        """
        NFS 直写模式下校验单个文件是否已存在
        
        Args:
            local_path: 本地文件路径
            minio_path: MinIO目标路径
            
        Returns:
            bool: 文件是否存在
        """
        try:
            if local_path.exists():
                full_path = self._get_full_minio_path(minio_path)
                logger.info(f"文件已存在于NFS挂载目录: {local_path.name} -> {full_path}")
                return True
            logger.warning(f"文件不存在，无法登记对象路径: {local_path}")
            return False
        except Exception as e:
            logger.error(f"登记文件路径出错 {local_path}: {e}")
            return False
    
    def _get_full_minio_path(self, minio_path: str) -> str:
        """
        获取完整的MinIO路径
        
        Args:
            minio_path: MinIO对象路径
            
        Returns:
            str: 完整的MinIO路径 (minio://bucket/path)
        """
        return PathHandler.build_minio_uri(PathHandler.get_storage_bucket(), minio_path)
    
    def _update_training_progress(self, current_epoch: int, total_epochs: int, current_time: float):
        """
        更新训练进度到Redis（实时更新current_epoch和eta_seconds）
        
        Args:
            current_epoch: 当前epoch（从0开始）
            total_epochs: 总epoch数
            current_time: 当前时间戳
        """
        if not self.redis_manager:
            return
        
        try:
            # 计算已完成的epoch数（current_epoch是从0开始的索引，回调在epoch结束时触发）
            completed_epochs = current_epoch + 1
            
            # 计算预计剩余时间
            eta_seconds = 0
            if completed_epochs > 0 and self._training_start_time:
                # 计算已用时间
                elapsed_time = current_time - self._training_start_time
                # 计算平均每个epoch的时间
                avg_epoch_time = elapsed_time / completed_epochs
                # 计算剩余epoch数
                remaining_epochs = total_epochs - completed_epochs
                # 计算预计剩余时间（秒）
                eta_seconds = int(avg_epoch_time * remaining_epochs)
                
                logger.debug(f"训练进度: {completed_epochs}/{total_epochs}, "
                           f"已用时间: {elapsed_time:.1f}s, "
                           f"平均每轮: {avg_epoch_time:.1f}s, "
                           f"预计剩余: {eta_seconds}s")
            
            # 更新Redis中的current_epoch和eta_seconds字段
            self.redis_manager.update_task_field(self.task_id, 'current_epoch', completed_epochs)
            self.redis_manager.update_task_field(self.task_id, 'eta_seconds', eta_seconds)
            logger.debug(f"Redis训练进度已更新: task_id={self.task_id}, current_epoch={completed_epochs}, eta_seconds={eta_seconds}")
            
        except Exception as e:
            logger.error(f"更新训练进度到Redis失败: {e}")
