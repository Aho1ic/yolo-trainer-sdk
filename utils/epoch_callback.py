#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 训练 Epoch 回调：每轮结束后登记结果路径并更新进度"""

import logging
import time
from pathlib import Path

from utils.path_handler import PathHandler

logger = logging.getLogger(__name__)


class TrainingEpochCallback:
    """每个 Epoch 结束时检查训练结果文件并更新数据库/Redis 进度的回调类"""

    def __init__(self, task_id: str, local_save_dir: Path, db_manager=None, redis_manager=None):
        """
        Args:
            task_id: 训练任务ID
            local_save_dir: 本地保存目录（训练结果目录）
            db_manager: 数据库管理器实例（用于更新 log_path、chart_path）
            redis_manager: Redis管理器实例（用于实时更新训练进度）
        """
        self.task_id = task_id
        self.local_save_dir = Path(local_save_dir)
        self.db_manager = db_manager
        self.redis_manager = redis_manager
        self._last_processed_epoch = -1
        self._log_path_updated = False
        self._chart_path_updated = False
        self._training_start_time = None

    def on_train_epoch_end(self, trainer):
        """每个 Epoch 结束时的回调函数"""
        try:
            current_epoch = trainer.epoch
            total_epochs = trainer.epochs

            if current_epoch == self._last_processed_epoch:
                return

            if hasattr(trainer, 'save_dir') and trainer.save_dir:
                actual_save_dir = Path(trainer.save_dir)
                if actual_save_dir != self.local_save_dir:
                    logger.info(f"更新保存目录: {self.local_save_dir} -> {actual_save_dir}")
                    self.local_save_dir = actual_save_dir

            current_time = time.time()
            if self._training_start_time is None:
                self._training_start_time = current_time

            self._update_training_progress(current_epoch, total_epochs, current_time)

            logger.info(f"Epoch {current_epoch} 结束，开始检查训练结果...")
            logger.info(f"本地保存目录: {self.local_save_dir}")

            results_csv = self.local_save_dir / 'results.csv'
            logger.info(f"检查 results.csv 路径: {results_csv}, 存在: {results_csv.exists()}")
            if results_csv.exists():
                local_path = PathHandler.build_data_path('train_results', self.task_id, 'results.csv')
                if not self._chart_path_updated and self.db_manager:
                    if self.db_manager.update_task_chart_path(self.task_id, local_path):
                        self._chart_path_updated = True
                        logger.info(f"数据库 chart_path 已更新: {local_path}")
            else:
                logger.warning(f"results.csv 不存在: {results_csv}")

            log_file = Path(PathHandler.build_data_path('train_log', self.task_id, 'training.log'))
            if log_file.exists():
                if not self._log_path_updated and self.db_manager:
                    if self.db_manager.update_task_log_path(self.task_id, str(log_file)):
                        self._log_path_updated = True
                        logger.info(f"数据库 log_path 已更新: {log_file}")
            else:
                logger.debug(f"训练日志文件不存在: {log_file}")

            self._last_processed_epoch = current_epoch
            logger.info(f"Epoch {current_epoch} 路径登记完成")

        except Exception as e:
            logger.error(f"Epoch 回调失败: {e}")

    def _update_training_progress(self, current_epoch: int, total_epochs: int, current_time: float):
        """更新训练进度到 Redis"""
        if not self.redis_manager:
            return

        try:
            completed_epochs = current_epoch + 1
            eta_seconds = 0
            if completed_epochs > 0 and self._training_start_time:
                elapsed_time = current_time - self._training_start_time
                avg_epoch_time = elapsed_time / completed_epochs
                remaining_epochs = total_epochs - completed_epochs
                eta_seconds = int(avg_epoch_time * remaining_epochs)
                logger.debug(
                    f"训练进度: {completed_epochs}/{total_epochs}, "
                    f"已用时间: {elapsed_time:.1f}s, "
                    f"平均每轮: {avg_epoch_time:.1f}s, "
                    f"预计剩余: {eta_seconds}s"
                )

            self.redis_manager.update_task_field(self.task_id, 'current_epoch', completed_epochs)
            self.redis_manager.update_task_field(self.task_id, 'eta_seconds', eta_seconds)
            logger.debug(
                f"Redis 训练进度已更新: task_id={self.task_id}, "
                f"current_epoch={completed_epochs}, eta_seconds={eta_seconds}"
            )
        except Exception as e:
            logger.error(f"更新训练进度到 Redis 失败: {e}")
