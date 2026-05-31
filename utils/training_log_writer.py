#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练日志写入器：实时把训练日志写入本地数据目录"""

import threading
import logging
from typing import Optional
from pathlib import Path

from utils.path_handler import PathHandler

logger = logging.getLogger(__name__)


class TrainingLogWriter:
    """将训练日志实时写入本地数据目录的写入器"""

    def __init__(self, task_id: str, flush_interval: float = 0.5):
        """
        Args:
            task_id: 训练任务ID
            flush_interval: 刷盘间隔（秒），默认 0.5 秒
        """
        self.task_id = task_id
        self.flush_interval = flush_interval
        self._pending_chunks = []
        self.log_buffer = None  # 兼容旧测试/调用方，实际写入使用 _pending_chunks
        self.stop_event = threading.Event()
        self.writer_thread: Optional[threading.Thread] = None
        self.log_path = PathHandler.build_data_path('train_log', task_id, 'training.log')
        self._lock = threading.Lock()

        self._ensure_log_dir_exists()

    def _ensure_log_dir_exists(self) -> None:
        """确保日志目录存在"""
        try:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"日志目录已创建: {Path(self.log_path).parent}")
        except Exception as e:
            logger.warning(f"创建日志目录失败: {e}")

    def set_local_save_path(self, path: str) -> None:
        """设置本地日志保存路径"""
        with self._lock:
            self.log_path = path
            logger.info(f"日志保存路径已更新: {path}")

    def start(self) -> None:
        """启动写入线程"""
        if self.writer_thread is not None and self.writer_thread.is_alive():
            logger.warning(f"写入线程已在运行: task_id={self.task_id}")
            return

        self._create_empty_log_file()

        self.stop_event.clear()
        self.writer_thread = threading.Thread(
            target=self._write_loop,
            name=f"TrainingLogWriter-{self.task_id}",
            daemon=True
        )
        self.writer_thread.start()
        logger.info(f"日志写入器已启动: task_id={self.task_id}, log_path={self.log_path}")

    def _create_empty_log_file(self) -> None:
        """创建空的日志文件"""
        try:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w', encoding='utf-8'):
                pass
            logger.info(f"日志文件已创建: {self.log_path}")
        except Exception as e:
            logger.error(f"创建日志文件失败: {e}")

    def stop(self) -> None:
        """停止写入线程，最终保存日志"""
        self.stop_event.set()

        if self.writer_thread is not None:
            self.writer_thread.join(timeout=5.0)
            if self.writer_thread.is_alive():
                logger.warning(f"写入线程未能在超时时间内停止: task_id={self.task_id}")

        self._save_to_local()
        logger.info(f"日志已保存到本地: {self.log_path}")
        logger.info(f"日志写入器已停止: task_id={self.task_id}")

    def write(self, content: str) -> None:
        """写入日志内容到缓冲区"""
        if not content:
            return
        with self._lock:
            if not hasattr(self, '_pending_chunks'):
                self._pending_chunks = []
            self._pending_chunks.append(content)

    def flush(self) -> None:
        """兼容文件接口"""
        self._save_to_local()

    def _write_loop(self) -> None:
        """每 flush_interval 秒保存一次本地日志"""
        save_count = 0
        while not self.stop_event.is_set():
            try:
                if self._save_to_local():
                    save_count += 1
                    if save_count % 20 == 0:
                        logger.info(f"日志已保存到本地: {self.log_path}")
            except Exception as e:
                logger.error(f"本地日志保存失败: task_id={self.task_id}, error={e}")
            self.stop_event.wait(timeout=self.flush_interval)

    def get_log_path(self) -> str:
        """返回日志文件本地完整路径"""
        return str(self.log_path)

    def _save_to_local(self) -> bool:
        """将新增日志内容追加到本地文件，返回是否有内容"""
        chunks = []
        log_path = self.log_path
        try:
            with self._lock:
                if not hasattr(self, '_pending_chunks'):
                    self._pending_chunks = []
                if not self._pending_chunks:
                    return False
                chunks = self._pending_chunks
                self._pending_chunks = []
                log_path = self.log_path

            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.writelines(chunks)
            return True
        except Exception as e:
            if chunks:
                with self._lock:
                    if not hasattr(self, '_pending_chunks'):
                        self._pending_chunks = []
                    self._pending_chunks = chunks + self._pending_chunks
            logger.error(f"保存本地日志失败: task_id={self.task_id}, error={e}")
            return False
