#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练日志写入器
实现训练日志实时写入 NFS 挂载目录，并提供标准对象路径
"""

import threading
import logging
from io import StringIO
from typing import Optional
from pathlib import Path

from utils.path_handler import PathHandler

logger = logging.getLogger(__name__)


class MinIOLogUploader:
    """
    MinIO日志上传器
    
    负责将训练日志实时写入 NFS 挂载目录，并提供标准 minio:// 路径。
    """
    
    def __init__(self, task_id: str, upload_interval: float = 0.5):
        """
        初始化MinIO日志上传器
        
        Args:
            task_id: 训练任务ID
            upload_interval: 上传间隔（秒），默认0.5秒
        """
        self.task_id = task_id
        self.upload_interval = upload_interval
        self.log_buffer = StringIO()
        self.stop_event = threading.Event()
        self.upload_thread: Optional[threading.Thread] = None
        self.minio_path = f"{PathHandler.get_storage_base_path()}/train_log/{task_id}/training.log"
        self.local_log_path = PathHandler.resolve_storage_local_path(
            'train_log', task_id, 'training.log', require_exists=False
        )
        self._lock = threading.Lock()
        self._last_uploaded_position = 0
        
        # 初始化时创建目录
        self._ensure_log_dir_exists()
    
    def _ensure_log_dir_exists(self) -> None:
        """确保日志目录存在"""
        try:
            log_dir = Path(self.local_log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"日志目录已创建: {log_dir}")
        except Exception as e:
            logger.warning(f"创建日志目录失败: {e}")
    
    def set_local_save_path(self, path: str) -> None:
        """
        设置本地日志保存路径（可选，用于覆盖默认路径）
        
        Args:
            path: 本地保存路径（完整文件路径）
        """
        with self._lock:
            self.local_log_path = path
            minio_uri = PathHandler.convert_local_to_minio_path(path, require_within_root=False)
            if minio_uri:
                _, object_path = PathHandler.parse_minio_path(minio_uri)
                self.minio_path = object_path
            logger.info(f"日志保存路径已更新: local={path}, minio={self.minio_path}")
    
    def start(self) -> None:
        """启动上传线程"""
        if self.upload_thread is not None and self.upload_thread.is_alive():
            logger.warning(f"上传线程已在运行: task_id={self.task_id}")
            return
        
        # 启动时立即创建空日志文件
        self._create_empty_log_file()
        
        self.stop_event.clear()
        self.upload_thread = threading.Thread(
            target=self._upload_loop,
            name=f"MinIOLogUploader-{self.task_id}",
            daemon=True
        )
        self.upload_thread.start()
        logger.info(f"日志写入器已启动: task_id={self.task_id}, local_path={self.local_log_path}")
    
    def _create_empty_log_file(self) -> None:
        """创建空的日志文件"""
        try:
            local_dir = Path(self.local_log_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            # 创建空文件（如果不存在）
            with open(self.local_log_path, 'a', encoding='utf-8') as f:
                pass
            logger.info(f"日志文件已创建: {self.local_log_path}")
        except Exception as e:
            logger.error(f"创建日志文件失败: {e}")
    
    def stop(self) -> None:
        """停止上传线程，保存本地日志"""
        self.stop_event.set()
        
        if self.upload_thread is not None:
            self.upload_thread.join(timeout=5.0)
            if self.upload_thread.is_alive():
                logger.warning(f"上传线程未能在超时时间内停止: task_id={self.task_id}")
        
        # 最终保存日志到本地文件
        self._save_to_local()
        logger.info(f"日志已保存到本地: {self.local_log_path}")
        
        # NFS 直写模式下无需再上传对象存储
        if self._upload_to_minio():
            logger.info(f"日志对象路径已就绪: {self.get_minio_full_path()}")
        else:
            logger.warning(f"日志对象路径登记失败，但本地日志已保存: {self.local_log_path}")
        
        logger.info(f"日志写入器已停止: task_id={self.task_id}")
    
    def write(self, content: str) -> None:
        """
        写入日志内容到缓冲区
        
        Args:
            content: 日志内容
        """
        if not content:
            return
        
        with self._lock:
            self.log_buffer.write(content)
    
    def flush(self) -> None:
        """刷新缓冲区（兼容文件接口）"""
        pass
    
    def _upload_loop(self) -> None:
        """上传循环，每0.5秒保存一次本地日志"""
        save_count = 0
        while not self.stop_event.is_set():
            # 实时保存到本地文件
            try:
                saved = self._save_to_local()
                if saved:
                    save_count += 1
                    # 每20次保存输出一次日志（约10秒一次）
                    if save_count % 20 == 0:
                        logger.info(f"日志已保存到本地: {self.local_log_path}")
            except Exception as e:
                logger.error(f"本地日志保存失败: task_id={self.task_id}, error={e}")
            
            # 等待指定间隔或直到收到停止信号
            self.stop_event.wait(timeout=self.upload_interval)
    
    def _upload_to_minio(self) -> bool:
        """
        NFS 直写模式下仅校验日志文件已经落盘
        
        Returns:
            bool: 路径是否有效
        """
        try:
            return Path(self.local_log_path).exists()
        except Exception as e:
            logger.error(f"日志路径校验失败: task_id={self.task_id}, error={e}")
            return False
    
    def get_minio_full_path(self) -> str:
        """
        获取完整的MinIO路径
        
        Returns:
            str: 完整的MinIO路径，格式为 minio://bucket/path
        """
        return PathHandler.build_minio_uri(PathHandler.get_storage_bucket(), self.minio_path)
    
    def _save_to_local(self) -> bool:
        """
        将日志保存到本地文件
        
        Returns:
            bool: 保存是否成功（有内容写入返回True）
        """
        try:
            with self._lock:
                current_content = self.log_buffer.getvalue()
            
            # 确保目录存在
            local_dir = Path(self.local_log_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入本地文件（覆盖模式，每次写入完整内容）
            with open(self.local_log_path, 'w', encoding='utf-8') as f:
                f.write(current_content)
            
            return bool(current_content)  # 有内容返回True
            
        except Exception as e:
            logger.error(f"保存本地日志失败: task_id={self.task_id}, error={e}")
            return False
