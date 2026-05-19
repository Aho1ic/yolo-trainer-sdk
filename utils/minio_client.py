#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinIO客户端工具类
用于处理文件上传、下载和管理
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple
from io import BytesIO
import traceback

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    print("警告: minio未安装，MinIO功能将不可用。运行: pip install minio")

logger = logging.getLogger(__name__)


class MinIOManager:
    """MinIO文件管理器"""
    
    def __init__(self, config: dict = None):
        """
        初始化MinIO客户端
        
        Args:
            config: MinIO配置字典，包含endpoint, access_key, secret_key等
        """
        if not MINIO_AVAILABLE:
            logger.warning("MinIO模块未安装，MinIO功能不可用")
            self.client = None
            return
            
        if config is None:
            # 使用默认配置
            from config import TrainerConfig
            trainer_config = TrainerConfig()
            config = trainer_config.get_minio_config()
        
        try:
            # 创建MinIO客户端
            self.client = Minio(
                config['endpoint'],
                access_key=config['access_key'],
                secret_key=config['secret_key'],
                secure=config.get('secure', False)
            )
            
            self.bucket_name = config.get('bucket', 'algorithm')
            self.base_path = config.get('base_path', 'trainer')
            
            # 确保bucket存在
            self._ensure_bucket_exists()
            
            logger.info(f"MinIO客户端初始化成功: {config['endpoint']}/{self.bucket_name}/{self.base_path}")
            
        except Exception as e:
            logger.error(f"MinIO客户端初始化失败: {str(e)}")
            self.client = None
    
    def _ensure_bucket_exists(self):
        """确保bucket存在，如果不存在则创建"""
        if not self.client:
            return
            
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"创建bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"检查/创建bucket失败: {str(e)}")
    
    def upload_file(
        self, 
        local_path: Union[str, Path], 
        minio_path: Optional[str] = None,
        folder_type: str = None
    ) -> Tuple[bool, str]:
        """
        上传文件到MinIO
        
        Args:
            local_path: 本地文件路径
            minio_path: MinIO中的路径，如果为None则自动生成
            folder_type: 文件夹类型(original_dataset/predict_data/train_data/train_results/training_logs)
            
        Returns:
            (成功标志, MinIO路径或错误信息)
        """
        if not self.client:
            return False, "MinIO客户端未初始化"
        
        try:
            local_path = Path(local_path)
            
            if not local_path.exists():
                return False, f"文件不存在: {local_path}"
            
            # 生成MinIO路径
            if minio_path is None:
                if folder_type:
                    # 使用指定的文件夹类型
                    minio_path = f"{self.base_path}/{folder_type}/{local_path.name}"
                else:
                    # 尝试从本地路径推断文件夹类型
                    path_str = str(local_path)
                    if 'original_dataset' in path_str:
                        folder_type = 'original_dataset'
                    elif 'predict_data' in path_str:
                        folder_type = 'predict_data'
                    elif 'train_data' in path_str:
                        folder_type = 'train_data'
                    elif 'train_results' in path_str:
                        folder_type = 'train_results'
                    elif 'training_logs' in path_str:
                        folder_type = 'training_logs'
                    else:
                        folder_type = 'others'
                    
                    # 保持原有的子路径结构
                    if folder_type in path_str:
                        # 获取folder_type之后的相对路径
                        parts = path_str.split(folder_type)
                        if len(parts) > 1:
                            relative_path = parts[-1].lstrip(os.sep).lstrip('/')
                            minio_path = f"{self.base_path}/{folder_type}/{relative_path}"
                        else:
                            minio_path = f"{self.base_path}/{folder_type}/{local_path.name}"
                    else:
                        minio_path = f"{self.base_path}/{folder_type}/{local_path.name}"
            
            # 确保路径使用正斜杠
            minio_path = minio_path.replace(os.sep, '/')
            
            # 获取文件大小
            file_stat = local_path.stat()
            
            # 上传文件
            with open(local_path, 'rb') as file_data:
                self.client.put_object(
                    self.bucket_name,
                    minio_path,
                    file_data,
                    file_stat.st_size
                )
            
            # 上传成功日志改为debug级别，避免输出过多
            logger.debug(f"文件上传成功: {local_path} -> {self.bucket_name}/{minio_path}")
            return True, f"{self.bucket_name}/{minio_path}"
            
        except Exception as e:
            error_msg = f"文件上传失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg
    
    def upload_directory(
        self,
        local_dir: Union[str, Path],
        minio_dir: Optional[str] = None,
        folder_type: str = None,
        recursive: bool = True
    ) -> Tuple[int, int, List[str]]:
        """
        上传整个目录到MinIO
        
        Args:
            local_dir: 本地目录路径
            minio_dir: MinIO中的目录路径
            folder_type: 文件夹类型
            recursive: 是否递归上传子目录
            
        Returns:
            (成功数量, 失败数量, 失败文件列表)
        """
        if not self.client:
            return 0, 0, ["MinIO客户端未初始化"]
        
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            return 0, 0, [f"目录不存在: {local_dir}"]
        
        success_count = 0
        fail_count = 0
        failed_files = []
        
        # 获取所有文件
        if recursive:
            files = list(local_dir.rglob('*'))
        else:
            files = list(local_dir.glob('*'))
        
        # 过滤出文件（排除目录）
        files = [f for f in files if f.is_file()]
        
        for file_path in files:
            # 计算相对路径
            relative_path = file_path.relative_to(local_dir)
            
            # 生成MinIO路径
            if minio_dir:
                minio_path = f"{minio_dir}/{relative_path}"
            else:
                if folder_type:
                    minio_path = f"{self.base_path}/{folder_type}/{relative_path}"
                else:
                    minio_path = None
            
            # 上传文件
            success, result = self.upload_file(file_path, minio_path, folder_type)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append(str(file_path))
        
        logger.info(f"目录上传完成: 成功{success_count}个，失败{fail_count}个")
        return success_count, fail_count, failed_files
    
    def download_file(
        self,
        minio_path: str,
        local_path: Union[str, Path]
    ) -> Tuple[bool, str]:
        """
        从MinIO下载文件
        
        Args:
            minio_path: MinIO中的文件路径
            local_path: 本地保存路径
            
        Returns:
            (成功标志, 本地路径或错误信息)
        """
        if not self.client:
            return False, "MinIO客户端未初始化"
        
        try:
            local_path = Path(local_path)
            
            # 确保本地目录存在
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载文件
            self.client.fget_object(
                self.bucket_name,
                minio_path,
                str(local_path)
            )
            
            # 下载成功日志改为debug级别，避免输出过多
            logger.debug(f"文件下载成功: {self.bucket_name}/{minio_path} -> {local_path}")
            return True, str(local_path)
            
        except Exception as e:
            error_msg = f"文件下载失败: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def list_objects(
        self,
        prefix: str = None,
        recursive: bool = False
    ) -> List[str]:
        """
        列出MinIO中的对象
        
        Args:
            prefix: 前缀过滤
            recursive: 是否递归列出
            
        Returns:
            对象路径列表
        """
        if not self.client:
            return []
        
        try:
            if prefix is None:
                prefix = self.base_path
            
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=prefix,
                recursive=recursive
            )
            
            object_list = [obj.object_name for obj in objects]
            return object_list
            
        except Exception as e:
            logger.error(f"列出对象失败: {str(e)}")
            return []
    
    def delete_object(self, minio_path: str) -> bool:
        """
        删除MinIO中的对象
        
        Args:
            minio_path: MinIO中的对象路径
            
        Returns:
            是否成功
        """
        if not self.client:
            return False
        
        try:
            self.client.remove_object(self.bucket_name, minio_path)
            logger.info(f"对象删除成功: {self.bucket_name}/{minio_path}")
            return True
        except Exception as e:
            logger.error(f"对象删除失败: {str(e)}")
            return False
    
    def get_object_url(
        self,
        minio_path: str,
        expires: int = 3600
    ) -> Optional[str]:
        """
        获取对象的临时访问URL
        
        Args:
            minio_path: MinIO中的对象路径
            expires: URL过期时间（秒）
            
        Returns:
            访问URL或None
        """
        if not self.client:
            return None
        
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                self.bucket_name,
                minio_path,
                expires=timedelta(seconds=expires)
            )
            return url
        except Exception as e:
            logger.error(f"获取对象URL失败: {str(e)}")
            return None
    
    def upload_file_async(
        self,
        local_path: Union[str, Path],
        minio_path: Optional[str] = None,
        folder_type: str = None
    ):
        """
        异步上传文件到MinIO（在后台线程执行）
        
        Args:
            local_path: 本地文件路径
            minio_path: MinIO中的路径
            folder_type: 文件夹类型
        """
        import threading
        
        def _upload():
            try:
                success, result = self.upload_file(local_path, minio_path, folder_type)
                if not success:
                    logger.error(f"异步上传失败: {result}")
            except Exception as e:
                logger.error(f"异步上传异常: {str(e)}")
        
        thread = threading.Thread(target=_upload)
        thread.daemon = True
        thread.start()
        logger.info(f"启动异步上传任务: {local_path}")


# 单例模式的全局MinIO管理器
_minio_manager = None

def get_minio_manager() -> MinIOManager:
    """获取全局MinIO管理器实例"""
    global _minio_manager
    if _minio_manager is None:
        try:
            from config import TrainerConfig
        except ImportError:
            import sys
            from pathlib import Path
            # 添加父目录到sys.path
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            from config import TrainerConfig
        
        config = TrainerConfig()
        _minio_manager = MinIOManager(config.get_minio_config())
    return _minio_manager
