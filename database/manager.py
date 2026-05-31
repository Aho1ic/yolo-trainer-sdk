# -*- coding:utf-8 -*-
"""
数据库管理模块
使用 SQLAlchemy 连接池管理 MySQL 连接
"""
import logging
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import QueuePool
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("警告: sqlalchemy 未安装，将使用 pymysql 直连")

try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器，支持连接池"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据库管理器

        Args:
            config: 数据库配置字典，包含 host, port, user, password, database, charset
        """
        self.config = config
        self._engine = None
        self._init_connection_pool()

    def _init_connection_pool(self):
        """初始化连接池"""
        if SQLALCHEMY_AVAILABLE:
            try:
                # 构建 SQLAlchemy 连接 URL
                user = self.config.get('user', 'root')
                password = self.config.get('password', '')
                host = self.config.get('host', 'localhost')
                port = self.config.get('port', 3306)
                database = self.config.get('database', '')
                charset = self.config.get('charset', 'utf8mb4')

                url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}"

                self._engine = create_engine(
                    url,
                    poolclass=QueuePool,
                    pool_size=5,           # 连接池大小
                    max_overflow=10,       # 最大溢出连接数
                    pool_timeout=30,       # 获取连接超时时间
                    pool_recycle=3600,     # 连接回收时间（秒）
                    pool_pre_ping=True,    # 使用前检测连接是否有效
                    echo=False             # 不打印 SQL 语句
                )
                logger.info(f"SQLAlchemy 连接池初始化成功: {host}:{port}/{database}")
                return
            except Exception as e:
                logger.warning(f"SQLAlchemy 连接池初始化失败，回退到 pymysql: {e}")

        # 回退到直接使用 pymysql
        self._engine = None
        logger.info("使用 pymysql 直连模式")

    @contextmanager
    def get_connection(self):
        """
        获取数据库连接（上下文管理器）

        使用方式:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(...)
        """
        if self._engine:
            # 使用 SQLAlchemy 连接池
            connection = self._engine.raw_connection()
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        else:
            # 回退到 pymysql 直连
            connection = None
            try:
                connect_config = {**self.config, 'connect_timeout': 10}
                connection = pymysql.connect(**connect_config)
                yield connection
                connection.commit()
            except Exception:
                if connection:
                    connection.rollback()
                raise
            finally:
                if connection:
                    connection.close()

    @contextmanager
    def get_cursor(self, cursor_class=None):
        """
        获取数据库游标（上下文管理器）

        Args:
            cursor_class: 游标类，默认为 pymysql.cursors.DictCursor

        使用方式:
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM table WHERE id = %s", (id,))
                result = cursor.fetchone()
        """
        if cursor_class is None:
            cursor_class = pymysql.cursors.DictCursor if PYMYSQL_AVAILABLE else None

        with self.get_connection() as connection:
            cursor = connection.cursor(cursor_class)
            try:
                yield cursor
            finally:
                cursor.close()

    def execute_query(self, sql: str, params: tuple = None) -> Optional[List[Dict]]:
        """
        执行查询语句

        Args:
            sql: SQL 查询语句
            params: 查询参数

        Returns:
            查询结果列表，失败返回 None
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"执行查询失败: {sql}, error={str(e)}")
            return None

    def execute_query_one(self, sql: str, params: tuple = None) -> Optional[Dict]:
        """
        执行查询语句并返回单条结果

        Args:
            sql: SQL 查询语句
            params: 查询参数

        Returns:
            单条查询结果，失败返回 None
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"执行查询失败: {sql}, error={str(e)}")
            return None

    def execute_update(self, sql: str, params: tuple = None) -> bool:
        """
        执行更新语句

        Args:
            sql: SQL 更新语句
            params: 更新参数

        Returns:
            是否成功
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, params)
                return True
        except Exception as e:
            logger.error(f"执行更新失败: {sql}, error={str(e)}")
            return False

    def execute_insert(self, sql: str, params: tuple = None) -> Optional[int]:
        """
        执行插入语句

        Args:
            sql: SQL 插入语句
            params: 插入参数

        Returns:
            插入的行 ID，失败返回 None
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"执行插入失败: {sql}, error={str(e)}")
            return None

    def close(self):
        """关闭连接池"""
        if self._engine:
            self._engine.dispose()
            logger.info("数据库连接池已关闭")
