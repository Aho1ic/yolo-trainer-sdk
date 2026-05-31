# -*- coding:utf-8 -*-
"""
Trainer 配置模块
优先从环境变量/.env文件读取，回退到 config.ini 文件
"""
import os
import configparser
from pathlib import Path


def _load_dotenv_file() -> None:
    """加载 .env 文件（如果存在）

    使用 setdefault：真实环境变量优先于 .env，保持
    "环境变量 > .env > config.ini" 的优先级。
    """
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / '.env'
    if not env_file.exists():
        return

    with open(env_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)


def _get_env(key: str, fallback=None, cast_type=None):
    """从环境变量获取配置值，支持类型转换"""
    value = os.environ.get(key)
    if value is None:
        return fallback
    if cast_type is None:
        return value
    try:
        if cast_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        return cast_type(value)
    except (ValueError, TypeError):
        return fallback


class TrainerConfig:

    def __init__(self, path: str = '') -> None:
        # 首先加载 .env 文件
        _load_dotenv_file()

        # 尝试读取 config.ini 作为回退
        cf = configparser.ConfigParser()
        if path == '':
            BASE_DIR = Path(__file__).resolve().parents[0]
            path = BASE_DIR / 'config.ini'
        cf.read(path, encoding='utf-8')

        def _get(key: str, env_key: str, fallback=None, cast_type=None):
            """优先从环境变量读取，回退到 config.ini"""
            env_value = _get_env(env_key, fallback=None, cast_type=cast_type)
            if env_value is not None:
                return env_value
            try:
                if cast_type == bool:
                    return cf.getboolean('trainer_config', key)
                if cast_type == int:
                    return int(cf.get('trainer_config', key))
                return cf.get('trainer_config', key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return fallback

        # 数据库配置
        self.db_host = _get('db_host', 'TRAINER_DB_HOST', fallback='localhost')
        self.db_port = _get('db_port', 'TRAINER_DB_PORT', fallback=3306, cast_type=int)
        self.db_user = _get('db_user', 'TRAINER_DB_USER', fallback='root')
        self.db_passwd = _get('db_passwd', 'TRAINER_DB_PASSWORD', fallback='')
        self.db_db = _get('db_db', 'TRAINER_DB_NAME', fallback='algorithm_trainer1')
        self.db_charset = _get('db_charset', 'TRAINER_DB_CHARSET', fallback='utf8mb4')

        # 训练配置
        self.log_level = _get('log_level', 'TRAINER_LOG_LEVEL', fallback='INFO')
        self.max_concurrent_tasks = _get('max_concurrent_tasks', 'TRAINER_MAX_CONCURRENT_TASKS', fallback=5, cast_type=int)
        self.model_preload = _get('model_preload', 'TRAINER_MODEL_PRELOAD', fallback=False, cast_type=bool)

        # 本地数据根目录
        self.data_root = _get('data_root', 'TRAINER_DATA_ROOT', fallback='/data/Sucai1/training/algorithm/trainer')

        # 数据处理完成回调地址
        self.process_callback_base_url = _get(
            'process_callback_base_url',
            'TRAINER_PROCESS_CALLBACK_BASE_URL',
            fallback=''
        )

        # Redis配置
        self.redis_host = _get('redis_host', 'TRAINER_REDIS_HOST', fallback='localhost')
        self.redis_port = _get('redis_port', 'TRAINER_REDIS_PORT', fallback=6379, cast_type=int)
        self.redis_database = _get('redis_database', 'TRAINER_REDIS_DATABASE', fallback=11, cast_type=int)
        self.redis_password = _get('redis_password', 'TRAINER_REDIS_PASSWORD', fallback='')
    
    def get_db_config(self):
        """返回数据库连接配置字典"""
        return {
            'host': self.db_host,
            'port': self.db_port,
            'user': self.db_user,
            'password': self.db_passwd,
            'database': self.db_db,
            'charset': self.db_charset
        }
    
    def get_redis_config(self):
        """返回Redis配置字典"""
        return {
            'host': self.redis_host,
            'port': self.redis_port,
            'db': self.redis_database,
            'password': self.redis_password
        }

    def get_data_root(self):
        """返回本地数据根目录路径"""
        return self.data_root

    def get_process_callback_base_url(self):
        """返回数据处理完成回调的基础地址"""
        return self.process_callback_base_url
