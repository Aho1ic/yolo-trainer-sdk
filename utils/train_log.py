# -*- coding: UTF-8 -*-
"""
@Project : trainer 
@Desc    : 训练日志配置
"""
import logging
import re


class EmojiFilter(logging.Filter):
    # 过滤有表情的日志
    def __init__(self):
        # 使用正则表达式匹配表情符号
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
    

def setup_task_logger(path: str):
    # 配置日志文件路径
    log_filename = f"{path}/train_log.log"
    
    # 获取YOLO库的记录器
    yolo_logger = logging.getLogger('ultralytics')
    yolo_logger.setLevel(logging.INFO)
    
    # 清空以前的处理器，防止重复日志
    yolo_logger.handlers.clear()

    # 添加文件处理器
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(EmojiFilter())
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    yolo_logger.addHandler(file_handler)
