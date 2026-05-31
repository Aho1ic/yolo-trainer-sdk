# -*- coding:utf-8 -*-
"""
同步训练结果 CSV 到数据库 train_chart 表
"""
import csv
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrainerConfig
from database.manager import DatabaseManager

# ==================== 配置区域 ====================
# 本地 CSV 文件路径
CSV_FILE_PATH = "/Users/macbook/Downloads/results.csv"

# 训练任务 ID（需要根据实际情况修改）
TRAIN_TASK_ID = 317305170578898947
# ================================================


# CSV 列名 -> 数据库列名 映射
COLUMN_MAPPING = {
    'epoch': 'epoch',
    'time': 'time',
    'train/box_loss': 'train_box_loss',
    'train/cls_loss': 'train_cls_loss',
    'train/dfl_loss': 'train_dfl_loss',
    'metrics/precision(B)': 'metrics_precision_B',
    'metrics/recall(B)': 'metrics_recall_B',
    'metrics/mAP50(B)': 'metrics_mAP50_B',
    'metrics/mAP50-95(B)': 'metrics_mAP50_95_B',
    'val/box_loss': 'val_box_loss',
    'val/cls_loss': 'val_cls_loss',
    'val/dfl_loss': 'val_dfl_loss',
    'lr/pg0': 'lr_pg0',
    'lr/pg1': 'lr_pg1',
    'lr/pg2': 'lr_pg2',
}

# 数据库列顺序（对应 INSERT 语句）
DB_COLUMNS = [
    'train_task_id', 'epoch', 'time',
    'train_box_loss', 'train_cls_loss', 'train_dfl_loss',
    'metrics_precision_B', 'metrics_recall_B', 'metrics_mAP50_B', 'metrics_mAP50_95_B',
    'val_box_loss', 'val_cls_loss', 'val_dfl_loss',
    'lr_pg0', 'lr_pg1', 'lr_pg2',
]


def read_csv(file_path: str) -> list[dict]:
    """读取 CSV 文件并转换列名"""
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapped_row = {}
            for csv_col, db_col in COLUMN_MAPPING.items():
                value = row.get(csv_col)
                if value is not None and value != '':
                    mapped_row[db_col] = float(value) if db_col != 'epoch' else int(value)
                else:
                    mapped_row[db_col] = None
            rows.append(mapped_row)
    return rows


def sync_to_db(rows: list[dict], train_task_id: int):
    """将数据同步到数据库"""
    config = TrainerConfig()
    db = DatabaseManager(config.get_db_config())

    # 构建 INSERT SQL（使用 REPLACE INTO 支持重复插入）
    placeholders = ', '.join(['%s'] * len(DB_COLUMNS))
    columns = ', '.join(DB_COLUMNS)
    sql = f"REPLACE INTO train_chart ({columns}) VALUES ({placeholders})"

    success_count = 0
    fail_count = 0

    with db.get_cursor() as cursor:
        for row in rows:
            try:
                params = tuple(row.get(col) for col in DB_COLUMNS)
                # 第一列是 train_task_id，需要替换为实际值
                params = (train_task_id,) + params[1:]
                cursor.execute(sql, params)
                success_count += 1
            except Exception as e:
                print(f"插入失败 epoch={row.get('epoch')}: {e}")
                fail_count += 1

    return success_count, fail_count


def main():
    csv_path = Path(CSV_FILE_PATH)
    if not csv_path.exists():
        print(f"错误: CSV 文件不存在: {CSV_FILE_PATH}")
        return

    print(f"读取 CSV 文件: {CSV_FILE_PATH}")
    rows = read_csv(CSV_FILE_PATH)
    print(f"共读取 {len(rows)} 条记录")

    if not rows:
        print("CSV 文件为空，无需同步")
        return

    print(f"开始同步到数据库 (train_task_id={TRAIN_TASK_ID})...")
    success, fail = sync_to_db(rows, TRAIN_TASK_ID)

    print(f"同步完成: 成功 {success} 条, 失败 {fail} 条")


if __name__ == '__main__':
    main()
