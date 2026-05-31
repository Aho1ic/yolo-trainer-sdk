-- 为 pose 数据集关键点顺序和原始数据 hash 规则增加持久化字段。
-- kpt_num 保存 pose 关键点数量 K，kpt_labels 保存 point 标签的规范顺序(JSON数组字符串)。
-- original_dataset_hash_schema 保存当前 hash 规则版本，避免旧规则 hash 误触发跳过。
--
-- 幂等: MySQL 的 ADD COLUMN 不支持 IF NOT EXISTS,这里用 information_schema 判断后
--       动态拼 DDL,重复执行不会报错。
--
-- 注意: train_api.py 的 get_dataset_full_info/get_dataset_json2txt_info/get_dataset_info
--       都把 kpt_num 写进了 SELECT 且没有列缺失降级。该列必须存在,否则这些查询会整体
--       抛错并导致 /stats、/yaml、/json2txt、/train 全部判定"未找到数据集信息"。

SET @ddl := (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE `train_dataset` ADD COLUMN `kpt_num` INT NOT NULL DEFAULT 0 COMMENT ''pose关键点数量K(非pose数据集为0)''',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'train_dataset'
      AND COLUMN_NAME = 'kpt_num'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @ddl := (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE `train_dataset` ADD COLUMN `kpt_labels` TEXT NULL DEFAULT NULL COMMENT ''pose关键点标签顺序(JSON数组字符串,与kpt_num对应)''',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'train_dataset'
      AND COLUMN_NAME = 'kpt_labels'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @ddl := (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE `train_dataset` ADD COLUMN `original_dataset_hash_schema` VARCHAR(64) NULL DEFAULT NULL COMMENT ''original_dataset_hash计算规则版本''',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'train_dataset'
      AND COLUMN_NAME = 'original_dataset_hash_schema'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
