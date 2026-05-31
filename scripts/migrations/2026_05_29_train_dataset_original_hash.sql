-- 为 train_dataset 增加 original_dataset_hash 字段
-- 用途: 记录 /data/Sucai1/training/algorithm/trainer/original_dataset/{dataset_id}
--       原始 images/json 内容哈希,/algorithm/datasets/stats 接口在执行前会比对该字段,
--       命中则跳过整个统计流程。
-- 字段长度按 sha256 十六进制摘要 64 字符设置。
--
-- 幂等: MySQL 的 ADD COLUMN 不支持 IF NOT EXISTS,这里用 information_schema 判断后
--       动态拼 DDL,重复执行不会报错。

SET @ddl := (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE `train_dataset` ADD COLUMN `original_dataset_hash` VARCHAR(64) NULL DEFAULT NULL COMMENT ''original_dataset/{dataset_id} 原始images/json内容哈希(sha256),用于 stats 跳过判断''',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'train_dataset'
      AND COLUMN_NAME = 'original_dataset_hash'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
