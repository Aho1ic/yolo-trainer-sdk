# YOLO Trainer SDK

[English](README.md)

一个生产级的 YOLOv8 模型训练、数据集管理和模型转换 REST API 平台。

## 功能特性

- **模型训练** - YOLOv8 目标检测、实例分割、YOLO11 关键点（pose）训练，支持单卡/多卡（DDP）
- **数据集管理** - JSON 转 TXT 标签格式、YAML 配置生成、标签统计、批量标签修改
- **模型转换** - PT 模型转换为 ONNX/RKNN/BModel，支持边缘部署（BM1684、BM1684X、RK3588）
- **任务管理** - 启动/停止训练任务，通过 Redis 实时同步训练进度
- **日志流式写入** - 训练日志实时写入本地数据目录
- **数据库连接池** - 基于 SQLAlchemy 的连接池，支持高并发访问

## 项目结构

```
trainer_v3/
├── train_api.py           # 主 API 入口（Flask）
├── train_method/
│   └── trainer.py         # YOLOv8 训练引擎
├── api/
│   ├── validators.py      # 输入验证函数
│   └── __init__.py
├── services/
│   ├── exceptions.py      # 自定义异常体系
│   └── __init__.py
├── database/
│   ├── manager.py         # SQLAlchemy 连接池基类
│   └── __init__.py
├── config/
│   ├── config.py          # 配置加载器（环境变量 > .env > config.ini）
│   ├── config.ini.example # 配置模板
│   └── __init__.py
├── utils/
│   ├── logger.py          # 统一日志配置
│   ├── path_handler.py    # 本地路径解析
│   ├── training_log_writer.py # 训练日志写入器（写入本地数据目录）
│   ├── epoch_callback.py  # 训练 Epoch 回调
│   ├── json2txt.py        # 标签格式转换器
│   ├── create_dataset_yaml.py
│   ├── get_dataset_detail.py
│   └── predict.py
├── pt2onnx/               # ONNX 导出工具
└── rknn_model_zoo/        # RKNN 转换工具集
```

## 快速开始

### 环境要求

- Python 3.9+
- MySQL 5.7+
- Redis 6.0+
- 本地数据目录（默认 `/data/Sucai1/algorithm/trainer`，可通过 `TRAINER_DATA_ROOT` 配置）
- CUDA GPU（可选，用于训练）

### 安装

```bash
# 克隆仓库
git clone https://github.com/Aho1ic/yolo-trainer-sdk.git
cd yolo-trainer-sdk

# 安装依赖
pip install -r requirements.txt

# 复制并编辑环境配置
cp .env.example .env
# 编辑 .env 文件，填入数据库、Redis 与 TRAINER_DATA_ROOT 等配置
```

### 配置说明

应用按以下优先级加载配置：

1. **环境变量**（最高优先级）
2. **`.env` 文件**（通过 python-dotenv 加载）
3. **`config/config.ini`**（回退配置）

所有可配置项请参考 `.env.example` 文件。

### 启动服务

```bash
python train_api.py
```

API 服务默认启动在 `http://0.0.0.0:5000`。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/algorithm/train` | 启动或停止训练任务 |
| POST | `/algorithm/train/log` | 获取训练日志 |
| POST | `/algorithm/train/predict` | 运行模型预测 |
| POST | `/algorithm/train/build_yolo_dataset` | 从源数据构建 YOLO 数据集 |
| POST | `/algorithm/datasets/yaml` | 生成数据集 YAML 配置 |
| POST | `/algorithm/datasets/json2txt` | JSON 标签转 TXT 格式 |
| POST | `/algorithm/datasets/create` | 创建数据集（含训练/验证集划分） |
| POST | `/algorithm/datasets/stats` | 获取数据集统计信息(根据 `train_dataset.original_dataset_hash` 比对原始目录,未变更则直接跳过) |
| POST | `/algorithm/datasets/batch_modify_labels` | 批量重命名/删除标签 |
| GET | `/algorithm/datasets/build_status` | 查询数据集构建状态 |
| POST | `/algorithm/models/convert` | 将 PT 模型转换为目标平台格式 |

## 技术栈

- **Web 框架**: Flask + flask-cors
- **训练引擎**: Ultralytics YOLOv8
- **数据库**: MySQL（pymysql + SQLAlchemy 连接池）
- **缓存**: Redis
- **数据存储**: 本地文件系统（`TRAINER_DATA_ROOT`）
- **模型导出**: ONNX、RKNN、BModel（算能 Sophon）

## 许可证

私有项目 - 保留所有权利。
