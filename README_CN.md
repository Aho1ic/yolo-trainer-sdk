# YOLO Trainer SDK

[English](README.md)

一个生产级的 YOLOv8 模型训练、数据集管理和模型转换 REST API 平台。

## 功能特性

- **模型训练** - YOLOv8 目标检测和实例分割训练，支持单卡/多卡（DDP）
- **数据集管理** - JSON 转 TXT 标签格式、YAML 配置生成、标签统计、批量标签修改
- **模型转换** - PT 模型转换为 ONNX/RKNN/BModel，支持边缘部署（BM1684、BM1684X、RK3588）
- **任务管理** - 启动/停止训练任务，通过 Redis 实时同步训练进度
- **日志流式上传** - 训练日志实时上传到 MinIO/RustFS 对象存储
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
│   ├── path_handler.py    # MinIO/本地路径解析
│   ├── minio_client.py    # MinIO/RustFS 客户端
│   ├── minio_log_uploader.py
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
- MinIO 或 RustFS（S3 兼容对象存储）
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
# 编辑 .env 文件，填入数据库、MinIO 和 Redis 的连接信息
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
| POST | `/algorithm/datasets/stats` | 获取数据集统计信息 |
| POST | `/algorithm/datasets/batch_modify_labels` | 批量重命名/删除标签 |
| GET | `/algorithm/datasets/build_status` | 查询数据集构建状态 |
| POST | `/algorithm/models/convert` | 将 PT 模型转换为目标平台格式 |

## 技术栈

- **Web 框架**: Flask + flask-cors
- **训练引擎**: Ultralytics YOLOv8
- **数据库**: MySQL（pymysql + SQLAlchemy 连接池）
- **缓存**: Redis
- **对象存储**: MinIO/RustFS（S3 API）
- **模型导出**: ONNX、RKNN、BModel（算能 Sophon）

## 许可证

私有项目 - 保留所有权利。
