# YOLO Trainer SDK

[中文文档](README_CN.md)

A production-ready REST API platform for YOLOv8 model training, dataset management, and model conversion.

## Features

- **Model Training** - YOLOv8 object detection and instance segmentation with single/multi-GPU support
- **Dataset Management** - JSON-to-TXT label conversion, YAML generation, label statistics, and batch label modification
- **Model Conversion** - PT to ONNX/RKNN/BModel conversion for edge deployment (BM1684, BM1684X, RK3588)
- **Task Management** - Start/stop training tasks, real-time progress monitoring via Redis
- **Log Streaming** - Training logs uploaded to MinIO/RustFS in real-time
- **Connection Pooling** - SQLAlchemy-based database connection pool for high concurrency

## Architecture

```
trainer_v3/
├── train_api.py           # Main API entry point (Flask)
├── train_method/
│   └── trainer.py         # YOLOv8 training engine
├── api/
│   ├── validators.py      # Input validation functions
│   └── __init__.py
├── services/
│   ├── exceptions.py      # Custom exception hierarchy
│   └── __init__.py
├── database/
│   ├── manager.py         # SQLAlchemy connection pool base
│   └── __init__.py
├── config/
│   ├── config.py          # Configuration loader (env > .env > config.ini)
│   ├── config.ini.example # Config template
│   └── __init__.py
├── utils/
│   ├── logger.py          # Unified logging
│   ├── path_handler.py    # MinIO/local path resolution
│   ├── minio_client.py    # MinIO/RustFS client
│   ├── minio_log_uploader.py
│   ├── epoch_callback.py  # Training epoch callbacks
│   ├── json2txt.py        # Label format converter
│   ├── create_dataset_yaml.py
│   ├── get_dataset_detail.py
│   └── predict.py
├── pt2onnx/               # ONNX export utilities
└── rknn_model_zoo/        # RKNN conversion toolkit
```

## Quick Start

### Prerequisites

- Python 3.9+
- MySQL 5.7+
- Redis 6.0+
- MinIO or RustFS (S3-compatible object storage)
- CUDA-capable GPU (optional, for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Aho1ic/yolo-trainer-sdk.git
cd yolo-trainer-sdk

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment config
cp .env.example .env
# Edit .env with your database, MinIO, and Redis credentials
```

### Configuration

The application loads configuration in the following priority order:

1. **Environment variables** (highest priority)
2. **`.env` file** (via python-dotenv)
3. **`config/config.ini`** (fallback)

See `.env.example` for all available configuration options.

### Running

```bash
python train_api.py
```

The API server starts on `http://0.0.0.0:5000` by default.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/algorithm/train` | Start or stop a training task |
| POST | `/algorithm/train/log` | Get training logs |
| POST | `/algorithm/train/predict` | Run model prediction |
| POST | `/algorithm/train/build_yolo_dataset` | Build YOLO dataset from source |
| POST | `/algorithm/datasets/yaml` | Generate dataset YAML config |
| POST | `/algorithm/datasets/json2txt` | Convert JSON labels to TXT format |
| POST | `/algorithm/datasets/create` | Create dataset with train/val split |
| POST | `/algorithm/datasets/stats` | Get dataset statistics |
| POST | `/algorithm/datasets/batch_modify_labels` | Batch rename/delete labels |
| GET | `/algorithm/datasets/build_status` | Check dataset build status |
| POST | `/algorithm/models/convert` | Convert PT model to target platform |

## Tech Stack

- **Framework**: Flask + flask-cors
- **Training**: Ultralytics YOLOv8
- **Database**: MySQL (via pymysql + SQLAlchemy connection pool)
- **Cache**: Redis
- **Storage**: MinIO/RustFS (S3 API)
- **Model Export**: ONNX, RKNN, BModel (Sophon)

## License

Private - All rights reserved.
