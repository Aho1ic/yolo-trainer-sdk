# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## 项目概览

这是一个基于 Flask 的 YOLOv8 训练与数据集处理服务。主入口是 `train_api.py`，它负责把训练、数据集处理、推理和模型转换串成统一的 REST API。代码运行时强依赖本地文件系统、MySQL 和 Redis，配置优先级是：环境变量 > `.env` > `config/config.ini`。

## 常用命令

### 启动服务
```bash
python train_api.py
```
服务默认监听 `0.0.0.0:12241`。

### 运行仓库自检
```bash
python test_all.py
```
这会生成 `test_report.md`。当前根目录没有独立的 pytest 单测入口，这个脚本是主要的整体验证方式。

### 单文件快速检查
```bash
python -m py_compile train_api.py train_method/trainer.py utils/json2txt.py
```
适合做单文件语法检查或在修改后快速确认没有语法错误。

### 常用工具脚本
```bash
python utils/json2txt.py --duty_type detect --original_train_data_address /path/to/data --labels duckweed ship
python utils/get_dataset_detail.py --original_train_data_address /path/to/data
python utils/split_dataset.py --dataset_id 123 --original_train_data_address /path/to/data --train_val_ratio 8:2
python utils/predict.py --dataset_address /path/to/images --model_address /path/to/model.pt --duty_type detect
python utils/pt2rknn.py --model_address /path/to/model.pt --duty_type detect --onnx_only
```

### `pt2onnx/` 子项目
这个目录是嵌入式的 Ultralytics 导出树，有自己的测试和工具配置。
```bash
cd pt2onnx
pip install -e .[dev]
pytest tests/test_cli.py::test_name
ruff check .
ruff format .
isort .
```

### 一次性同步脚本
```bash
python scripts/sync_train_chart.py
```
这个脚本里有硬编码的 CSV 路径和任务 ID，运行前先改常量。

## 高层架构

### 1. `train_api.py` 是编排层
它集中承载大部分 API 路由：
- 训练启动/停止
- 数据集 YAML 生成、JSON 转 TXT、数据集统计、批量标签修改、数据集切分
- 模型转换
- 推理预测

长耗时任务通常通过线程或子进程异步执行，返回统一的 JSON 响应格式 `{code, data}`。

### 2. 配置集中在 `config/`
`TrainerConfig` 统一读取数据库、Redis、日志级别、并发数和数据根目录。路径和存储行为都围绕 `TRAINER_DATA_ROOT` 展开，优先使用本地路径。

### 3. 数据库层在 `database/`
`DatabaseManager` 封装 MySQL 连接：优先使用 SQLAlchemy 连接池，失败时回退到 `pymysql` 直连。不要在业务层直接散写连接逻辑。

### 4. 输入校验和异常在 `api/`、`services/`
- `api/validators.py` 负责参数清洗、ID 校验和 JSON 请求体校验
- `services/exceptions.py` 定义业务异常类型，并由 `train_api.py` 统一转换成 API 响应

### 5. 训练核心在 `train_method/trainer.py`
这里封装 Ultralytics YOLO 训练流程，并把日志、结果路径和 epoch 进度同步到本地文件、数据库和 Redis。`utils/training_log_writer.py`、`utils/epoch_callback.py` 和它一起工作。

任务类型由 `dutyType` 字段决定：`detect`/`segment` 走 YOLOv8（`yolov8{n,s,m,l,x}.pt`/`-seg.pt`），`pose` 走 YOLO11（`yolo11{n,s,m,l,x}-pose.pt`，预训练权重放在 `pre_model/`）。pose 任务额外依赖 `train_dataset.kpt_num`，dataset.yaml 需要写出 `kpt_shape: [K, 3]`，TXT 标签格式是 `cls cx cy w h x1 y1 v1 ... xK yK vK`。stats 阶段会按"point 标注落在哪个 rectangle 内"决定关键点归属，关键点标签若全为数字则按数值排序作为顺序，否则按首次出现顺序。

### 6. `utils/` 是共享能力层
这里放的是跨接口复用的实现：
- 路径处理与本地化兼容：`path_handler.py`
- 数据集分析、切分、JSON/TXT 转换、YAML 生成、预测
- 日志写入与训练回调
- `utils/__init__.py` 只做导出聚合，便于 `from utils import ...`

路径语义很重要：当前代码已经偏向“本地工作区优先”，`ensure_local_path()` 主要是兼容旧调用点。

### 7. `pt2onnx/`、`rknn_model_zoo/` 是导出链路
这部分是嵌入式的模型导出工具链，不要把它和主 API 的业务层混在一起。改动模型导出时，优先先看这里的独立测试和工具配置。

## 文档同步

`README.md` 和 `README_CN.md` 都是对外说明，修改用户可见行为时尽量保持两份内容一致。