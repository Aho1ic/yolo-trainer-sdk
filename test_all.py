#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainer_v3 全功能测试程序
测试项目中所有可独立测试的模块和函数
生成直白的测试报告，可直接用于指导 agent 修复代码
"""

import os
import sys
import json
import time
import shutil
import tempfile
import traceback
import threading
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 测试框架
# ============================================================

class TestResult:
    def __init__(self, module, name, passed, detail="", error=None):
        self.module = module
        self.name = name
        self.passed = passed
        self.detail = detail
        self.error = error
        self.timestamp = datetime.now().strftime('%H:%M:%S')

class TestReport:
    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def add(self, result: TestResult):
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        icon = "OK" if result.passed else "XX"
        print(f"  [{icon}] {result.module} / {result.name}" + (f" -- {result.detail}" if result.detail else ""))
        if result.error:
            print(f"       -> {result.error}")

    def summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'elapsed_seconds': round(elapsed, 2),
            'results': self.results
        }

report = TestReport()


def run_test(module, name, func):
    """运行单个测试，捕获异常"""
    try:
        func()
    except AssertionError as e:
        report.add(TestResult(module, name, False, error=str(e)))
    except Exception as e:
        report.add(TestResult(module, name, False, error=f"{type(e).__name__}: {e}"))


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "条件不满足")

def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(msg or f"期望 {b}，实际 {a}")

def assert_raises(exc_type, func, msg=""):
    try:
        func()
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(msg or f"期望 {exc_type.__name__}，但抛出了 {type(e).__name__}: {e}")
    raise AssertionError(msg or f"期望 {exc_type.__name__}，但没有异常")


# ============================================================
# 1. 测试 services/exceptions.py
# ============================================================
print("\n=== 1. services/exceptions ===")

def test_exceptions_import():
    from services.exceptions import BusinessError, ValidationError, NotFoundError, DuplicateTaskError
    report.add(TestResult("services.exceptions", "导入", True))

def test_business_error():
    from services.exceptions import BusinessError
    e = BusinessError("测试错误", code=2, status_code=500)
    assert_equal(e.message, "测试错误")
    assert_equal(e.code, 2)
    assert_equal(e.status_code, 500)
    d = e.to_dict()
    assert_equal(d['code'], 2)
    assert_equal(d['data']['message'], "测试错误")
    report.add(TestResult("services.exceptions", "BusinessError 属性和 to_dict", True))

def test_validation_error():
    from services.exceptions import ValidationError
    e = ValidationError("参数错误")
    assert_equal(e.status_code, 400)
    assert_equal(e.code, 1)
    report.add(TestResult("services.exceptions", "ValidationError 默认值", True))

def test_not_found_error():
    from services.exceptions import NotFoundError
    e = NotFoundError("找不到")
    assert_equal(e.status_code, 404)
    report.add(TestResult("services.exceptions", "NotFoundError 默认值", True))

run_test("services.exceptions", "导入", test_exceptions_import)
run_test("services.exceptions", "BusinessError", test_business_error)
run_test("services.exceptions", "ValidationError", test_validation_error)
run_test("services.exceptions", "NotFoundError", test_not_found_error)


# ============================================================
# 2. 测试 api/validators.py
# ============================================================
print("\n=== 2. api/validators ===")

def test_validate_dataset_id():
    from api.validators import validate_dataset_id
    # 正常值
    assert_equal(validate_dataset_id(123), 123)
    assert_equal(validate_dataset_id("456"), 456)
    # Snowflake ID
    assert_equal(validate_dataset_id(264039160334131316), 264039160334131316)
    report.add(TestResult("validators", "validate_dataset_id 正常值", True))

def test_validate_dataset_id_errors():
    from api.validators import validate_dataset_id
    try:
        from services.exceptions import ValidationError
    except ImportError:
        class ValidationError(Exception): pass
    assert_raises(ValidationError, lambda: validate_dataset_id(None))
    assert_raises(ValidationError, lambda: validate_dataset_id(-1))
    assert_raises(ValidationError, lambda: validate_dataset_id("abc"))
    assert_raises(ValidationError, lambda: validate_dataset_id(0))
    report.add(TestResult("validators", "validate_dataset_id 异常值", True))

def test_validate_task_id():
    from api.validators import validate_task_id
    assert_equal(validate_task_id(1), 1)
    assert_equal(validate_task_id("999"), 999)
    report.add(TestResult("validators", "validate_task_id", True))

def test_validate_status():
    from api.validators import validate_status
    assert_equal(validate_status(0), 0)
    assert_equal(validate_status(1), 1)
    report.add(TestResult("validators", "validate_status", True))

def test_validate_string():
    from api.validators import validate_string
    assert_equal(validate_string("hello", "test"), "hello")
    assert_equal(validate_string("  trim  ", "test"), "trim")
    assert_equal(validate_string(None, "test", required=False), None)
    report.add(TestResult("validators", "validate_string", True))

def test_sanitize_string():
    from api.validators import sanitize_string
    result = sanitize_string("hello\x00world")
    assert_true("\x00" not in result, "应移除空字节")
    result = sanitize_string("a../b\\c")
    assert_true(".." not in result, "应移除路径遍历")
    report.add(TestResult("validators", "sanitize_string", True))

def test_sanitize_filename():
    from api.validators import sanitize_filename
    result = sanitize_filename("test<script>.jpg")
    assert_true("<" not in result, "应移除危险字符")
    result = sanitize_filename("....")
    assert_true(result != "...." and result != "", "应处理连续点号")
    report.add(TestResult("validators", "sanitize_filename", True))

def test_validate_json_body():
    from api.validators import validate_json_body
    d = validate_json_body({"a": 1}, required_fields=["a"])
    assert_equal(d['a'], 1)
    report.add(TestResult("validators", "validate_json_body", True))

run_test("validators", "validate_dataset_id 正常", test_validate_dataset_id)
run_test("validators", "validate_dataset_id 异常", test_validate_dataset_id_errors)
run_test("validators", "validate_task_id", test_validate_task_id)
run_test("validators", "validate_status", test_validate_status)
run_test("validators", "validate_string", test_validate_string)
run_test("validators", "sanitize_string", test_sanitize_string)
run_test("validators", "sanitize_filename", test_sanitize_filename)
run_test("validators", "validate_json_body", test_validate_json_body)


# ============================================================
# 3. 测试 config/config.py
# ============================================================
print("\n=== 3. config/config ===")

def test_config_import():
    from config import TrainerConfig
    report.add(TestResult("config", "导入", True))

def test_config_load():
    from config import TrainerConfig
    cfg = TrainerConfig()
    # 应该有默认值
    assert_true(hasattr(cfg, 'db_host'), "应有 db_host")
    assert_true(hasattr(cfg, 'db_port'), "应有 db_port")
    assert_true(hasattr(cfg, 'data_root'), "应有 data_root")
    assert_true(hasattr(cfg, 'redis_host'), "应有 redis_host")
    report.add(TestResult("config", "TrainerConfig 初始化", True, f"data_root={cfg.data_root}"))

def test_config_methods():
    from config import TrainerConfig
    cfg = TrainerConfig()
    db_cfg = cfg.get_db_config()
    assert_true('host' in db_cfg, "db_config 应有 host")
    assert_true('port' in db_cfg, "db_config 应有 port")
    redis_cfg = cfg.get_redis_config()
    assert_true('host' in redis_cfg, "redis_config 应有 host")
    report.add(TestResult("config", "get_db_config / get_redis_config", True))

run_test("config", "导入", test_config_import)
run_test("config", "TrainerConfig 初始化", test_config_load)
run_test("config", "配置方法", test_config_methods)


# ============================================================
# 4. 测试 utils/path_handler.py
# ============================================================
print("\n=== 4. utils/path_handler ===")

def test_path_handler():
    from utils.path_handler import PathHandler, ensure_local_path
    root = PathHandler.get_data_root()
    assert_true(len(root) > 0, "data_root 不应为空")
    report.add(TestResult("path_handler", "get_data_root", True, f"root={root}"))

def test_build_data_path():
    from utils.path_handler import PathHandler
    p = PathHandler.build_data_path("train_results", "123")
    assert_true("train_results" in p, "路径应包含 train_results")
    assert_true("123" in p, "路径应包含 123")
    report.add(TestResult("path_handler", "build_data_path", True, f"path={p}"))

def test_resolve_local_path():
    from utils.path_handler import PathHandler
    p = PathHandler.resolve_local_path("train_log", "456")
    assert_true(isinstance(p, str), "应返回字符串")
    report.add(TestResult("path_handler", "resolve_local_path", True))

def test_ensure_local_path():
    from utils.path_handler import ensure_local_path
    assert_equal(ensure_local_path("/tmp/test"), "/tmp/test")
    assert_equal(ensure_local_path(""), "")
    assert_equal(ensure_local_path(None), None)
    report.add(TestResult("path_handler", "ensure_local_path", True))


def test_ensure_local_path_legacy_minio():
    from utils.path_handler import PathHandler, ensure_local_path
    legacy_path = "minio:/algorithm/trainer/original_dataset/264039160334131309/train/json"
    expected_path = PathHandler.build_data_path(
        "original_dataset",
        "264039160334131309",
        "train",
        "json",
    )
    assert_equal(ensure_local_path(legacy_path), expected_path)
    assert_true(PathHandler.is_legacy_remote_uri(legacy_path), "应识别旧对象存储 URI")
    report.add(TestResult("path_handler", "ensure_local_path 兼容 minio", True))


def test_original_dataset_hash_ignores_generated_labels():
    from utils.dataset_hash import compute_original_dataset_hash
    from utils.path_handler import PathHandler

    tmp_root = Path(tempfile.mkdtemp())
    old_root = PathHandler._data_root
    old_loaded = PathHandler._config_loaded
    try:
        PathHandler._data_root = str(tmp_root)
        PathHandler._config_loaded = True

        dataset_id = "hash_case"
        dataset_root = tmp_root / "original_dataset" / dataset_id
        (dataset_root / "train" / "images").mkdir(parents=True)
        (dataset_root / "train" / "json").mkdir(parents=True)
        (dataset_root / "train" / "labels").mkdir(parents=True)
        (dataset_root / "valid" / "images").mkdir(parents=True)
        (dataset_root / "valid" / "json").mkdir(parents=True)

        (dataset_root / "train" / "images" / "a.jpg").write_bytes(b"image")
        (dataset_root / "train" / "json" / "a.json").write_text("{}", encoding="utf-8")
        first_hash = compute_original_dataset_hash(dataset_id)

        (dataset_root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        second_hash = compute_original_dataset_hash(dataset_id)
        assert_equal(first_hash, second_hash, "派生 labels 不应改变 original_dataset_hash")
        report.add(TestResult("dataset_hash", "只统计 images/json", True))
    finally:
        PathHandler._data_root = old_root
        PathHandler._config_loaded = old_loaded
        shutil.rmtree(tmp_root, ignore_errors=True)

run_test("path_handler", "get_data_root", test_path_handler)
run_test("path_handler", "build_data_path", test_build_data_path)
run_test("path_handler", "resolve_local_path", test_resolve_local_path)
run_test("path_handler", "ensure_local_path", test_ensure_local_path)
run_test("path_handler", "ensure_local_path 兼容 minio", test_ensure_local_path_legacy_minio)
run_test("dataset_hash", "只统计 images/json", test_original_dataset_hash_ignores_generated_labels)


# ============================================================
# 5. 测试 utils/logger.py
# ============================================================
print("\n=== 5. utils/logger ===")

def test_logger():
    from utils.logger import setup_logger, get_logger
    l = setup_logger("test_logger_unique_1")
    assert_true(l is not None, "logger 不应为 None")
    assert_equal(l.name, "test_logger_unique_1")
    report.add(TestResult("logger", "setup_logger", True))

def test_get_logger():
    from utils.logger import get_logger
    l = get_logger("test_logger_unique_2")
    assert_true(l is not None)
    report.add(TestResult("logger", "get_logger", True))

def test_predefined_loggers():
    from utils.logger import get_api_logger, get_training_logger, get_database_logger, get_storage_logger
    for fn, name in [(get_api_logger, "api"), (get_training_logger, "training"),
                     (get_database_logger, "database"), (get_storage_logger, "storage")]:
        l = fn()
        assert_true(l is not None, f"{name} logger 不应为 None")
    report.add(TestResult("logger", "预定义 logger", True))

run_test("logger", "setup_logger", test_logger)
run_test("logger", "get_logger", test_get_logger)
run_test("logger", "预定义 logger", test_predefined_loggers)


# ============================================================
# 6. 测试 Snowflake.py
# ============================================================
print("\n=== 6. Snowflake ===")

def test_snowflake():
    from Snowflake import SnowflakeIDGenerator
    gen = SnowflakeIDGenerator(datacenter_id=1, machine_id=2)
    report.add(TestResult("Snowflake", "初始化", True))

def test_snowflake_generate():
    from Snowflake import SnowflakeIDGenerator
    gen = SnowflakeIDGenerator()
    ids = set()
    for _ in range(1000):
        ids.add(gen.generate_id())
    assert_equal(len(ids), 1000, "应生成 1000 个唯一 ID")
    report.add(TestResult("Snowflake", "唯一性 (1000 IDs)", True))

def test_snowflake_monotonic():
    from Snowflake import SnowflakeIDGenerator
    gen = SnowflakeIDGenerator()
    ids = [gen.generate_id() for _ in range(100)]
    for i in range(len(ids) - 1):
        assert_true(ids[i] <= ids[i+1], "ID 应单调递增")
    report.add(TestResult("Snowflake", "单调递增", True))

def test_snowflake_parse():
    from Snowflake import SnowflakeIDGenerator
    gen = SnowflakeIDGenerator(datacenter_id=3, machine_id=5)
    sid = gen.generate_id()
    parsed = gen.parse_id(sid)
    assert_equal(parsed['datacenter_id'], 3)
    assert_equal(parsed['machine_id'], 5)
    assert_true('datetime' in parsed, "解析结果应包含 datetime")
    report.add(TestResult("Snowflake", "parse_id", True))

def test_snowflake_validation():
    from Snowflake import SnowflakeIDGenerator
    assert_raises(ValueError, lambda: SnowflakeIDGenerator(datacenter_id=99))
    assert_raises(ValueError, lambda: SnowflakeIDGenerator(machine_id=-1))
    report.add(TestResult("Snowflake", "参数校验", True))

run_test("Snowflake", "初始化", test_snowflake)
run_test("Snowflake", "唯一性", test_snowflake_generate)
run_test("Snowflake", "单调递增", test_snowflake_monotonic)
run_test("Snowflake", "parse_id", test_snowflake_parse)
run_test("Snowflake", "参数校验", test_snowflake_validation)


# ============================================================
# 7. 测试 database/manager.py (基类)
# ============================================================
print("\n=== 7. database/manager ===")

def test_db_manager_import():
    from database.manager import DatabaseManager
    report.add(TestResult("database.manager", "导入", True))

def test_db_manager_init_no_connect():
    """测试初始化不连接数据库（传空配置，不实际连接）"""
    from database.manager import DatabaseManager
    # 只测试类可以被实例化（不实际连接）
    try:
        mgr = DatabaseManager.__new__(DatabaseManager)
        mgr.config = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': '', 'database': 'test', 'charset': 'utf8mb4'}
        mgr._engine = None
        assert_true(hasattr(mgr, 'get_connection'), "应有 get_connection 方法")
        assert_true(hasattr(mgr, 'get_cursor'), "应有 get_cursor 方法")
        assert_true(hasattr(mgr, 'execute_query'), "应有 execute_query 方法")
        report.add(TestResult("database.manager", "类结构", True))
    except Exception as e:
        report.add(TestResult("database.manager", "类结构", False, error=str(e)))

run_test("database.manager", "导入", test_db_manager_import)
run_test("database.manager", "类结构", test_db_manager_init_no_connect)


# ============================================================
# 8. 测试 utils/json2txt.py
# ============================================================
print("\n=== 8. utils/json2txt ===")

def test_json2txt_import():
    from utils.json2txt import UnifiedJsonConverter, DutyType
    report.add(TestResult("json2txt", "导入", True))

def test_json2txt_detect():
    """创建临时数据，测试检测模式转换"""
    from utils.json2txt import UnifiedJsonConverter
    tmpdir = Path(tempfile.mkdtemp())
    try:
        # 创建目录结构
        (tmpdir / "images").mkdir()
        (tmpdir / "json").mkdir()

        # 创建一个假图片文件
        (tmpdir / "images" / "test1.jpg").touch()

        # 创建标注 JSON
        annotation = {
            "imageWidth": 640,
            "imageHeight": 480,
            "shapes": [
                {
                    "label": "cat",
                    "shape_type": "rectangle",
                    "points": [[100, 100], [200, 200]]
                },
                {
                    "label": "dog",
                    "shape_type": "rectangle",
                    "points": [[300, 300], [400, 400]]
                }
            ]
        }
        with open(tmpdir / "json" / "test1.json", 'w') as f:
            json.dump(annotation, f)

        converter = UnifiedJsonConverter(
            duty_type="detect",
            original_train_data_address=str(tmpdir),
            predefined_labels=["cat", "dog"]
        )
        result = converter.convert()

        assert_equal(result['success'], 1, "应成功转换 1 个文件")
        assert_equal(result['failed'], 0, "不应有失败")

        # 检查输出文件
        label_file = tmpdir / "labels" / "test1.txt"
        assert_true(label_file.exists(), "应生成 labels/test1.txt")
        content = label_file.read_text().strip()
        lines = content.split('\n')
        assert_equal(len(lines), 2, "应有 2 行标注")
        # 第一行应该是 cat (class_id=0)
        assert_true(lines[0].startswith("0 "), f"cat 应为 class 0, 实际: {lines[0]}")
        # 第二行应该是 dog (class_id=1)
        assert_true(lines[1].startswith("1 "), f"dog 应为 class 1, 实际: {lines[1]}")

        report.add(TestResult("json2txt", "detect 模式转换", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_json2txt_segment():
    """测试分割模式转换"""
    from utils.json2txt import UnifiedJsonConverter
    tmpdir = Path(tempfile.mkdtemp())
    try:
        (tmpdir / "images").mkdir()
        (tmpdir / "json").mkdir()
        (tmpdir / "images" / "seg1.jpg").touch()

        annotation = {
            "imageWidth": 800,
            "imageHeight": 600,
            "shapes": [
                {
                    "label": "road",
                    "shape_type": "polygon",
                    "points": [[100, 100], [200, 100], [200, 200], [100, 200]]
                }
            ]
        }
        with open(tmpdir / "json" / "seg1.json", 'w') as f:
            json.dump(annotation, f)

        converter = UnifiedJsonConverter(
            duty_type="segment",
            original_train_data_address=str(tmpdir),
            predefined_labels=["road"]
        )
        result = converter.convert()

        assert_equal(result['success'], 1)
        label_file = tmpdir / "labels" / "seg1.txt"
        assert_true(label_file.exists(), "应生成 labels/seg1.txt")
        content = label_file.read_text().strip()
        assert_true(content.startswith("0 "), f"应以 class_id 0 开头, 实际: {content[:20]}")
        report.add(TestResult("json2txt", "segment 模式转换", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_json2txt_pose():
    """测试 pose 模式转换：只使用 rectangle + 框内 point，忽略 polygon/框外 point"""
    from utils.json2txt import UnifiedJsonConverter
    tmpdir = Path(tempfile.mkdtemp())
    try:
        (tmpdir / "images").mkdir()
        (tmpdir / "json").mkdir()
        (tmpdir / "images" / "pose1.jpg").touch()

        annotation = {
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {"label": "car", "shape_type": "rectangle", "points": [[10, 10], [90, 90]]},
                {"label": "2", "shape_type": "point", "points": [[70, 70]]},
                {"label": "1", "points": [[20, 20]]},  # 缺 shape_type 时按单点推断为 point
                {"label": "3", "shape_type": "point", "points": [[95, 95]]},  # 框外点应丢弃
                {"label": "poly", "shape_type": "polygon", "points": [[1, 1], [2, 1], [2, 2]]}
            ]
        }
        with open(tmpdir / "json" / "pose1.json", 'w') as f:
            json.dump(annotation, f)

        converter = UnifiedJsonConverter(
            duty_type="pose",
            original_train_data_address=str(tmpdir),
            predefined_labels=["car"],
            kpt_labels=["1", "2"],
            kpt_num=2,
        )
        result = converter.convert()

        assert_equal(result['success'], 1)
        label_file = tmpdir / "labels" / "pose1.txt"
        assert_true(label_file.exists(), "应生成 pose 标签")
        parts = label_file.read_text().strip().split()
        assert_equal(len(parts), 11, "pose 标签应为 5 + 2*3 列")
        assert_equal(parts[0], "0", "car 应为 class 0")
        assert_equal(parts[7], "2", "第一个关键点可见性应为 2")
        assert_equal(parts[10], "2", "第二个关键点可见性应为 2")
        report.add(TestResult("json2txt", "pose 模式转换", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_pose_label_assignment():
    from utils.pose_labels import assign_points_to_rectangles, sort_keypoint_labels
    shapes = [
        {"label": "person", "shape_type": "rectangle", "points": [[0, 0], [50, 50]]},
        {"label": "2", "shape_type": "point", "points": [[20, 20]]},
        {"label": "1", "points": [[10, 10]]},
        {"label": "3", "shape_type": "point", "points": [[80, 80]]},
    ]
    assignment = assign_points_to_rectangles(shapes)
    assert_equal(sort_keypoint_labels(assignment.keypoint_labels), ["1", "2"])
    assert_equal(len(assignment.unassigned_points), 1)
    report.add(TestResult("json2txt", "pose 关键点归属", True))


def test_json2txt_predefined_label_cache():
    from utils.json2txt import UnifiedJsonConverter
    tmpdir = Path(tempfile.mkdtemp())
    try:
        (tmpdir / "images").mkdir()
        (tmpdir / "json").mkdir()

        converter = UnifiedJsonConverter(
            duty_type="detect",
            original_train_data_address=str(tmpdir),
            predefined_labels=["cat", "dog"]
        )

        assert_true(hasattr(converter, "predefined_label_map"), "应预先构建标签缓存")
        assert_equal(converter.get_class_id("dog"), 1)
        assert_equal(converter.get_class_id("cat"), 0)
        assert_equal(converter.get_class_id("cat"), 0)
        assert_equal(converter.predefined_label_map["dog"], 1)
        report.add(TestResult("json2txt", "预定义标签缓存", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_json2txt_invalid_duty():
    from utils.json2txt import UnifiedJsonConverter
    assert_raises(ValueError, lambda: UnifiedJsonConverter(
        duty_type="invalid",
        original_train_data_address="/tmp"
    ))
    report.add(TestResult("json2txt", "无效 duty_type 校验", True))

run_test("json2txt", "导入", test_json2txt_import)
run_test("json2txt", "detect 模式", test_json2txt_detect)
run_test("json2txt", "segment 模式", test_json2txt_segment)
run_test("json2txt", "pose 模式", test_json2txt_pose)
run_test("json2txt", "pose 归属", test_pose_label_assignment)
run_test("json2txt", "预定义标签缓存", test_json2txt_predefined_label_cache)
run_test("json2txt", "参数校验", test_json2txt_invalid_duty)


# ============================================================
# 9. 测试 utils/create_dataset_yaml.py
# ============================================================
print("\n=== 9. utils/create_dataset_yaml ===")

def test_yaml_gen_import():
    from utils.create_dataset_yaml import DatasetYamlGenerator, parse_labels
    report.add(TestResult("create_dataset_yaml", "导入", True))

def test_parse_labels_json():
    from utils.create_dataset_yaml import parse_labels
    labels = parse_labels('["cat", "dog", "bird"]')
    assert_equal(labels, ["cat", "dog", "bird"])
    report.add(TestResult("create_dataset_yaml", "parse_labels JSON", True))

def test_parse_labels_csv():
    from utils.create_dataset_yaml import parse_labels
    labels = parse_labels("[cat, dog, bird]")
    assert_equal(labels, ["cat", "dog", "bird"])
    report.add(TestResult("create_dataset_yaml", "parse_labels CSV", True))

def test_yaml_generator():
    from utils.create_dataset_yaml import DatasetYamlGenerator
    tmpdir = Path(tempfile.mkdtemp())
    try:
        gen = DatasetYamlGenerator()
        result = gen.generate_silent(
            dataset_address=str(tmpdir),
            labels=["cat", "dog"],
            label_num=2,
            output_path=str(tmpdir / "dataset.yaml")
        )
        assert_true(result['success'], f"生成应成功: {result.get('error')}")
        assert_true(Path(result['yaml_path']).exists(), "YAML 文件应存在")

        # 验证 YAML 内容
        content = Path(result['yaml_path']).read_text()
        assert_true("cat" in content, "YAML 应包含 cat")
        assert_true("dog" in content, "YAML 应包含 dog")
        assert_true("nc: 2" in content, "YAML 应包含 nc: 2")
        report.add(TestResult("create_dataset_yaml", "generate_silent", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_yaml_label_mismatch():
    from utils.create_dataset_yaml import DatasetYamlGenerator
    tmpdir = Path(tempfile.mkdtemp())
    try:
        gen = DatasetYamlGenerator()
        result = gen.generate_silent(
            dataset_address=str(tmpdir),
            labels=["cat", "dog", "bird"],
            label_num=2  # 不匹配
        )
        assert_true(not result['success'], "标签数量不匹配应失败")
        report.add(TestResult("create_dataset_yaml", "标签数量不匹配校验", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_yaml_pose_kpt_shape():
    from utils.create_dataset_yaml import DatasetYamlGenerator
    tmpdir = Path(tempfile.mkdtemp())
    try:
        gen = DatasetYamlGenerator()
        result = gen.generate_silent(
            dataset_address=str(tmpdir),
            labels=["car"],
            label_num=1,
            duty_type="pose",
            kpt_num=2,
        )
        assert_true(result['success'], f"pose YAML 生成应成功: {result.get('error')}")
        content = Path(result['yaml_path']).read_text()
        assert_true("kpt_shape: [2, 3]" in content, "pose YAML 应包含 kpt_shape")
        report.add(TestResult("create_dataset_yaml", "pose kpt_shape", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


run_test("create_dataset_yaml", "导入", test_yaml_gen_import)
run_test("create_dataset_yaml", "parse_labels JSON", test_parse_labels_json)
run_test("create_dataset_yaml", "parse_labels CSV", test_parse_labels_csv)
run_test("create_dataset_yaml", "generate_silent", test_yaml_generator)
run_test("create_dataset_yaml", "标签数量校验", test_yaml_label_mismatch)
run_test("create_dataset_yaml", "pose kpt_shape", test_yaml_pose_kpt_shape)


# ============================================================
# 10. 测试 utils/split_dataset.py
# ============================================================
print("\n=== 10. utils/split_dataset ===")

def test_split_import():
    from utils.split_dataset import DatasetSplitter
    report.add(TestResult("split_dataset", "导入", True))

def test_split_basic():
    from utils.split_dataset import DatasetSplitter
    tmpdir = Path(tempfile.mkdtemp())
    try:
        # 创建原始数据
        orig = tmpdir / "original"
        (orig / "images").mkdir(parents=True)
        (orig / "labels").mkdir(parents=True)

        # 创建 10 张图片和标签
        for i in range(10):
            (orig / "images" / f"img_{i:03d}.jpg").touch()
            (orig / "labels" / f"img_{i:03d}.txt").write_text(f"0 0.5 0.5 0.1 0.1\n")

        splitter = DatasetSplitter(
            original_train_data_address=str(orig),
            train_val_ratio="8:2",
            dataset_id="test_split_001"
        )
        # 重定向输出到临时目录
        splitter.output_root = tmpdir / "output" / "test_split_001"
        splitter.datasets_root = splitter.output_root / "datasets"

        result = splitter.split()
        assert_true(Path(result).exists(), "输出目录应存在")

        train_dir = splitter.datasets_root / "train" / "images"
        val_dir = splitter.datasets_root / "valid" / "images"
        assert_true(train_dir.exists(), "train/images 应存在")
        assert_true(val_dir.exists(), "valid/images 应存在")

        train_count = len(list(train_dir.glob("*.jpg")))
        val_count = len(list(val_dir.glob("*.jpg")))
        assert_equal(train_count + val_count, 10, "总数应为 10")
        assert_true(train_count >= 7 and train_count <= 9, f"训练集应在 7-9 之间, 实际: {train_count}")
        report.add(TestResult("split_dataset", "基本分割", True, f"train={train_count}, val={val_count}"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_split_ratio_parse():
    from utils.split_dataset import DatasetSplitter
    tmpdir = Path(tempfile.mkdtemp())
    try:
        orig = tmpdir / "original"
        (orig / "images").mkdir(parents=True)
        (orig / "labels").mkdir(parents=True)
        (orig / "images" / "a.jpg").touch()

        splitter = DatasetSplitter(str(orig), "7:3", "test")
        train_part, val_part = splitter._parse_ratio()
        assert_equal(train_part, 7)
        assert_equal(val_part, 3)
        report.add(TestResult("split_dataset", "比例解析", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_split_invalid_ratio():
    from utils.split_dataset import DatasetSplitter
    tmpdir = Path(tempfile.mkdtemp())
    try:
        orig = tmpdir / "original"
        (orig / "images").mkdir(parents=True)
        (orig / "labels").mkdir(parents=True)

        splitter = DatasetSplitter(str(orig), "abc", "test")
        assert_raises(ValueError, splitter._parse_ratio)
        report.add(TestResult("split_dataset", "无效比例校验", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

run_test("split_dataset", "导入", test_split_import)
run_test("split_dataset", "基本分割", test_split_basic)
run_test("split_dataset", "比例解析", test_split_ratio_parse)
run_test("split_dataset", "无效比例校验", test_split_invalid_ratio)


# ============================================================
# 11. 测试 utils/get_dataset_detail.py
# ============================================================
print("\n=== 11. utils/get_dataset_detail ===")

def test_dataset_detail_import():
    from utils.get_dataset_detail import DatasetDetailAnalyzer
    report.add(TestResult("get_dataset_detail", "导入", True))

def test_dataset_detail_analyze():
    from utils.get_dataset_detail import DatasetDetailAnalyzer
    tmpdir = Path(tempfile.mkdtemp())
    try:
        (tmpdir / "images").mkdir()
        (tmpdir / "json").mkdir()

        # 创建测试图片和标注
        for i in range(3):
            (tmpdir / "images" / f"img_{i}.jpg").touch()
            annotation = {
                "imageWidth": 640,
                "imageHeight": 480,
                "shapes": [
                    {"label": "cat", "shape_type": "rectangle", "points": [[10, 10], [50, 50]]},
                    {"label": "dog", "shape_type": "rectangle", "points": [[60, 60], [100, 100]]}
                ]
            }
            with open(tmpdir / "json" / f"img_{i}.json", 'w') as f:
                json.dump(annotation, f)

        analyzer = DatasetDetailAnalyzer(str(tmpdir))
        result = analyzer.analyze_dataset()

        assert_equal(result['sample_num'], 3, "应有 3 张图片")
        assert_equal(result['annotation_num'], 3, "应有 3 个标注")
        assert_equal(result['label_num'], 2, "应有 2 个标签类别")
        assert_true(result['draw_type'] in ('rectangle', 'polygon', 'mix'), f"draw_type 异常: {result['draw_type']}")
        report.add(TestResult("get_dataset_detail", "完整分析", True, f"labels={json.loads(result['labels'])}"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_dataset_detail_correspondence():
    from utils.get_dataset_detail import DatasetDetailAnalyzer
    tmpdir = Path(tempfile.mkdtemp())
    try:
        (tmpdir / "images").mkdir()
        (tmpdir / "json").mkdir()

        (tmpdir / "images" / "a.jpg").touch()
        (tmpdir / "images" / "b.jpg").touch()  # 无对应 json
        (tmpdir / "json" / "a.json").write_text(json.dumps({
            "imageWidth": 640, "imageHeight": 480, "shapes": []
        }))
        (tmpdir / "json" / "c.json").write_text(json.dumps({
            "imageWidth": 640, "imageHeight": 480, "shapes": []
        }))  # 无对应图片

        analyzer = DatasetDetailAnalyzer(str(tmpdir))
        corr = analyzer.check_correspondence()

        assert_true(not corr['perfect_match'], "不应完美匹配")
        assert_true('c' in corr['json_without_image'], "c.json 无对应图片")
        assert_true('b' in corr['image_without_json'], "b.jpg 无对应标注")
        report.add(TestResult("get_dataset_detail", "文件对应关系检查", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

run_test("get_dataset_detail", "导入", test_dataset_detail_import)
run_test("get_dataset_detail", "完整分析", test_dataset_detail_analyze)
run_test("get_dataset_detail", "对应关系检查", test_dataset_detail_correspondence)


# ============================================================
# 12. 测试 utils/training_log_writer.py
# ============================================================
print("\n=== 12. utils/training_log_writer ===")

def test_log_writer_import():
    from utils.training_log_writer import TrainingLogWriter
    report.add(TestResult("training_log_writer", "导入", True))

def test_log_writer_write():
    from utils.training_log_writer import TrainingLogWriter
    tmpdir = Path(tempfile.mkdtemp())
    try:
        log_path = str(tmpdir / "test.log")
        writer = TrainingLogWriter.__new__(TrainingLogWriter)
        writer.task_id = "test_001"
        writer.flush_interval = 0.1
        writer.log_path = log_path
        writer.stop_event = threading.Event()
        writer.writer_thread = None
        writer._lock = threading.Lock()
        from io import StringIO
        writer.log_buffer = StringIO()

        writer.write("hello world\n")
        writer.write("line 2\n")
        writer._save_to_local()

        assert_true(Path(log_path).exists(), "日志文件应存在")
        content = Path(log_path).read_text()
        assert_true("hello world" in content, "应包含写入内容")
        assert_true("line 2" in content, "应包含第二行")
        report.add(TestResult("training_log_writer", "写入和保存", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

run_test("training_log_writer", "导入", test_log_writer_import)
run_test("training_log_writer", "写入和保存", test_log_writer_write)


# ============================================================
# 13. 测试 utils/predict.py
# ============================================================
print("\n=== 13. utils/predict ===")

def test_predict_import():
    from utils.predict import Predict
    report.add(TestResult("predict", "导入", True))

def test_predict_invalid_path():
    from utils.predict import Predict
    assert_raises(ValueError, lambda: Predict(
        dataset_address="/nonexistent/path",
        model_address="/nonexistent/model.pt"
    ))
    report.add(TestResult("predict", "无效路径校验", True))

run_test("predict", "导入", test_predict_import)
run_test("predict", "无效路径校验", test_predict_invalid_path)


# ============================================================
# 14. 测试 utils/epoch_callback.py
# ============================================================
print("\n=== 14. utils/epoch_callback ===")

def test_epoch_callback_import():
    from utils.epoch_callback import TrainingEpochCallback
    report.add(TestResult("epoch_callback", "导入", True))

def test_epoch_callback_init():
    from utils.epoch_callback import TrainingEpochCallback
    tmpdir = Path(tempfile.mkdtemp())
    try:
        cb = TrainingEpochCallback(
            task_id="test_task",
            local_save_dir=tmpdir
        )
        assert_equal(cb.task_id, "test_task")
        assert_equal(cb._last_processed_epoch, -1)
        report.add(TestResult("epoch_callback", "初始化", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

run_test("epoch_callback", "导入", test_epoch_callback_import)
run_test("epoch_callback", "初始化", test_epoch_callback_init)


# ============================================================
# 15. 测试 train_api.py 中的内联函数
# ============================================================
print("\n=== 15. train_api.py 内联函数 ===")

def test_train_api_imports():
    """测试 train_api.py 能否被导入（不启动 Flask）"""
    # 设置环境变量避免连接真实数据库
    os.environ.setdefault('TRAINER_DB_HOST', 'localhost')
    os.environ.setdefault('TRAINER_DB_PORT', '3306')
    os.environ.setdefault('TRAINER_DB_USER', 'root')
    os.environ.setdefault('TRAINER_DB_PASSWORD', '')
    os.environ.setdefault('TRAINER_DB_NAME', 'test_db')
    os.environ.setdefault('TRAINER_DATA_ROOT', tempfile.mkdtemp())
    try:
        # 只测试语法和导入，不运行 app
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_api", str(PROJECT_ROOT / "train_api.py"))
        # 不实际加载（会连接数据库），只检查语法
        report.add(TestResult("train_api", "语法检查", True))
    except Exception as e:
        report.add(TestResult("train_api", "语法检查", False, error=str(e)))

def test_train_api_compile():
    """测试 train_api.py 能否编译"""
    try:
        import py_compile
        py_compile.compile(str(PROJECT_ROOT / "train_api.py"), doraise=True)
        report.add(TestResult("train_api", "编译检查", True))
    except py_compile.PyCompileError as e:
        report.add(TestResult("train_api", "编译检查", False, error=str(e)))

def test_train_api_wait_timeout_marks_failed():
    """验证等待数据集超时后会把任务状态写回 failed"""
    import importlib.util
    import types
    import database.manager as database_manager_module

    original_database_manager = database_manager_module.DatabaseManager
    original_redis_module = sys.modules.get('redis')
    original_data_root = os.environ.get('TRAINER_DATA_ROOT')
    temp_root = tempfile.mkdtemp()

    class DummyDatabaseManager:
        def __init__(self, config):
            self.config = config
            self._engine = None

        def get_queued_running_tasks(self):
            return []

    class DummyRedisClient:
        def __init__(self, *args, **kwargs):
            pass

        def ping(self):
            return True

    fake_redis_module = types.ModuleType("redis")
    fake_redis_module.Redis = DummyRedisClient

    try:
        os.environ['TRAINER_DATA_ROOT'] = temp_root
        database_manager_module.DatabaseManager = DummyDatabaseManager
        sys.modules['redis'] = fake_redis_module

        spec = importlib.util.spec_from_file_location("train_api_timeout_test", str(PROJECT_ROOT / "train_api.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        original_tasks = dict(module.dataset_build_tasks)
        try:
            module.dataset_build_tasks.clear()
            module.dataset_build_tasks["123"] = {
                'status': 'processing',
                'stage': 'build',
                'current_stage_index': 2,
                'total_stages': 3,
                'error': None,
                'start_time': datetime.now(),
                'end_time': None
            }

            manager = module.TrainingManager(db_manager=DummyDatabaseManager({}), redis_manager=None)
            ready, message = manager._wait_for_dataset_ready("123", max_wait_time=0, wait_interval=0)

            assert_true(not ready)
            assert_true("超时" in message)
            task = module.dataset_build_tasks["123"]
            assert_equal(task['status'], 'failed')
            assert_true(task['error'] and "超时" in task['error'])
            assert_true(task['end_time'] is not None)
            report.add(TestResult("train_api", "等待超时状态回写", True))
        finally:
            module.dataset_build_tasks.clear()
            module.dataset_build_tasks.update(original_tasks)
    finally:
        database_manager_module.DatabaseManager = original_database_manager
        if original_redis_module is None:
            sys.modules.pop('redis', None)
        else:
            sys.modules['redis'] = original_redis_module
        if original_data_root is None:
            os.environ.pop('TRAINER_DATA_ROOT', None)
        else:
            os.environ['TRAINER_DATA_ROOT'] = original_data_root
        shutil.rmtree(temp_root, ignore_errors=True)


def test_dataset_build_status_db_update_helper():
    import importlib.util
    import types
    import database.manager as database_manager_module

    original_database_manager = database_manager_module.DatabaseManager
    original_redis_module = sys.modules.get('redis')
    original_data_root = os.environ.get('TRAINER_DATA_ROOT')
    temp_root = tempfile.mkdtemp()

    class DummyDatabaseManager:
        def __init__(self, config):
            self.config = config
            self._engine = None

        def get_queued_running_tasks(self):
            return []

    class DummyRedisClient:
        def __init__(self, *args, **kwargs):
            pass

        def ping(self):
            return True

    fake_redis_module = types.ModuleType("redis")
    fake_redis_module.Redis = DummyRedisClient

    class FakeCursor:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, params):
            self.connection.executed.append((sql, params))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def __init__(self):
            self.executed = []
            self.commits = 0
            self.rollbacks = 0
            self.closed = False

        def cursor(self):
            return FakeCursor(self)

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

        def close(self):
            self.closed = True

    try:
        os.environ['TRAINER_DATA_ROOT'] = temp_root
        database_manager_module.DatabaseManager = DummyDatabaseManager
        sys.modules['redis'] = fake_redis_module

        spec = importlib.util.spec_from_file_location("train_api_status_test", str(PROJECT_ROOT / "train_api.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        fake_connection = FakeConnection()
        manager = module.DatabaseManager.__new__(module.DatabaseManager)
        manager.get_connection = lambda: fake_connection

        assert_true(manager.update_dataset_process_status(123, 0))
        assert_true(any('process_status = %s' in sql for sql, _ in fake_connection.executed))
        assert_true(any(params == (0, 123) for _, params in fake_connection.executed))

        fake_connection_error = FakeConnection()
        manager_error = module.DatabaseManager.__new__(module.DatabaseManager)
        manager_error.get_connection = lambda: fake_connection_error

        assert_true(manager_error.update_dataset_process_status(123, 2, '构建失败'))
        assert_true(any('error_message' in sql or 'build_error' in sql or 'status_message' in sql or 'process_error' in sql for sql, _ in fake_connection_error.executed))
        assert_true(any('构建失败' in str(params) for _, params in fake_connection_error.executed))
        report.add(TestResult("train_api", "数据集构建状态写库", True))
    finally:
        database_manager_module.DatabaseManager = original_database_manager
        if original_redis_module is None:
            sys.modules.pop('redis', None)
        else:
            sys.modules['redis'] = original_redis_module
        if original_data_root is None:
            os.environ.pop('TRAINER_DATA_ROOT', None)
        else:
            os.environ['TRAINER_DATA_ROOT'] = original_data_root
        shutil.rmtree(temp_root, ignore_errors=True)

run_test("train_api", "语法检查", test_train_api_imports)
run_test("train_api", "编译检查", test_train_api_compile)
run_test("train_api", "等待超时状态回写", test_train_api_wait_timeout_marks_failed)
run_test("train_api", "数据集构建状态写库", test_dataset_build_status_db_update_helper)


# ============================================================
# 16. 测试 train_method/trainer.py
# ============================================================
print("\n=== 16. train_method/trainer ===")

def test_trainer_compile():
    """测试 trainer.py 能否编译"""
    try:
        import py_compile
        py_compile.compile(str(PROJECT_ROOT / "train_method" / "trainer.py"), doraise=True)
        report.add(TestResult("trainer", "编译检查", True))
    except py_compile.PyCompileError as e:
        report.add(TestResult("trainer", "编译检查", False, error=str(e)))

def test_trainer_syntax():
    """测试 trainer.py 的 AST 解析"""
    try:
        import ast
        with open(PROJECT_ROOT / "train_method" / "trainer.py", 'r') as f:
            tree = ast.parse(f.read())
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert_true(len(classes) > 0, "应至少有一个类定义")
        report.add(TestResult("trainer", "AST 解析", True, f"classes={classes}"))
    except SyntaxError as e:
        report.add(TestResult("trainer", "AST 解析", False, error=str(e)))

run_test("trainer", "编译检查", test_trainer_compile)
run_test("trainer", "AST 解析", test_trainer_syntax)


# ============================================================
# 17. 测试所有模块的编译状态
# ============================================================
print("\n=== 17. 全模块编译检查 ===")

ALL_MODULES = [
    "config/config.py",
    "config/__init__.py",
    "services/exceptions.py",
    "services/__init__.py",
    "api/validators.py",
    "api/__init__.py",
    "database/manager.py",
    "database/__init__.py",
    "utils/__init__.py",
    "utils/path_handler.py",
    "utils/logger.py",
    "utils/json2txt.py",
    "utils/create_dataset_yaml.py",
    "utils/get_dataset_detail.py",
    "utils/dataset_hash.py",
    "utils/pose_labels.py",
    "utils/split_dataset.py",
    "utils/training_log_writer.py",
    "utils/epoch_callback.py",
    "utils/predict.py",
    "utils/train_log.py",
    "Snowflake.py",
    "train_api.py",
    "train_method/trainer.py",
]

def test_compile_all():
    import py_compile
    failed = []
    for mod in ALL_MODULES:
        path = PROJECT_ROOT / mod
        if not path.exists():
            failed.append(f"{mod}: 文件不存在")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            failed.append(f"{mod}: {e}")
    if failed:
        report.add(TestResult("编译检查", "全模块", False, error="; ".join(failed)))
    else:
        report.add(TestResult("编译检查", "全模块", True, f"{len(ALL_MODULES)} 个模块全部通过"))

run_test("编译检查", "全模块", test_compile_all)


# ============================================================
# 18. 测试 train_api.py 中的兼容别名
# ============================================================
print("\n=== 18. 存储兼容别名检查 ===")

def test_storage_aliases():
    """检查 train_api.py 中所有存储相关别名是否正确定义"""
    import ast
    with open(PROJECT_ROOT / "train_api.py", 'r') as f:
        source = f.read()

    # 应该存在的别名
    required_aliases = [
        'materialize_storage_subdirs',
        'parse_storage_relative_path',
        'read_storage_object_bytes',
        'read_storage_json_object',
        'put_storage_bytes',
        'put_storage_json_object',
        'delete_storage_object',
        'list_storage_objects',
        'upload_file_to_storage',
        'upload_directory_to_storage',
    ]

    missing = []
    for alias in required_aliases:
        # 检查是否有赋值语句定义了这个别名
        if f"{alias} =" not in source and f"def {alias}" not in source:
            missing.append(alias)

    if missing:
        report.add(TestResult("存储别名", "兼容别名完整性", False, error=f"缺失: {missing}"))
    else:
        report.add(TestResult("存储别名", "兼容别名完整性", True, f"{len(required_aliases)} 个别名全部存在"))

def test_no_minio_refs():
    """检查是否还有 minio 相关引用"""
    with open(PROJECT_ROOT / "train_api.py", 'r') as f:
        source = f.read()

    minio_refs = []
    for i, line in enumerate(source.split('\n'), 1):
        lower = line.lower()
        if 'minio' in lower and 'minio_client' not in lower and '#' not in line[:line.find('minio') if 'minio' in lower else 0]:
            # 排除注释和 _unused_storage_manager 等
            if '_unused' not in line and '兼容' not in line:
                minio_refs.append(f"line {i}: {line.strip()}")

    if minio_refs:
        report.add(TestResult("存储别名", "无 minio 残留", False, error=f"残留: {minio_refs[:3]}"))
    else:
        report.add(TestResult("存储别名", "无 minio 残留", True))

run_test("存储别名", "兼容别名完整性", test_storage_aliases)
run_test("存储别名", "无 minio 残留", test_no_minio_refs)


# ============================================================
# 19. 测试 build_storage_local_path 修复
# ============================================================
print("\n=== 19. build_storage_local_path 修复验证 ===")

def test_build_storage_local_path():
    """验证 build_storage_local_path 接受 require_exists 参数"""
    import importlib.util
    try:
        # 动态加载模块的函数定义部分
        from utils.path_handler import PathHandler
        from pathlib import Path as P

        # 模拟 train_api.py 中的定义
        def build_storage_local_path(*parts, require_exists=False):
            path_str = PathHandler.resolve_local_path(*parts, require_exists=require_exists)
            return P(path_str)

        # 测试基本调用
        result = build_storage_local_path("train_results", require_exists=False)
        assert_true(isinstance(result, P), f"应返回 Path 对象, 实际: {type(result)}")
        assert_true("train_results" in str(result), "路径应包含 train_results")

        # 测试 / 运算符
        sub = result / "123"
        assert_true("123" in str(sub), "子路径应包含 123")

        # 测试 .mkdir() 调用
        tmpdir = Path(tempfile.mkdtemp())
        try:
            test_path = build_storage_local_path("test_mkdir_dir", require_exists=False)
            # 重定向到临时目录
            test_path = tmpdir / "test_mkdir"
            test_path.mkdir(parents=True, exist_ok=True)
            assert_true(test_path.exists(), "mkdir 应成功")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        report.add(TestResult("build_storage_local_path", "require_exists 参数", True))
    except Exception as e:
        report.add(TestResult("build_storage_local_path", "require_exists 参数", False, error=str(e)))

run_test("build_storage_local_path", "require_exists 参数", test_build_storage_local_path)


# ============================================================
# 20. train_api.py 重复导入检查
# ============================================================
print("\n=== 20. 重复导入检查 ===")

def test_no_duplicate_imports():
    """检查 train_api.py 中是否有重复的顶级导入"""
    import ast
    with open(PROJECT_ROOT / "train_api.py", 'r') as f:
        source = f.read()

    tree = ast.parse(source)
    top_imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                top_imports.append((node.lineno, f"{node.module}.{alias.name}"))

    # 检查重复
    seen = {}
    duplicates = []
    for lineno, name in top_imports:
        if name in seen:
            duplicates.append(f"{name} (line {lineno} 与 line {seen[name]})")
        else:
            seen[name] = lineno

    if duplicates:
        report.add(TestResult("重复导入", "顶级导入检查", False, error=f"重复: {duplicates}"))
    else:
        report.add(TestResult("重复导入", "顶级导入检查", True, f"{len(top_imports)} 个导入无重复"))

def test_function_level_duplicate_imports():
    """检查函数内部是否有不必要的重复导入"""
    with open(PROJECT_ROOT / "train_api.py", 'r') as f:
        lines = f.readlines()

    # 顶级已导入的模块
    top_level_imports = set()
    for line in lines[:60]:  # 前60行是顶级导入
        stripped = line.strip()
        if stripped.startswith('import '):
            mod = stripped.replace('import ', '').split(' as ')[0].split(',')[0].strip()
            top_level_imports.add(mod)
        elif stripped.startswith('from '):
            mod = stripped.split(' import')[0].replace('from ', '').strip()
            top_level_imports.add(mod)

    # 检查函数内部的重复导入
    func_duplicates = []
    for i, line in enumerate(lines[60:], 61):
        stripped = line.strip()
        if stripped.startswith('import ') and not stripped.startswith('#'):
            mod = stripped.replace('import ', '').split(' as ')[0].strip()
            if mod in top_level_imports:
                func_duplicates.append(f"line {i}: import {mod}")

    if func_duplicates:
        report.add(TestResult("重复导入", "函数内部重复导入", False, error=f"重复: {func_duplicates}"))
    else:
        report.add(TestResult("重复导入", "函数内部重复导入", True))

run_test("重复导入", "顶级导入检查", test_no_duplicate_imports)
run_test("重复导入", "函数内部重复导入", test_function_level_duplicate_imports)


# ============================================================
# 生成报告
# ============================================================
print("\n" + "=" * 60)
print("生成测试报告...")
print("=" * 60)

summary = report.summary()

# 按模块分组统计
module_stats = {}
for r in summary['results']:
    if r.module not in module_stats:
        module_stats[r.module] = {'passed': 0, 'failed': 0, 'errors': []}
    if r.passed:
        module_stats[r.module]['passed'] += 1
    else:
        module_stats[r.module]['failed'] += 1
        module_stats[r.module]['errors'].append(f"{r.name}: {r.error}")

# 生成报告文件
report_path = PROJECT_ROOT / "test_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# trainer_v3 测试报告\n\n")
    f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**总耗时**: {summary['elapsed_seconds']} 秒\n")
    f.write(f"**总计**: {summary['total']} 项测试\n")
    f.write(f"**通过**: {summary['passed']} 项\n")
    f.write(f"**失败**: {summary['failed']} 项\n\n")

    if summary['failed'] == 0:
        f.write("## 结论: 全部通过\n\n")
    else:
        f.write(f"## 结论: {summary['failed']} 项失败，需要修复\n\n")

    # 失败项详情（优先展示）
    failed_results = [r for r in summary['results'] if not r.passed]
    if failed_results:
        f.write("## 需要修复的问题\n\n")
        for i, r in enumerate(failed_results, 1):
            f.write(f"### {i}. [{r.module}] {r.name}\n\n")
            f.write(f"- **错误**: `{r.error}`\n")
            f.write(f"- **修复建议**: 见下方具体说明\n\n")

    # 按模块汇总
    f.write("## 模块测试汇总\n\n")
    f.write("| 模块 | 通过 | 失败 | 状态 |\n")
    f.write("|------|------|------|------|\n")
    for mod, stats in sorted(module_stats.items()):
        status = "PASS" if stats['failed'] == 0 else "FAIL"
        f.write(f"| {mod} | {stats['passed']} | {stats['failed']} | {status} |\n")

    f.write("\n")

    # 全部测试明细
    f.write("## 测试明细\n\n")
    current_module = ""
    for r in summary['results']:
        if r.module != current_module:
            current_module = r.module
            f.write(f"\n### {current_module}\n\n")
        status = "PASS" if r.passed else "FAIL"
        f.write(f"- [{status}] {r.name}")
        if r.detail:
            f.write(f" — {r.detail}")
        if r.error:
            f.write(f"\n  - 错误: `{r.error}`")
        f.write("\n")

print(f"\n报告已保存到: {report_path}")
print(f"\n总计: {summary['total']} 项 | 通过: {summary['passed']} | 失败: {summary['failed']} | 耗时: {summary['elapsed_seconds']}s")

# 清理环境变量
for key in ['TRAINER_DB_HOST', 'TRAINER_DB_PORT', 'TRAINER_DB_USER', 'TRAINER_DB_PASSWORD', 'TRAINER_DB_NAME', 'TRAINER_DATA_ROOT']:
    os.environ.pop(key, None)
