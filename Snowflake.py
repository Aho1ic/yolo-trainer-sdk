"""
生成雪花ID - Snowflake算法实现
基于Twitter的Snowflake算法，生成64位唯一ID
"""

import time
import threading
from datetime import datetime


class SnowflakeIDGenerator:
    """
    Snowflake ID生成器
    
    64位ID结构：
    - 1位符号位（固定为0）
    - 41位时间戳（毫秒级）
    - 10位机器ID（5位数据中心ID + 5位机器ID）
    - 12位序列号
    """
    
    def __init__(self, datacenter_id=0, machine_id=0, epoch=None):
        """
        初始化Snowflake生成器
        
        Args:
            datacenter_id: 数据中心ID (0-31)
            machine_id: 机器ID (0-31) 
            epoch: 起始时间戳（毫秒），默认为2024-01-01 00:00:00 UTC
        """
        # 位数配置
        self.datacenter_id_bits = 5
        self.machine_id_bits = 5
        self.sequence_bits = 12
        
        # 最大值计算
        self.max_datacenter_id = (1 << self.datacenter_id_bits) - 1  # 31
        self.max_machine_id = (1 << self.machine_id_bits) - 1  # 31
        self.max_sequence = (1 << self.sequence_bits) - 1  # 4095
        
        # 位移量
        self.machine_id_shift = self.sequence_bits  # 12
        self.datacenter_id_shift = self.sequence_bits + self.machine_id_bits  # 17
        self.timestamp_shift = self.sequence_bits + self.machine_id_bits + self.datacenter_id_bits  # 22
        
        # 参数验证
        if datacenter_id > self.max_datacenter_id or datacenter_id < 0:
            raise ValueError(f"数据中心ID必须在0-{self.max_datacenter_id}之间")
        if machine_id > self.max_machine_id or machine_id < 0:
            raise ValueError(f"机器ID必须在0-{self.max_machine_id}之间")
            
        self.datacenter_id = datacenter_id
        self.machine_id = machine_id
        
        # 设置起始时间戳（2024-01-01 00:00:00 UTC）
        if epoch is None:
            self.epoch = int(time.mktime(datetime(2024, 1, 1).timetuple()) * 1000)
        else:
            self.epoch = epoch
            
        # 状态变量
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()
    
    def _current_timestamp(self):
        """获取当前时间戳（毫秒）"""
        return int(time.time() * 1000)
    
    def _wait_next_millis(self, last_timestamp):
        """等待直到下一毫秒"""
        timestamp = self._current_timestamp()
        while timestamp <= last_timestamp:
            timestamp = self._current_timestamp()
        return timestamp
    
    def generate_id(self):
        """
        生成Snowflake ID
        
        Returns:
            int: 64位唯一ID
        """
        with self.lock:
            timestamp = self._current_timestamp()
            
            # 时钟回拨检测
            if timestamp < self.last_timestamp:
                raise Exception(f"时钟回拨检测到！当前时间：{timestamp}，上次时间：{self.last_timestamp}")
            
            # 同一毫秒内的序列号处理
            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & self.max_sequence
                # 序列号溢出，等待下一毫秒
                if self.sequence == 0:
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                # 新的毫秒，序列号重置
                self.sequence = 0
            
            self.last_timestamp = timestamp
            
            # 计算相对时间戳
            timestamp_offset = timestamp - self.epoch
            
            # 组装ID
            snowflake_id = (
                (timestamp_offset << self.timestamp_shift) |
                (self.datacenter_id << self.datacenter_id_shift) |
                (self.machine_id << self.machine_id_shift) |
                self.sequence
            )
            
            return snowflake_id
    
    def parse_id(self, snowflake_id):
        """
        解析Snowflake ID
        
        Args:
            snowflake_id: 要解析的ID
            
        Returns:
            dict: 包含时间戳、数据中心ID、机器ID、序列号的字典
        """
        # 提取各部分
        sequence = snowflake_id & self.max_sequence
        machine_id = (snowflake_id >> self.machine_id_shift) & ((1 << self.machine_id_bits) - 1)
        datacenter_id = (snowflake_id >> self.datacenter_id_shift) & ((1 << self.datacenter_id_bits) - 1)
        timestamp_offset = snowflake_id >> self.timestamp_shift
        
        # 计算实际时间戳
        actual_timestamp = timestamp_offset + self.epoch
        
        return {
            'id': snowflake_id,
            'timestamp': actual_timestamp,
            'datetime': datetime.fromtimestamp(actual_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'datacenter_id': datacenter_id,
            'machine_id': machine_id,
            'sequence': sequence
        }
    
    def get_info(self):
        """获取生成器配置信息"""
        return {
            'datacenter_id': self.datacenter_id,
            'machine_id': self.machine_id,
            'epoch': self.epoch,
            'epoch_datetime': datetime.fromtimestamp(self.epoch / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'max_ids_per_ms': self.max_sequence + 1,
            'max_datacenter_id': self.max_datacenter_id,
            'max_machine_id': self.max_machine_id
        }


def test_snowflake():
    """测试Snowflake算法功能"""
    print("=" * 50)
    print("Snowflake算法测试")
    print("=" * 50)
    
    # 创建生成器实例
    generator = SnowflakeIDGenerator(datacenter_id=1, machine_id=2)
    
    # 显示配置信息
    info = generator.get_info()
    print("\n配置信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 生成测试ID
    print("\n生成ID测试:")
    test_ids = []
    for i in range(5):
        snowflake_id = generator.generate_id()
        test_ids.append(snowflake_id)
        print(f"  ID {i+1}: {snowflake_id}")
    
    # 解析测试ID
    print("\nID解析测试:")
    for i, snowflake_id in enumerate(test_ids[:3]):  # 只解析前3个
        parsed = generator.parse_id(snowflake_id)
        print(f"  ID {i+1} ({snowflake_id}) 解析结果:")
        for key, value in parsed.items():
            print(f"    {key}: {value}")
        print()
    
    # 性能测试
    print("性能测试 (生成10000个ID):")
    start_time = time.time()
    ids = set()
    for _ in range(10000):
        snowflake_id = generator.generate_id()
        ids.add(snowflake_id)
    end_time = time.time()
    
    print(f"  生成时间: {end_time - start_time:.4f} 秒")
    print(f"  平均每个ID: {(end_time - start_time) / 10000 * 1000:.4f} 毫秒")
    print(f"  唯一性检查: {'通过' if len(ids) == 10000 else '失败'}")
    print(f"  生成的ID数量: {len(ids)}")
    
    # 排序性测试
    sorted_ids = sorted(list(ids))
    is_monotonic = all(sorted_ids[i] <= sorted_ids[i+1] for i in range(len(sorted_ids)-1))
    print(f"  单调递增性: {'通过' if is_monotonic else '失败'}")


def demo_usage():
    """演示基本用法"""
    print("=" * 50) 
    print("Snowflake使用示例")
    print("=" * 50)
    
    # 基本使用
    print("\n1. 基本使用:")
    generator = SnowflakeIDGenerator()
    for i in range(3):
        snowflake_id = generator.generate_id()
        print(f"   生成ID: {snowflake_id}")
    
    # 自定义配置
    print("\n2. 自定义配置:")
    custom_generator = SnowflakeIDGenerator(datacenter_id=5, machine_id=10)
    custom_id = custom_generator.generate_id()
    parsed = custom_generator.parse_id(custom_id)
    print(f"   自定义生成器ID: {custom_id}")
    print(f"   数据中心ID: {parsed['datacenter_id']}")
    print(f"   机器ID: {parsed['machine_id']}")
    
    # 时间信息
    print("\n3. 时间信息:")
    current_id = generator.generate_id()
    parsed_info = generator.parse_id(current_id)
    print(f"   当前ID: {current_id}")
    print(f"   生成时间: {parsed_info['datetime']}")
    print(f"   时间戳: {parsed_info['timestamp']}")


if __name__ == "__main__":
    # 运行测试
    test_snowflake()
    
    print("\n" + "=" * 50)
    
    # 运行演示
    demo_usage()
    
    print("\n" + "=" * 50)
    print("测试完成！")
