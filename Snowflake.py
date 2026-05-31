#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容旧导入路径的 Snowflake ID 生成器入口。"""

from utils.snowflake import SnowflakeIDGenerator

__all__ = ['SnowflakeIDGenerator']


def main():
    generator = SnowflakeIDGenerator(datacenter_id=1, machine_id=2)
    print(generator.generate_id())


if __name__ == "__main__":
    main()
