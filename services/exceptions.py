# -*- coding:utf-8 -*-
"""
自定义业务异常模块
用于区分业务异常和系统异常
"""


class BusinessError(Exception):
    """业务逻辑异常"""

    def __init__(self, message: str, code: int = 1, status_code: int = 400):
        """
        初始化业务异常

        Args:
            message: 错误信息
            code: 业务错误码
            status_code: HTTP 状态码
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code

    def to_dict(self) -> dict:
        """转换为 API 响应格式"""
        return {
            "code": self.code,
            "data": {"message": self.message}
        }


class ValidationError(BusinessError):
    """输入验证异常"""

    def __init__(self, message: str):
        super().__init__(message, code=1, status_code=400)


class NotFoundError(BusinessError):
    """资源未找到异常"""

    def __init__(self, message: str):
        super().__init__(message, code=1, status_code=404)


class DuplicateTaskError(BusinessError):
    """重复任务异常"""

    def __init__(self, message: str):
        super().__init__(message, code=0, status_code=200)
