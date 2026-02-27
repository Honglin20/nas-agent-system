"""
NAS CLI Error Handling System
错误处理系统
"""
from enum import Enum, auto
from typing import Optional, Dict, Any


class ErrorCode(Enum):
    """错误码枚举"""
    # 系统错误 (1-99)
    UNKNOWN_ERROR = 1
    INITIALIZATION_ERROR = 2
    CONFIG_ERROR = 3
    
    # 文件错误 (100-199)
    FILE_NOT_FOUND = 100
    FILE_PERMISSION_DENIED = 101
    FILE_READ_ERROR = 102
    FILE_WRITE_ERROR = 103
    FILE_TOO_LARGE = 104
    INVALID_PATH = 105
    BACKUP_ERROR = 106
    
    # LLM 错误 (200-299)
    LLM_NOT_INITIALIZED = 200
    LLM_API_ERROR = 201
    LLM_TIMEOUT = 202
    LLM_RATE_LIMIT = 203
    LLM_INVALID_RESPONSE = 204
    LLM_CONTEXT_TOO_LONG = 205
    LLM_ALL_MODELS_FAILED = 206
    LLM_CONNECTION_ERROR = 207  # v1.3.1: 连接错误
    LLM_AUTHENTICATION_ERROR = 208  # v1.3.1: 认证错误
    
    # 代码分析错误 (300-399)
    PARSE_ERROR = 300
    AST_ERROR = 301
    CST_ERROR = 302
    INVALID_CODE = 303
    
    # 修改错误 (400-499)
    MODIFICATION_FAILED = 400
    ROLLBACK_FAILED = 401
    INVALID_MODIFICATION = 402
    
    # 用户输入错误 (500-599)
    INVALID_INPUT = 500
    CANCELLED_BY_USER = 501
    

class NASCLIError(Exception):
    """NAS CLI 基础异常类"""
    
    def __init__(self, 
                 code: ErrorCode, 
                 message: str, 
                 details: Optional[Dict[str, Any]] = None,
                 original_error: Optional[Exception] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)
    
    def __str__(self) -> str:
        error_str = f"[{self.code.name}] {self.message}"
        if self.details:
            error_str += f"\nDetails: {self.details}"
        if self.original_error:
            error_str += f"\nOriginal error: {self.original_error}"
        return error_str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'code': self.code.value,
            'code_name': self.code.name,
            'message': self.message,
            'details': self.details,
            'original_error': str(self.original_error) if self.original_error else None
        }


class FileError(NASCLIError):
    """文件操作错误"""
    def __init__(self, 
                 code: ErrorCode,
                 message: str,
                 file_path: Optional[str] = None,
                 **kwargs):
        details = kwargs.pop('details', {})
        if file_path:
            details['file_path'] = file_path
        super().__init__(code, message, details, **kwargs)


class LLMError(NASCLIError):
    """LLM 调用错误"""
    def __init__(self,
                 code: ErrorCode,
                 message: str,
                 model: Optional[str] = None,
                 **kwargs):
        details = kwargs.pop('details', {})
        if model:
            details['model'] = model
        super().__init__(code, message, details, **kwargs)


class ValidationError(NASCLIError):
    """验证错误"""
    def __init__(self,
                 message: str,
                 field: Optional[str] = None,
                 **kwargs):
        details = kwargs.pop('details', {})
        if field:
            details['field'] = field
        super().__init__(ErrorCode.INVALID_INPUT, message, details, **kwargs)


# 便捷函数
def raise_file_not_found(path: str, original_error: Optional[Exception] = None):
    """抛出文件未找到错误"""
    raise FileError(
        ErrorCode.FILE_NOT_FOUND,
        f"文件未找到: {path}",
        file_path=path,
        original_error=original_error
    )


def raise_file_permission_denied(path: str, original_error: Optional[Exception] = None):
    """抛出文件权限错误"""
    raise FileError(
        ErrorCode.FILE_PERMISSION_DENIED,
        f"文件权限不足: {path}",
        file_path=path,
        original_error=original_error
    )


def raise_llm_timeout(model: str, timeout: int, original_error: Optional[Exception] = None):
    """抛出 LLM 超时错误"""
    raise LLMError(
        ErrorCode.LLM_TIMEOUT,
        f"LLM 调用超时 ({timeout}s): {model}",
        model=model,
        original_error=original_error,
        details={'timeout': timeout}
    )


def raise_llm_api_error(model: str, message: str, original_error: Optional[Exception] = None):
    """抛出 LLM API 错误"""
    raise LLMError(
        ErrorCode.LLM_API_ERROR,
        f"LLM API 错误: {message}",
        model=model,
        original_error=original_error
    )


# 用户友好的错误消息映射
USER_FRIENDLY_MESSAGES = {
    ErrorCode.UNKNOWN_ERROR: "发生未知错误，请查看日志获取详细信息。",
    ErrorCode.FILE_NOT_FOUND: "找不到指定的文件或目录，请检查路径是否正确。",
    ErrorCode.FILE_PERMISSION_DENIED: "权限不足，无法访问文件。请检查文件权限。",
    ErrorCode.FILE_READ_ERROR: "读取文件时出错，文件可能已损坏。",
    ErrorCode.FILE_WRITE_ERROR: "写入文件时出错，请检查磁盘空间和权限。",
    ErrorCode.LLM_NOT_INITIALIZED: "LLM 客户端未初始化，请检查 API Key 配置。",
    ErrorCode.LLM_API_ERROR: "调用 LLM API 时出错，请检查网络连接和 API Key。",
    ErrorCode.LLM_TIMEOUT: "LLM 调用超时，请稍后重试或增加超时时间。",
    ErrorCode.LLM_RATE_LIMIT: "API 调用频率超限，请稍后重试。",
    ErrorCode.LLM_ALL_MODELS_FAILED: "所有 LLM 模型都调用失败，请检查配置。",
    ErrorCode.LLM_CONNECTION_ERROR: "无法连接到 LLM 服务，请检查网络连接或配置代理。",  # v1.3.1
    ErrorCode.LLM_AUTHENTICATION_ERROR: "LLM API Key 验证失败，请检查 API Key 是否正确。",  # v1.3.1
    ErrorCode.PARSE_ERROR: "代码解析错误，请检查代码语法。",
    ErrorCode.MODIFICATION_FAILED: "代码修改失败，已自动回滚到修改前状态。",
    ErrorCode.INVALID_INPUT: "输入无效，请检查输入参数。",
    ErrorCode.CANCELLED_BY_USER: "操作已取消。",
}


def get_user_friendly_message(error: NASCLIError) -> str:
    """获取用户友好的错误消息"""
    return USER_FRIENDLY_MESSAGES.get(error.code, error.message)
