"""
MAS Core - Package Init (v1.3.0 Enhanced)
"""
from .registry import CentralRegistry, ScopeInfo
from .base_agent import BaseAgent
from .scope_agent import ScopeAgent
from .cli_agent import CLIUIAgent
from .modifier_agent import ModifierAgent
from .orchestrator import NASOrchestrator
from .llm_client import LLMClient, MockLLMClient, init_llm, get_llm_client, is_llm_available, BaseLLMClient

# v1.2.0 新增模块
from .model_discovery import (
    ModelDiscoveryAnalyzer, 
    discover_models_in_project
)
from .cross_file_modifier import (
    CrossFileParameterModifier,
    ConfigFileHandler
)
from .report_injector import (
    ReportInjector,
    inject_report_to_project
)
from .search_space_expander import (
    SearchSpaceExpander,
    expand_search_space_in_project
)

# v1.3.0 新增模块
from .config import (
    Config, ConfigManager, 
    LLMConfig, UIConfig, AnalysisConfig, NASConfig,
    get_config, load_config, config_manager
)
from .exceptions import (
    NASCLIError, FileError, LLMError, ValidationError,
    ErrorCode, get_user_friendly_message
)
from .retry_cache import (
    RetryConfig, retry_with_backoff,
    Cache, FileCache, cached, CircuitBreaker,
    get_global_cache
)
from .backup import (
    BackupManager, Operation, FileChange
)

__version__ = "1.3.0"

__all__ = [
    # 基础组件
    'CentralRegistry',
    'ScopeInfo',
    'BaseAgent',
    'ScopeAgent',
    'CLIUIAgent',
    'ModifierAgent',
    'NASOrchestrator',
    'LLMClient',
    'MockLLMClient',
    'BaseLLMClient',
    'init_llm',
    'get_llm_client',
    'is_llm_available',
    
    # v1.2.0 新增
    'ModelDiscoveryAnalyzer',
    'discover_models_in_project',
    'CrossFileParameterModifier',
    'ConfigFileHandler',
    'ReportInjector',
    'inject_report_to_project',
    'SearchSpaceExpander',
    'expand_search_space_in_project',
    
    # v1.3.0 新增 - 配置
    'Config',
    'ConfigManager',
    'LLMConfig',
    'UIConfig',
    'AnalysisConfig',
    'NASConfig',
    'get_config',
    'load_config',
    'config_manager',
    
    # v1.3.0 新增 - 异常
    'NASCLIError',
    'FileError',
    'LLMError',
    'ValidationError',
    'ErrorCode',
    'get_user_friendly_message',
    
    # v1.3.0 新增 - 重试和缓存
    'RetryConfig',
    'retry_with_backoff',
    'Cache',
    'FileCache',
    'cached',
    'CircuitBreaker',
    'get_global_cache',
    
    # v1.3.0 新增 - 备份
    'BackupManager',
    'Operation',
    'FileChange',
    
    # 版本
    '__version__',
]
