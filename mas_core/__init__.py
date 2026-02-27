"""
MAS Core - Package Init (v1.2.0 Enhanced)
"""
from .registry import CentralRegistry, ScopeInfo
from .base_agent import BaseAgent
from .scope_agent import ScopeAgent
from .cli_agent import CLIUIAgent
from .modifier_agent import ModifierAgent
from .orchestrator import NASOrchestrator
from .llm_client import LLMClient, init_llm, get_llm_client

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

__all__ = [
    'CentralRegistry',
    'ScopeInfo',
    'BaseAgent',
    'ScopeAgent',
    'CLIUIAgent',
    'ModifierAgent',
    'NASOrchestrator',
    'LLMClient',
    'init_llm',
    'get_llm_client',
    # v1.2.0 新增
    'ModelDiscoveryAnalyzer',
    'discover_models_in_project',
    'CrossFileParameterModifier',
    'ConfigFileHandler',
    'ReportInjector',
    'inject_report_to_project',
    'SearchSpaceExpander',
    'expand_search_space_in_project',
]
