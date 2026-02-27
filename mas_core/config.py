"""
NAS CLI Configuration System
配置文件管理系统
"""
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class LLMConfig:
    """LLM 配置"""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    models: List[str] = field(default_factory=lambda: [
        "moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k"
    ])
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.2
    
    def validate(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        if not self.api_key:
            errors.append("LLM API Key 未设置")
        if not self.base_url:
            errors.append("LLM Base URL 未设置")
        return errors


@dataclass
class UIConfig:
    """UI 配置"""
    theme: str = "default"
    show_progress: bool = True
    confirm_before_modify: bool = True
    auto_backup: bool = True
    verbose: bool = False
    language: str = "zh"


@dataclass
class AnalysisConfig:
    """分析配置"""
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", "venv", "env", ".venv", ".env",
        "node_modules", ".pytest_cache", ".mypy_cache"
    ])
    include_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    max_file_size: int = 1024 * 1024  # 1MB
    use_cache: bool = True
    cache_ttl: int = 3600  # 1小时


@dataclass
class NASConfig:
    """NAS 配置"""
    value_keywords: List[str] = field(default_factory=lambda: [
        'lr', 'learning_rate', 'batch_size', 'epoch', 'dropout',
        'dim', 'hidden', 'layer', 'head', 'rate', 'weight_decay',
        'momentum', 'beta', 'gamma', 'alpha', 'num_layers',
        'hidden_dim', 'd_model', 'n_heads', 'max_len'
    ])
    layer_keywords: List[str] = field(default_factory=lambda: [
        'activation', 'optimizer', 'norm', 'loss', 'scheduler'
    ])
    default_search_space_sizes: Dict[str, List[Any]] = field(default_factory=lambda: {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'batch_size': [16, 32, 64, 128, 256],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'num_layers': [1, 2, 3, 4, 6],
        'hidden_dim': [64, 128, 256, 512, 1024],
    })


@dataclass
class Config:
    """完整配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    version: str = "1.3.0"


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG_DIR = Path.home() / ".nas-cli"
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"
    PROJECT_CONFIG_FILE = ".nas-cli.yaml"
    
    def __init__(self):
        self._config: Optional[Config] = None
        self._config_file: Optional[Path] = None
    
    def load_config(self, project_path: Optional[Path] = None) -> Config:
        """
        加载配置，优先级：命令行 > 项目配置 > 用户配置 > 默认配置
        """
        config = Config()
        
        # 1. 加载用户配置
        if self.DEFAULT_CONFIG_FILE.exists():
            config = self._merge_config(config, self._load_yaml(self.DEFAULT_CONFIG_FILE))
        
        # 2. 加载项目配置
        if project_path:
            project_config = project_path / self.PROJECT_CONFIG_FILE
            if project_config.exists():
                config = self._merge_config(config, self._load_yaml(project_config))
        
        # 3. 从环境变量加载
        config = self._load_from_env(config)
        
        self._config = config
        return config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """加载 YAML 文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[ConfigManager] Warning: Failed to load {path}: {e}")
            return {}
    
    def _merge_config(self, base: Config, updates: Dict[str, Any]) -> Config:
        """合并配置"""
        if 'llm' in updates:
            for key, value in updates['llm'].items():
                if hasattr(base.llm, key):
                    setattr(base.llm, key, value)
        
        if 'ui' in updates:
            for key, value in updates['ui'].items():
                if hasattr(base.ui, key):
                    setattr(base.ui, key, value)
        
        if 'analysis' in updates:
            for key, value in updates['analysis'].items():
                if hasattr(base.analysis, key):
                    setattr(base.analysis, key, value)
        
        if 'nas' in updates:
            for key, value in updates['nas'].items():
                if hasattr(base.nas, key):
                    setattr(base.nas, key, value)
        
        return base
    
    def _load_from_env(self, config: Config) -> Config:
        """从环境变量加载配置"""
        # LLM 配置
        if os.getenv('OPENAI_API_KEY'):
            config.llm.api_key = os.getenv('OPENAI_API_KEY')
        if os.getenv('OPENAI_BASE_URL'):
            config.llm.base_url = os.getenv('OPENAI_BASE_URL')
        if os.getenv('NAS_CLI_LLM_TIMEOUT'):
            config.llm.timeout = int(os.getenv('NAS_CLI_LLM_TIMEOUT'))
        if os.getenv('NAS_CLI_LLM_MAX_RETRIES'):
            config.llm.max_retries = int(os.getenv('NAS_CLI_LLM_MAX_RETRIES'))
        
        # UI 配置
        if os.getenv('NAS_CLI_VERBOSE'):
            config.ui.verbose = os.getenv('NAS_CLI_VERBOSE').lower() in ('1', 'true', 'yes')
        if os.getenv('NAS_CLI_LANGUAGE'):
            config.ui.language = os.getenv('NAS_CLI_LANGUAGE')
        
        return config
    
    def save_user_config(self, config: Config) -> bool:
        """保存用户配置"""
        try:
            self.DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            
            config_dict = {
                'version': config.version,
                'llm': {
                    'base_url': config.llm.base_url,
                    'models': config.llm.models,
                    'timeout': config.llm.timeout,
                    'max_retries': config.llm.max_retries,
                    'retry_delay': config.llm.retry_delay,
                    'temperature': config.llm.temperature,
                    # 注意：不保存 api_key 到文件，使用环境变量
                },
                'ui': {
                    'theme': config.ui.theme,
                    'show_progress': config.ui.show_progress,
                    'confirm_before_modify': config.ui.confirm_before_modify,
                    'auto_backup': config.ui.auto_backup,
                    'verbose': config.ui.verbose,
                    'language': config.ui.language,
                },
                'analysis': {
                    'exclude_patterns': config.analysis.exclude_patterns,
                    'include_patterns': config.analysis.include_patterns,
                    'max_file_size': config.analysis.max_file_size,
                    'use_cache': config.analysis.use_cache,
                    'cache_ttl': config.analysis.cache_ttl,
                },
                'nas': {
                    'value_keywords': config.nas.value_keywords,
                    'layer_keywords': config.nas.layer_keywords,
                }
            }
            
            with open(self.DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            return True
        except Exception as e:
            print(f"[ConfigManager] Error saving config: {e}")
            return False
    
    def get_config(self) -> Config:
        """获取当前配置"""
        if self._config is None:
            self.load_config()
        return self._config


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> Config:
    """获取全局配置"""
    return config_manager.get_config()


def load_config(project_path: Optional[Path] = None) -> Config:
    """加载配置"""
    return config_manager.load_config(project_path)
