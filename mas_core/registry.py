"""
MAS Core - Central Registry
中心路由：全局路由表，管理各 Agent 的注册与查询
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class ScopeInfo:
    """作用域信息"""
    agent_id: str
    scope_type: str  # 'file', 'class', 'function', 'config'
    scope_name: str
    file_path: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'agent_id': self.agent_id,
            'scope_type': self.scope_type,
            'scope_name': self.scope_name,
            'file_path': self.file_path,
            'variables': list(self.variables.keys())
        }


class CentralRegistry:
    """
    中心注册表
    - 各 Agent 在此注册自己拥有的变量/类解析权
    - 不存储具体代码值，只存储解析权信息
    - 支持 P2P 查询：Agent -> Registry -> Target Agent
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._registry: Dict[str, ScopeInfo] = {}
        self._variable_index: Dict[str, str] = {}  # variable_name -> agent_id
        self._logs: List[str] = []
        self._log("[CentralRegistry] Initialized")
    
    def _log(self, message: str):
        """记录日志"""
        self._logs.append(message)
        print(message)
    
    def register(self, scope_info: ScopeInfo) -> bool:
        """
        注册 Agent 的作用域
        
        Args:
            scope_info: 作用域信息
            
        Returns:
            bool: 是否注册成功
        """
        self._registry[scope_info.agent_id] = scope_info
        
        # 更新变量索引
        for var_name in scope_info.variables:
            self._variable_index[var_name] = scope_info.agent_id
        
        self._log(f"[CentralRegistry] Registered Agent '{scope_info.agent_id}' "
                  f"for scope '{scope_info.scope_name}' ({scope_info.scope_type})")
        return True
    
    def unregister(self, agent_id: str):
        """注销 Agent"""
        if agent_id in self._registry:
            scope = self._registry[agent_id]
            # 清理变量索引
            for var_name in scope.variables:
                if var_name in self._variable_index:
                    del self._variable_index[var_name]
            del self._registry[agent_id]
            self._log(f"[CentralRegistry] Unregistered Agent '{agent_id}'")
    
    def query_variable_owner(self, variable_name: str) -> Optional[str]:
        """
        查询变量的所有者 Agent ID
        
        Args:
            variable_name: 变量名
            
        Returns:
            Optional[str]: Agent ID 或 None
        """
        agent_id = self._variable_index.get(variable_name)
        if agent_id:
            self._log(f"[CentralRegistry] Query '{variable_name}' -> Agent '{agent_id}'")
        else:
            self._log(f"[CentralRegistry] Query '{variable_name}' -> NOT FOUND")
        return agent_id
    
    def get_scope_info(self, agent_id: str) -> Optional[ScopeInfo]:
        """获取 Agent 的作用域信息"""
        return self._registry.get(agent_id)
    
    def list_all_agents(self) -> List[str]:
        """列出所有注册的 Agent"""
        return list(self._registry.keys())
    
    def get_logs(self) -> List[str]:
        """获取所有日志"""
        return self._logs.copy()
    
    def clear_logs(self):
        """清空日志"""
        self._logs.clear()
    
    def print_summary(self):
        """打印注册表摘要"""
        print("\n" + "="*60)
        print("[CentralRegistry Summary]")
        print("="*60)
        print(f"Total Agents: {len(self._registry)}")
        print(f"Total Variables: {len(self._variable_index)}")
        print("\nRegistered Agents:")
        for agent_id, scope in self._registry.items():
            print(f"  - {agent_id}: {scope.scope_name} ({scope.scope_type})")
            print(f"    Variables: {list(scope.variables.keys())}")
        print("="*60 + "\n")


# 全局注册表实例
registry = CentralRegistry()
