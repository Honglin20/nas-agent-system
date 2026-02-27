"""
MAS Core - Base Agent
所有 Agent 的基类，定义通用接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import uuid

from .registry import CentralRegistry, ScopeInfo


class BaseAgent(ABC):
    """
    Agent 基类
    - 每个 Agent 绑定一个作用域
    - 支持向 Central Registry 注册
    - 支持 P2P 通信
    """
    
    def __init__(self, scope_type: str, scope_name: str, file_path: Optional[str] = None):
        self.agent_id = f"{scope_type}_{scope_name}_{uuid.uuid4().hex[:8]}"
        self.scope_type = scope_type
        self.scope_name = scope_name
        self.file_path = file_path
        self.registry = CentralRegistry()
        self._logs: List[str] = []
        self._cot: List[str] = []  # Chain of Thought
        
        self._log(f"[Agent:{self.agent_id}] Created for {scope_name}")
    
    def _log(self, message: str):
        """记录日志"""
        self._logs.append(message)
        print(message)
    
    def _think(self, thought: str):
        """记录思考过程 (Chain of Thought)"""
        self._cot.append(thought)
        self._log(f"[CoT:{self.agent_id}] {thought}")
    
    def register_scope(self, variables: Dict[str, Any]):
        """
        向 Central Registry 注册作用域
        
        Args:
            variables: 该作用域拥有的变量字典
        """
        scope_info = ScopeInfo(
            agent_id=self.agent_id,
            scope_type=self.scope_type,
            scope_name=self.scope_name,
            file_path=self.file_path,
            variables=variables
        )
        self.registry.register(scope_info)
        self._think(f"Registered {len(variables)} variables: {list(variables.keys())}")
    
    def query_variable(self, variable_name: str) -> Optional[str]:
        """
        查询变量的所有者
        
        Args:
            variable_name: 变量名
            
        Returns:
            Optional[str]: 拥有该变量的 Agent ID
        """
        self._think(f"Querying ownership of variable '{variable_name}'")
        return self.registry.query_variable_owner(variable_name)
    
    def fetch_from_agent(self, target_agent_id: str, query: str) -> Any:
        """
        P2P 通信：向目标 Agent 发起查询
        
        Args:
            target_agent_id: 目标 Agent ID
            query: 查询内容
            
        Returns:
            Any: 查询结果
        """
        self._think(f"Initiating P2P fetch to '{target_agent_id}' for '{query}'")
        
        # 这里模拟 P2P 通信
        # 实际实现中，这会是一个异步调用或消息传递
        target_scope = self.registry.get_scope_info(target_agent_id)
        
        if target_scope is None:
            self._think(f"ERROR: Target agent '{target_agent_id}' not found")
            return None
        
        # 模拟从目标 Agent 获取值
        result = self._simulate_p2p_response(target_scope, query)
        self._think(f"P2P response received: {result}")
        return result
    
    def _simulate_p2p_response(self, target_scope: ScopeInfo, query: str) -> Any:
        """模拟 P2P 响应（子类可重写）"""
        # 基础实现：检查变量是否在目标作用域中
        if query in target_scope.variables:
            return target_scope.variables.get(query)
        return None
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        分析作用域内容
        
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
    
    @abstractmethod
    def get_nas_candidates(self) -> List[Dict[str, Any]]:
        """
        获取 NAS 寻优候选参数
        
        Returns:
            List[Dict[str, Any]]: 候选参数列表
        """
        pass
    
    def get_logs(self) -> List[str]:
        """获取日志"""
        return self._logs.copy()
    
    def get_cot(self) -> List[str]:
        """获取思考链"""
        return self._cot.copy()
