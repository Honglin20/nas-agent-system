"""
MAS Core - Scope Agent (v1.4.0 Fixed)
作用域智能体：所有分析通过真实 LLM 完成，严禁规则模拟
- 参数过滤：只推荐模型超参和结构参数
"""
import ast
from typing import Dict, Any, List, Optional
import libcst as cst

from .base_agent import BaseAgent
from .llm_client import get_llm_client, filter_nas_candidates


class ScopeAgent(BaseAgent):
    """
    作用域智能体 - v1.4.0
    使用真实 LLM 分析代码，只推荐模型结构参数
    """
    
    def __init__(self, file_path: str, scope_name: str = "global"):
        super().__init__(
            scope_type="file",
            scope_name=scope_name,
            file_path=file_path
        )
        self.file_path = file_path
        self.source_code: Optional[str] = None
        self.cst_tree: Optional[cst.Module] = None
        self._nas_candidates: List[Dict[str, Any]] = []
        
    def load_file(self) -> bool:
        """加载并解析文件"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.source_code = f.read()
            
            self.cst_tree = cst.parse_module(self.source_code)
            
            self._think(f"Loaded file: {self.file_path}")
            self._think(f"Code lines: {len(self.source_code.splitlines())}")
            return True
            
        except Exception as e:
            self._think(f"ERROR loading file: {e}")
            return False
    
    def analyze(self) -> Dict[str, Any]:
        """
        v1.4.0: 使用真实 LLM 分析代码，过滤训练参数
        """
        if self.cst_tree is None:
            self.load_file()
        
        self._think("Starting LLM-based code analysis...")
        
        # 使用真实 LLM 分析
        llm = get_llm_client()
        candidates = llm.analyze_code_for_nas(self.source_code, self.file_path)
        
        # v1.4.0: 再次应用参数过滤（双重保险）
        self._nas_candidates = filter_nas_candidates(candidates)
        
        # 基础 AST 分析（仅用于展示，不参与 NAS 决策）
        ast_tree = ast.parse(self.source_code)
        classes = []
        functions = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                classes.append({'name': node.name, 'line': node.lineno})
            elif isinstance(node, ast.FunctionDef):
                functions.append({'name': node.name, 'line': node.lineno})
        
        result = {
            'file_path': self.file_path,
            'classes': classes,
            'functions': functions,
            'nas_candidates': self._nas_candidates,
            'llm_analyzed': True
        }
        
        self._think(f"LLM analysis complete: {len(candidates)} raw candidates")
        self._think(f"After filtering: {len(self._nas_candidates)} model structure candidates")
        for cand in self._nas_candidates[:3]:
            self._think(f"  - {cand.get('name')}: {cand.get('current_value')}")
        
        return result
    
    def get_nas_candidates(self) -> List[Dict[str, Any]]:
        """获取 NAS 候选参数"""
        if not self._nas_candidates:
            self.analyze()
        return self._nas_candidates
    
    def get_source_code(self) -> str:
        """获取源代码"""
        return self.source_code or ""
    
    def get_cst_tree(self) -> Optional[cst.Module]:
        """获取 libcst 解析树"""
        return self.cst_tree
