"""
MAS Core - Main Orchestrator
主协调器：协调所有 Agent 完成 NAS 注入任务
"""
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from .registry import CentralRegistry
from .scope_agent import ScopeAgent
from .cli_agent import CLIUIAgent
from .modifier_agent import ModifierAgent
from .llm_client import get_llm_client


class NASOrchestrator:
    """
    NAS 注入任务协调器
    - 管理所有 Agent 的生命周期
    - 协调跨 Agent 通信
    - 执行完整的 NAS 注入流程
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.registry = CentralRegistry()
        self.ui_agent = CLIUIAgent()
        self.modifier_agent = ModifierAgent()
        self.scope_agents: Dict[str, ScopeAgent] = {}
        
        print(f"[NASOrchestrator] Initialized for project: {project_path}")
    
    def scan_project(self) -> List[str]:
        """
        扫描项目，发现所有 Python 文件
        
        Returns:
            List[str]: Python 文件路径列表
        """
        print("\n[Phase 1] Scanning project...")
        print(f"  Path: {self.project_path}")
        print(f"  Exists: {self.project_path.exists()}")
        
        python_files = []
        
        # 确保路径存在
        if not self.project_path.exists():
            print(f"  ERROR: Path does not exist!")
            return python_files
        
        # 遍历查找 Python 文件
        for py_file in self.project_path.rglob("*.py"):
            py_path = Path(py_file)
            # 排除常见目录
            skip = False
            for part in py_path.parts:
                if part.startswith('.') or part in ['__pycache__', 'venv', 'env', 'mas_core', 'tests']:
                    skip = True
                    break
            if skip:
                continue
            python_files.append(str(py_path))
            print(f"  Found: {py_path.name}")
        
        print(f"Found {len(python_files)} Python files")
        return python_files
    
    def create_scope_agents(self, file_paths: List[str]):
        """
        为每个文件创建 Scope Agent
        
        Args:
            file_paths: Python 文件路径列表
        """
        print("\n[Phase 2] Creating Scope Agents...")
        
        for file_path in file_paths:
            agent = ScopeAgent(file_path)
            
            # 加载并分析文件
            if agent.load_file():
                analysis = agent.analyze()
                
                # 注册到 Central Registry
                variables = {c['name']: c['current_value'] 
                           for c in analysis['nas_candidates']}
                agent.register_scope(variables)
                
                self.scope_agents[agent.agent_id] = agent
        
        print(f"Created {len(self.scope_agents)} Scope Agents")
    
    def run_p2p_resolution(self):
        """
        运行 P2P 变量解析
        处理跨文件引用和动态反射
        """
        print("\n[Phase 3] P2P Variable Resolution...")
        
        for agent_id, agent in self.scope_agents.items():
            # 检查是否有动态引用需要解析
            source = agent.get_source_code()
            
            if 'getattr' in source:
                print(f"\n[Dynamic Reflection] Found in {agent.file_path}")
                # 尝试解析动态引用
                # 这里需要 LLM 帮助解析
                llm = get_llm_client()
                resolution = llm.resolve_dynamic_reference(source, "model_class")
                print(f"  LLM Resolution: {resolution}")
    
    def collect_nas_candidates(self) -> List[Dict[str, Any]]:
        """
        收集所有 NAS 候选参数
        
        Returns:
            List[Dict[str, Any]]: 合并后的候选列表
        """
        print("\n[Phase 4] Collecting NAS Candidates...")
        
        all_candidates = []
        for agent_id, agent in self.scope_agents.items():
            candidates = agent.get_nas_candidates()
            for cand in candidates:
                cand['source_file'] = agent.file_path
                cand['source_agent'] = agent_id
            all_candidates.extend(candidates)
        
        print(f"Collected {len(all_candidates)} NAS candidates")
        return all_candidates
    
    def run_user_interaction(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        运行用户交互流程
        
        Args:
            candidates: NAS 候选参数
            
        Returns:
            List[Dict[str, Any]]: 用户选择的候选
        """
        print("\n[Phase 5] User Interaction...")
        
        # 展示候选
        self.ui_agent.display_candidates(candidates)
        
        # 获取用户选择
        selected = self.ui_agent.select_candidates(candidates)
        
        # 确认注入
        if self.ui_agent.get_user_confirmation("Confirm NAS injection?"):
            return selected
        else:
            print("User cancelled injection")
            return []
    
    def generate_modifications(self, selected: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        生成修改计划
        
        Args:
            selected: 用户选择的候选
            
        Returns:
            Dict[str, List[Dict]]: 按文件分组的修改计划
        """
        print("\n[Phase 6] Generating Modifications...")
        
        modifications_by_file: Dict[str, List[Dict]] = {}
        
        for cand in selected:
            file_path = cand['source_file']
            
            if file_path not in modifications_by_file:
                modifications_by_file[file_path] = []
            
            # 根据类型生成修改
            if cand['type'] == 'value':
                mod = {
                    'type': 'value_space',
                    'target': cand['name'],
                    'search_space': [cand['current_value'] // 2, 
                                   cand['current_value'],
                                   cand['current_value'] * 2],
                    'line': cand.get('line', 0)
                }
            elif cand['type'] == 'layer':
                mod = {
                    'type': 'layer_space',
                    'target': cand['name'],
                    'layer_options': ['nn.ReLU()', 'nn.Sigmoid()', 'nn.Tanh()'],
                    'line': cand.get('line', 0)
                }
            else:
                continue
            
            modifications_by_file[file_path].append(mod)
        
        # 为每个训练文件添加 report 注入
        for file_path, mods in modifications_by_file.items():
            # 检查是否是训练文件
            agent = self._get_agent_by_file(file_path)
            if agent and 'train' in agent.get_source_code().lower():
                mods.append({
                    'type': 'report',
                    'metrics': {'loss': 'loss', 'accuracy': 'accuracy'}
                })
        
        # 展示修改计划
        all_mods = []
        for file_path, mods in modifications_by_file.items():
            for mod in mods:
                mod['file'] = file_path
                all_mods.append(mod)
        
        self.ui_agent.show_modification_plan(all_mods)
        
        return modifications_by_file
    
    def apply_modifications(self, modifications_by_file: Dict[str, List[Dict]]):
        """
        应用修改到文件
        
        Args:
            modifications_by_file: 按文件分组的修改计划
        """
        print("\n[Phase 7] Applying Modifications...")
        
        success_count = 0
        total_files = len(modifications_by_file)
        
        for i, (file_path, mods) in enumerate(modifications_by_file.items(), 1):
            self.ui_agent.show_progress(i, total_files, f"Processing {Path(file_path).name}")
            
            if self.modifier_agent.apply_modifications(file_path, mods):
                success_count += 1
                print(f"  ✓ Modified: {file_path}")
            else:
                print(f"  ✗ Failed: {file_path}")
        
        print(f"\nSuccessfully modified {success_count}/{total_files} files")
    
    def _get_agent_by_file(self, file_path: str) -> Optional[ScopeAgent]:
        """根据文件路径获取 Agent"""
        for agent in self.scope_agents.values():
            if agent.file_path == file_path:
                return agent
        return None
    
    def run(self):
        """运行完整的 NAS 注入流程"""
        print("\n" + "="*70)
        print("🚀 NAS Agent System - Starting Injection Process")
        print("="*70)
        
        # Phase 1: 扫描项目
        files = self.scan_project()
        
        if not files:
            print("No Python files found!")
            return
        
        # Phase 2: 创建 Scope Agents
        self.create_scope_agents(files)
        
        # Phase 3: P2P 解析
        self.run_p2p_resolution()
        
        # Phase 4: 收集候选
        candidates = self.collect_nas_candidates()
        
        if not candidates:
            print("No NAS candidates found!")
            return
        
        # Phase 5: 用户交互
        selected = self.run_user_interaction(candidates)
        
        if not selected:
            print("No candidates selected for injection")
            return
        
        # Phase 6: 生成修改计划
        modifications = self.generate_modifications(selected)
        
        # Phase 7: 应用修改
        self.apply_modifications(modifications)
        
        # 完成
        print("\n" + "="*70)
        print("✅ NAS Injection Complete!")
        print("="*70)
        
        # 打印注册表摘要
        self.registry.print_summary()
