"""
MAS Core - Main Orchestrator (v1.4.0 Fixed)
修复版主协调器：
- 智能模型识别（动态反射解析）
- 跨文件参数修改
- LLM 驱动的 Report 插入
- 寻优空间张开
- 只修改 backbone 主模型的 __init__ 方法
- Level 4 YAML 配置文件支持
"""
import os
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .registry import CentralRegistry
from .scope_agent import ScopeAgent
from .cli_agent import CLIUIAgent
from .modifier_agent import ModifierAgent
from .llm_client import get_llm_client, filter_nas_candidates, is_excluded_param

# v1.4.0 新增导入
from .model_discovery import ModelDiscoveryAnalyzer
from .cross_file_modifier import CrossFileParameterModifier, ConfigFileHandler
from .report_injector import ReportInjector, inject_report_to_project
from .search_space_expander import SearchSpaceExpander
from .backup import BackupManager


class NASOrchestrator:
    """
    修复版 NAS 注入任务协调器 (v1.4.0)
    """
    
    def __init__(self, project_path: str, entry_file: str = None):
        self.project_path = Path(project_path)
        self.entry_file = entry_file
        self.registry = CentralRegistry()
        self.ui_agent = CLIUIAgent()
        self.modifier_agent = ModifierAgent()
        self.scope_agents: Dict[str, ScopeAgent] = {}
        
        # v1.4.0 新增组件
        self.llm_client = get_llm_client() if self._is_llm_available() else None
        self.model_discovery = ModelDiscoveryAnalyzer(project_path, self.llm_client)
        self.cross_file_modifier = CrossFileParameterModifier(project_path)
        self.report_injector = ReportInjector(self.llm_client)
        self.search_space_expander = SearchSpaceExpander(self.llm_client)
        self.backup_manager = BackupManager(project_path)
        
        # v1.4.0: 存储发现的模型信息
        self.discovered_models = []
        self.instantiated_model = None
        self.backbone_class_name = None
        
        print(f"[NASOrchestrator v1.4.0] Initialized for project: {project_path}")
    
    def _is_llm_available(self) -> bool:
        """检查 LLM 是否可用"""
        try:
            get_llm_client()
            return True
        except:
            return False
    
    def scan_project(self) -> List[str]:
        """扫描项目，发现所有 Python 文件"""
        print("\n[Phase 1] Scanning project...")
        print(f"  Path: {self.project_path}")
        
        python_files = []
        
        if not self.project_path.exists():
            print(f"  ERROR: Path does not exist!")
            return python_files
        
        for py_file in self.project_path.rglob("*.py"):
            py_path = Path(py_file)
            skip = False
            for part in py_path.parts:
                if part.startswith('.') or part in ['__pycache__', 'venv', 'env', 'mas_core', 'tests']:
                    skip = True
                    break
            if skip:
                continue
            python_files.append(str(py_path))
        
        print(f"Found {len(python_files)} Python files")
        return python_files
    
    def create_scope_agents(self, file_paths: List[str]):
        """为每个文件创建 Scope Agent"""
        print("\n[Phase 2] Creating Scope Agents...")
        
        for file_path in file_paths:
            agent = ScopeAgent(file_path)
            
            if agent.load_file():
                analysis = agent.analyze()
                
                variables = {c['name']: c['current_value'] 
                           for c in analysis['nas_candidates']}
                agent.register_scope(variables)
                
                self.scope_agents[agent.agent_id] = agent
        
        print(f"Created {len(self.scope_agents)} Scope Agents")
    
    def run_model_discovery(self) -> Dict[str, Any]:
        """
        v1.4.0: 运行智能模型发现
        """
        print("\n[Phase 3] Smart Model Discovery...")
        
        if not self.entry_file:
            print("  No entry file specified, skipping model discovery")
            return {}
        
        entry_path = self.project_path / self.entry_file
        if not entry_path.exists():
            print(f"  Entry file not found: {entry_path}")
            return {}
        
        result = self.model_discovery.run_full_discovery(entry_path)
        
        # 存储发现的模型信息
        self.discovered_models = result.get("all_models", [])
        self.instantiated_model = result.get("instantiated_model")
        
        # v1.4.0: 设置 backbone 类名
        if self.instantiated_model:
            self.backbone_class_name = self.instantiated_model.get("instantiated_model")
            print(f"[NASOrchestrator] Backbone model identified: {self.backbone_class_name}")
            # 设置到 modifier agent
            self.modifier_agent.set_backbone_class(self.backbone_class_name)
        
        return result
    
    def run_p2p_resolution(self):
        """运行 P2P 变量解析"""
        print("\n[Phase 4] P2P Variable Resolution...")
        
        for agent_id, agent in self.scope_agents.items():
            source = agent.get_source_code()
            
            if 'getattr' in source:
                print(f"\n[Dynamic Reflection] Found in {agent.file_path}")
                
                if self.llm_client:
                    resolution = self.llm_client.resolve_dynamic_reference(source, "model_class")
                    print(f"  LLM Resolution: {resolution}")
    
    def collect_nas_candidates(self) -> List[Dict[str, Any]]:
        """
        v1.4.0: 收集所有 NAS 候选参数
        应用参数过滤，只保留模型结构参数
        """
        print("\n[Phase 5] Collecting NAS Candidates...")
        
        all_candidates = []
        
        # 从 Python 文件收集
        for agent_id, agent in self.scope_agents.items():
            candidates = agent.get_nas_candidates()
            for cand in candidates:
                cand['source_file'] = agent.file_path
                cand['source_agent'] = agent_id
            all_candidates.extend(candidates)
        
        # v1.4.0: 从 Python 配置文件收集
        config_candidates = self._collect_config_candidates()
        all_candidates.extend(config_candidates)
        
        # v1.4.0: 从 YAML 配置文件收集
        yaml_candidates = self._collect_yaml_candidates()
        all_candidates.extend(yaml_candidates)
        
        # v1.4.0: 应用参数过滤
        print(f"\n[Phase 5.1] Filtering candidates...")
        print(f"  Before filtering: {len(all_candidates)} candidates")
        filtered_candidates = filter_nas_candidates(all_candidates)
        print(f"  After filtering: {len(filtered_candidates)} candidates")
        
        return filtered_candidates
    
    def _collect_config_candidates(self) -> List[Dict[str, Any]]:
        """v1.4.0: 从 Python 配置文件中收集候选参数"""
        candidates = []
        
        # 查找 Python 配置文件
        for config_file in self.project_path.rglob("*_config.py"):
            if any(part.startswith('.') for part in config_file.parts):
                continue
            
            try:
                config = ConfigFileHandler.load_config(config_file)
                
                # 递归查找数值参数
                self._extract_candidates_from_dict(
                    config, 
                    str(config_file.relative_to(self.project_path)),
                    candidates
                )
            except Exception as e:
                print(f"  Error loading config {config_file}: {e}")
        
        return candidates
    
    def _collect_yaml_candidates(self) -> List[Dict[str, Any]]:
        """
        v1.4.0: 从 YAML 配置文件中收集候选参数
        """
        candidates = []
        
        # 查找 YAML 配置文件
        for yaml_file in self.project_path.rglob("*.yaml"):
            if any(part.startswith('.') for part in yaml_file.parts):
                continue
            if '.nas_backup' in str(yaml_file):
                continue
            
            try:
                import yaml
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    yaml_content = f.read()
                    yaml_data = yaml.safe_load(yaml_content)
                
                if not yaml_data:
                    continue
                
                print(f"[NASOrchestrator] Found YAML config: {yaml_file}")
                
                # 使用 LLM 分析 YAML
                if self.llm_client:
                    yaml_candidates = self.llm_client.analyze_yaml_config_for_nas(
                        yaml_content,
                        str(yaml_file.relative_to(self.project_path))
                    )
                    
                    for cand in yaml_candidates:
                        cand['source_file'] = str(yaml_file.relative_to(self.project_path))
                        cand['is_yaml'] = True
                    
                    candidates.extend(yaml_candidates)
                else:
                    # 手动提取
                    self._extract_from_yaml_dict(
                        yaml_data,
                        str(yaml_file.relative_to(self.project_path)),
                        candidates
                    )
                    
            except Exception as e:
                print(f"  Error loading YAML {yaml_file}: {e}")
        
        return candidates
    
    def _extract_from_yaml_dict(self, data: Dict, file_path: str, 
                                 candidates: List, prefix: str = ""):
        """从 YAML 字典中提取候选参数"""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._extract_from_yaml_dict(value, file_path, candidates, full_key)
            elif isinstance(value, (int, float)):
                # 检查是否是模型结构参数
                if is_excluded_param(key):
                    continue
                
                # 检查是否是推荐的模型参数
                if any(kw in key.lower() for kw in ['dim', 'layer', 'head', 'dropout', 'depth']):
                    candidates.append({
                        'name': full_key,
                        'type': 'value',
                        'current_value': value,
                        'source_file': file_path,
                        'yaml_path': full_key.split('.'),
                        'is_yaml': True,
                        'reason': f'YAML config parameter: {key}'
                    })
            elif isinstance(value, str):
                if key.lower() in ['activation', 'norm_type', 'norm']:
                    candidates.append({
                        'name': full_key,
                        'type': 'layer',
                        'current_value': value,
                        'source_file': file_path,
                        'yaml_path': full_key.split('.'),
                        'is_yaml': True,
                        'reason': f'YAML layer selection: {key}'
                    })
    
    def _extract_candidates_from_dict(self, data: Dict, file_path: str, 
                                       candidates: List, prefix: str = ""):
        """从字典中提取候选参数"""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._extract_candidates_from_dict(value, file_path, candidates, full_key)
            elif isinstance(value, (int, float)):
                # v1.4.0: 检查是否是训练参数
                if is_excluded_param(key):
                    continue
                
                # 检查是否是模型结构参数
                if any(kw in key.lower() for kw in ['dim', 'layer', 'head', 'dropout', 'depth']):
                    candidates.append({
                        'name': full_key,
                        'type': 'value',
                        'current_value': value,
                        'source_file': file_path,
                        'config_key': full_key,
                        'reason': f'Config parameter: {key}'
                    })
            elif isinstance(value, str):
                if key.lower() in ['activation', 'norm_type', 'norm']:
                    candidates.append({
                        'name': full_key,
                        'type': 'layer',
                        'current_value': value,
                        'source_file': file_path,
                        'config_key': full_key,
                        'reason': f'Layer selection: {key}'
                    })
    
    def run_user_interaction(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """运行用户交互流程"""
        print("\n[Phase 6] User Interaction...")
        
        self.ui_agent.display_candidates(candidates)
        
        selected = self.ui_agent.select_candidates(candidates)
        
        if self.ui_agent.get_user_confirmation("Confirm NAS injection?"):
            return selected
        else:
            print("User cancelled injection")
            return []
    
    def generate_modifications(self, selected: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        v1.4.0: 生成修改计划
        只修改 backbone 主模型的 __init__ 方法
        """
        print("\n[Phase 7] Generating Modifications...")
        
        # v1.4.0: 如果只修改 backbone，显示提示
        if self.backbone_class_name:
            print(f"  Target backbone class: {self.backbone_class_name}")
        
        modifications_by_file: Dict[str, List[Dict]] = {}
        
        for cand in selected:
            file_path = cand.get('source_file', '')
            
            # 处理 YAML 配置中的参数
            if cand.get('is_yaml'):
                file_path = str(self.project_path / file_path)
                if file_path not in modifications_by_file:
                    modifications_by_file[file_path] = []
                
                if cand['type'] == 'value':
                    mod = {
                        'type': 'yaml_value_space',
                        'target': cand.get('yaml_path', [cand['name']]),
                        'param_name': cand['name'],
                        'search_space': self._generate_search_space(cand),
                        'is_yaml': True
                    }
                elif cand['type'] == 'layer':
                    mod = {
                        'type': 'yaml_layer_space',
                        'target': cand.get('yaml_path', [cand['name']]),
                        'param_name': cand['name'],
                        'layer_options': self._generate_layer_options(cand),
                        'is_yaml': True
                    }
                else:
                    continue
                
                modifications_by_file[file_path].append(mod)
                continue
            
            # 处理 Python 配置文件中的参数
            if 'config_key' in cand:
                file_path = str(self.project_path / file_path)
            else:
                # 确保是绝对路径
                if not Path(file_path).is_absolute():
                    file_path = str(self.project_path / file_path)
            
            if not file_path:
                continue
            
            if file_path not in modifications_by_file:
                modifications_by_file[file_path] = []
            
            # 生成修改
            if cand['type'] == 'value':
                mod = {
                    'type': 'value_space',
                    'target': cand.get('config_key', cand['name']),
                    'param_name': cand['name'],
                    'search_space': self._generate_search_space(cand),
                    'line': cand.get('line', 0),
                    'is_config': 'config_key' in cand,
                    # v1.4.0: 添加 backbone 类名限制
                    'class_name': self.backbone_class_name if cand.get('is_backbone_param') else None
                }
            elif cand['type'] == 'layer':
                mod = {
                    'type': 'layer_space',
                    'target': cand.get('config_key', cand['name']),
                    'param_name': cand['name'],
                    'layer_options': self._generate_layer_options(cand),
                    'line': cand.get('line', 0),
                    'is_config': 'config_key' in cand,
                    # v1.4.0: 添加 backbone 类名限制
                    'class_name': self.backbone_class_name if cand.get('is_backbone_param') else None
                }
            else:
                continue
            
            modifications_by_file[file_path].append(mod)
        
        # 展示修改计划
        all_mods = []
        for file_path, mods in modifications_by_file.items():
            for mod in mods:
                mod['file'] = file_path
                all_mods.append(mod)
        
        self.ui_agent.show_modification_plan(all_mods)
        
        return modifications_by_file
    
    def _resolve_config_file_path(self, cand: Dict) -> str:
        """解析配置文件路径"""
        source_file = cand.get('source_file', '')
        return str(self.project_path / source_file)
    
    def _generate_search_space(self, cand: Dict) -> List[Any]:
        """生成搜索空间"""
        current = cand.get('current_value', 0)
        param_name = cand.get('name', '')
        
        # v1.4.0: 根据参数类型生成合理的搜索空间
        if isinstance(current, (int, float)):
            # 模型维度参数
            if any(kw in param_name.lower() for kw in ['d_model', 'hidden_dim', 'embed_dim', 'dim_feedforward', 'ffn_dim']):
                if current <= 64:
                    return [32, 64, 128]
                elif current <= 256:
                    return [128, 256, 512]
                else:
                    return [256, 512, 1024]
            
            # 层数参数
            if any(kw in param_name.lower() for kw in ['num_layers', 'n_layers', 'depth', 'num_blocks']):
                if current <= 2:
                    return [1, 2, 3]
                elif current <= 6:
                    return [2, 4, 6, 8]
                else:
                    return [4, 6, 8, 12]
            
            # 注意力头数
            if any(kw in param_name.lower() for kw in ['num_heads', 'n_heads', 'nhead']):
                return [4, 8, 16]
            
            # Dropout 率
            if 'dropout' in param_name.lower():
                return [0.1, 0.2, 0.3, 0.5]
            
            # 默认
            if current < 1:
                return [current / 2, current, current * 2]
            return [max(1, int(current / 2)), current, current * 2]
        
        return [current]
    
    def _generate_layer_options(self, cand: Dict) -> List[str]:
        """生成层选项"""
        name = cand['name'].lower()
        current = cand.get('current_value', '')
        
        if 'activation' in name:
            return ['relu', 'gelu', 'tanh', 'sigmoid']
        elif 'optimizer' in name:
            return ['Adam', 'SGD', 'RMSprop']
        elif 'norm' in name:
            return ['layernorm', 'batchnorm']
        
        return [str(current)]
    
    def create_backup(self):
        """
        v1.4.0: 使用 BackupManager 创建备份
        """
        print("\n[Phase 8] Creating Backups...")
        
        operation = self.backup_manager.create_backup(
            description=f"NAS injection backup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return operation.id
    
    def apply_modifications(self, modifications_by_file: Dict[str, List[Dict]]):
        """
        v1.4.0: 应用修改到文件
        支持 YAML 配置文件修改
        """
        print("\n[Phase 9] Applying Modifications...")
        
        success_count = 0
        total_files = len(modifications_by_file)
        
        for i, (file_path, mods) in enumerate(modifications_by_file.items(), 1):
            self.ui_agent.show_progress(i, total_files, f"Processing {Path(file_path).name}")
            
            # 分离不同类型的修改
            regular_mods = [m for m in mods if not m.get('is_config') and not m.get('is_yaml')]
            config_mods = [m for m in mods if m.get('is_config')]
            yaml_mods = [m for m in mods if m.get('is_yaml')]
            
            success = True
            
            # 应用普通修改
            if regular_mods:
                print(f"  Applying regular modifications to {file_path}")
                if not self.modifier_agent.apply_modifications(file_path, regular_mods):
                    success = False
            
            # 应用 Python 配置修改
            for mod in config_mods:
                print(f"  Applying config modification: {mod['target']}")
                if not self._apply_config_modification(file_path, mod):
                    success = False
            
            # 应用 YAML 配置修改
            for mod in yaml_mods:
                print(f"  Applying YAML modification: {mod['target']}")
                if not self._apply_yaml_modification(file_path, mod):
                    success = False
            
            if success:
                success_count += 1
                print(f"  ✓ Modified: {file_path}")
            else:
                print(f"  ✗ Failed: {file_path}")
        
        print(f"\nSuccessfully modified {success_count}/{total_files} files")
    
    def _apply_config_modification(self, file_path: str, mod: Dict) -> bool:
        """应用 Python 配置文件的修改"""
        try:
            path = Path(file_path)
            key_path = mod['target'].split('.')
            
            # 生成新值
            if mod['type'] == 'value_space':
                new_value = f"ValueSpace({mod['search_space']})"
            else:
                new_value = f"LayerSpace({mod['layer_options']})"
            
            # 修改配置文件
            return ConfigFileHandler.modify_config_value(path, key_path, new_value)
            
        except Exception as e:
            print(f"  Error applying config modification: {e}")
            return False
    
    def _apply_yaml_modification(self, file_path: str, mod: Dict) -> bool:
        """
        v1.4.0: 应用 YAML 配置文件的修改
        """
        try:
            import yaml
            path = Path(file_path)
            
            # 读取 YAML
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 导航到目标位置
            current = data
            key_path = mod['target']
            
            if isinstance(key_path, str):
                key_path = key_path.split('.')
            
            for key in key_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # 生成新值
            if mod['type'] == 'yaml_value_space':
                current[key_path[-1]] = f"ValueSpace({mod['search_space']})"
            else:
                current[key_path[-1]] = f"LayerSpace({mod['layer_options']})"
            
            # 写回 YAML
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            print(f"  ✓ Modified YAML: {file_path}")
            return True
            
        except Exception as e:
            print(f"  Error applying YAML modification: {e}")
            return False
    
    def run_search_space_expansion(self):
        """
        v1.4.0: 运行寻优空间张开
        """
        print("\n[Phase 10] Search Space Expansion...")
        
        expanded_files = self.search_space_expander.expand_project(str(self.project_path))
        
        if expanded_files:
            print(f"Expanded {len(expanded_files)} files:")
            for f in expanded_files:
                print(f"  - {f}")
        else:
            print("No files needed expansion")
    
    def run_report_injection(self):
        """
        v1.4.0: 运行 Report 注入
        """
        print("\n[Phase 11] Report Injection...")
        
        if not self.entry_file:
            print("  No entry file specified, skipping report injection")
            return
        
        modified_files = inject_report_to_project(
            str(self.project_path),
            self.entry_file,
            self.llm_client
        )
        
        if modified_files:
            print(f"Injected report to {len(modified_files)} files:")
            for f in modified_files:
                print(f"  - {f}")
        else:
            print("No files needed report injection")
    
    def run(self):
        """运行完整的 NAS 注入流程 (v1.4.0)"""
        print("\n" + "="*70)
        print("🚀 NAS Agent System v1.4.0 - Starting Injection Process")
        print("="*70)
        
        # Phase 1: 扫描项目
        files = self.scan_project()
        if not files:
            print("No Python files found!")
            return
        
        # Phase 2: 创建 Scope Agents
        self.create_scope_agents(files)
        
        # Phase 3: 智能模型发现
        self.run_model_discovery()
        
        # Phase 4: P2P 解析
        self.run_p2p_resolution()
        
        # Phase 5: 收集候选
        candidates = self.collect_nas_candidates()
        if not candidates:
            print("No NAS candidates found!")
            return
        
        # Phase 6: 用户交互
        selected = self.run_user_interaction(candidates)
        if not selected:
            print("No candidates selected for injection")
            return
        
        # Phase 7: 生成修改计划
        modifications = self.generate_modifications(selected)
        
        # Phase 8: 创建备份
        backup_id = self.create_backup()
        
        # Phase 9: 应用修改
        self.apply_modifications(modifications)
        
        # Phase 10: 寻优空间张开
        self.run_search_space_expansion()
        
        # Phase 11: Report 注入
        self.run_report_injection()
        
        # 完成
        print("\n" + "="*70)
        print("✅ NAS Injection Complete!")
        print("="*70)
        
        # 打印注册表摘要
        self.registry.print_summary()
    
    def _get_agent_by_file(self, file_path: str) -> Optional[ScopeAgent]:
        """根据文件路径获取 Agent"""
        for agent in self.scope_agents.values():
            if agent.file_path == file_path:
                return agent
        return None
    
    def undo_last_operation(self) -> bool:
        """
        v1.4.0: 撤销上次操作
        """
        print("\n[Undo] Reverting last NAS injection...")
        return self.backup_manager.undo()


# 导入 datetime 用于备份描述
from datetime import datetime
