"""
MAS Core - Smart Model Discovery (v1.2.0)
智能模型识别模块：
- 解析动态反射（getattr）
- 识别实际被实例化的模型
- 处理跨文件的模型引用
"""
import ast
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path


class ModelDiscoveryAnalyzer:
    """
    智能模型发现分析器
    分析代码以识别实际被实例化的模型
    """
    
    def __init__(self, project_path: str, llm_client=None):
        self.project_path = Path(project_path)
        self.llm_client = llm_client
        self.discovered_models: Dict[str, Dict] = {}
        self.instantiated_model: Optional[Dict] = None
        self.model_files: List[Path] = []
    
    def discover_model_files(self) -> List[Path]:
        """
        发现项目中所有可能包含模型定义的 Python 文件
        """
        model_files = []
        
        # 常见模型目录
        model_dirs = ['models', 'model', 'networks', 'nets', 'architectures']
        
        for py_file in self.project_path.rglob("*.py"):
            # 排除常见目录
            if any(part.startswith('.') or part in ['__pycache__', 'venv', 'env'] 
                   for part in py_file.parts):
                continue
            
            # 检查是否在模型目录中
            if any(model_dir in py_file.parts for model_dir in model_dirs):
                model_files.append(py_file)
                continue
            
            # 检查文件内容是否包含模型定义
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if self._is_model_file(content):
                    model_files.append(py_file)
            except:
                pass
        
        self.model_files = model_files
        return model_files
    
    def _is_model_file(self, content: str) -> bool:
        """检查文件内容是否包含模型定义"""
        # 检查是否包含 nn.Module 子类
        patterns = [
            r'class\s+\w+\s*\(\s*nn\.Module\s*\)',
            r'class\s+\w+\s*\(\s*torch\.nn\.Module\s*\)',
            r'from\s+torch\s+import\s+nn',
            r'import\s+torch\.nn',
        ]
        return any(re.search(p, content) for p in patterns)
    
    def extract_model_classes(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        从文件中提取所有模型类定义
        
        Returns:
            List[Dict]: 模型类信息列表
            [{
                "name": "ModelClass",
                "file": "path/to/file.py",
                "line": 10,
                "base_classes": ["nn.Module"],
                "init_params": ["input_dim", "hidden_dim", ...]
            }]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # 检查是否继承自 nn.Module
                    is_model = False
                    base_classes = []
                    
                    for base in node.bases:
                        base_name = self._get_attr_name(base)
                        base_classes.append(base_name)
                        if 'Module' in base_name or 'nn.' in base_name:
                            is_model = True
                    
                    if is_model:
                        # 提取 __init__ 参数
                        init_params = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                                init_params = self._extract_init_params(item)
                                break
                        
                        classes.append({
                            "name": node.name,
                            "file": str(file_path),
                            "line": node.lineno,
                            "base_classes": base_classes,
                            "init_params": init_params
                        })
            
            return classes
            
        except Exception as e:
            print(f"[ModelDiscovery] Error extracting classes from {file_path}: {e}")
            return []
    
    def _get_attr_name(self, node) -> str:
        """获取属性节点的完整名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attr_name(node.value)}.{node.attr}"
        return ""
    
    def _extract_init_params(self, init_node: ast.FunctionDef) -> List[str]:
        """提取 __init__ 方法的参数名"""
        params = []
        if init_node.args.args:
            # 跳过 self
            for arg in init_node.args.args[1:]:
                params.append(arg.arg)
        return params
    
    def analyze_entry_file(self, entry_file: Path) -> Dict[str, Any]:
        """
        分析入口文件，识别模型实例化逻辑
        
        Returns:
            Dict: {
                "has_dynamic_instantiation": bool,
                "getattr_calls": [...],
                "model_instantiation": {...},
                "imported_modules": [...]
            }
        """
        try:
            with open(entry_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            result = {
                "has_dynamic_instantiation": False,
                "getattr_calls": [],
                "model_instantiation": None,
                "imported_modules": [],
                "direct_model_calls": []
            }
            
            # 查找导入的模块
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result["imported_modules"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        result["imported_modules"].append(f"{module}.{alias.name}" if module else alias.name)
            
            # 查找 getattr 调用
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = self._get_attr_name(node.func)
                    if 'getattr' in func_name:
                        result["has_dynamic_instantiation"] = True
                        result["getattr_calls"].append({
                            "line": node.lineno,
                            "args": [self._get_node_repr(arg) for arg in node.args]
                        })
            
            # 查找模型实例化
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = self._get_attr_name(node.func)
                    # 检查是否是模型类实例化
                    if func_name and func_name[0].isupper():
                        result["direct_model_calls"].append({
                            "line": node.lineno,
                            "class_name": func_name,
                            "args": [self._get_node_repr(arg) for arg in node.args],
                            "keywords": {kw.arg: self._get_node_repr(kw.value) for kw in node.keywords}
                        })
            
            return result
            
        except Exception as e:
            print(f"[ModelDiscovery] Error analyzing entry file: {e}")
            return {}
    
    def _get_node_repr(self, node) -> str:
        """获取 AST 节点的字符串表示"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Str):  # Python < 3.8
            return repr(node.s)
        elif isinstance(node, ast.Num):  # Python < 3.8
            return repr(node.n)
        elif isinstance(node, ast.List):
            return "[...]"
        elif isinstance(node, ast.Dict):
            return "{...}"
        return "..."
    
    def identify_instantiated_model(self, entry_file: Path, 
                                     available_models: List[str]) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 识别实际被实例化的模型
        
        Args:
            entry_file: 入口文件路径
            available_models: 可用的模型类名列表
            
        Returns:
            Dict: 实例化模型信息
        """
        if not self.llm_client:
            return None
        
        try:
            with open(entry_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取与模型实例化相关的代码片段
            relevant_lines = self._extract_model_instantiation_snippet(content)
            
            # 使用 LLM 分析
            result = self.llm_client.analyze_code_snippet_for_model_instantiation(
                relevant_lines,
                available_models
            )
            
            if result and result.get("instantiated_model"):
                self.instantiated_model = result
                return result
            
            return None
            
        except Exception as e:
            print(f"[ModelDiscovery] Error identifying instantiated model: {e}")
            return None
    
    def _extract_model_instantiation_snippet(self, content: str) -> str:
        """提取模型实例化相关的代码片段"""
        lines = content.split('\n')
        relevant_lines = []
        
        for i, line in enumerate(lines, 1):
            # 查找包含模型相关关键字的行
            if any(keyword in line.lower() for keyword in [
                'model', 'getattr', 'instantiate', 'class', 'nn.module'
            ]):
                # 包含前后文
                start = max(0, i - 3)
                end = min(len(lines), i + 3)
                relevant_lines.extend(lines[start:end])
        
        # 去重并保持顺序
        seen = set()
        unique_lines = []
        for line in relevant_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return '\n'.join(unique_lines[:50])  # 限制长度
    
    def get_all_available_models(self) -> List[Dict[str, Any]]:
        """
        获取项目中所有可用的模型
        
        Returns:
            List[Dict]: 所有模型类信息
        """
        all_models = []
        
        # 发现模型文件
        if not self.model_files:
            self.discover_model_files()
        
        # 从每个文件提取模型类
        for model_file in self.model_files:
            models = self.extract_model_classes(model_file)
            all_models.extend(models)
        
        self.discovered_models = {m["name"]: m for m in all_models}
        return all_models
    
    def run_full_discovery(self, entry_file: Path) -> Dict[str, Any]:
        """
        运行完整的模型发现流程
        
        Args:
            entry_file: 入口文件路径
            
        Returns:
            Dict: 完整的发现结果
        """
        print("[ModelDiscovery] Starting full model discovery...")
        
        # 1. 发现所有模型文件
        model_files = self.discover_model_files()
        print(f"[ModelDiscovery] Found {len(model_files)} model files")
        
        # 2. 提取所有可用模型
        all_models = self.get_all_available_models()
        print(f"[ModelDiscovery] Found {len(all_models)} model classes:")
        for m in all_models:
            print(f"  - {m['name']} (in {m['file']})")
        
        # 3. 分析入口文件
        entry_analysis = self.analyze_entry_file(entry_file)
        print(f"[ModelDiscovery] Entry file analysis:")
        print(f"  - Has dynamic instantiation: {entry_analysis.get('has_dynamic_instantiation')}")
        print(f"  - getattr calls: {len(entry_analysis.get('getattr_calls', []))}")
        
        # 4. 识别实际被实例化的模型
        model_names = [m["name"] for m in all_models]
        instantiated = self.identify_instantiated_model(entry_file, model_names)
        
        if instantiated:
            print(f"[ModelDiscovery] Identified instantiated model:")
            print(f"  - Model: {instantiated.get('instantiated_model')}")
            print(f"  - Variable: {instantiated.get('model_variable')}")
            print(f"  - Confidence: {instantiated.get('confidence')}")
        
        return {
            "all_models": all_models,
            "entry_analysis": entry_analysis,
            "instantiated_model": instantiated,
            "model_files": [str(f) for f in model_files]
        }


def discover_models_in_project(project_path: str, entry_file: str, 
                                llm_client=None) -> Dict[str, Any]:
    """
    便捷函数：发现项目中的模型
    
    Args:
        project_path: 项目路径
        entry_file: 入口文件路径（相对于项目路径）
        llm_client: LLM 客户端
        
    Returns:
        Dict: 模型发现结果
    """
    analyzer = ModelDiscoveryAnalyzer(project_path, llm_client)
    entry_path = Path(project_path) / entry_file
    return analyzer.run_full_discovery(entry_path)
