"""
MAS Core - Cross-File Parameter Modifier (v1.2.0)
跨文件参数修改器：
- 处理分布在不同文件夹的参数
- 正确识别参数位置并修改
- 支持配置文件（Python dict、YAML、JSON）
"""
import ast
import re
import json
import yaml
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import libcst as cst


class CrossFileParameterModifier:
    """
    跨文件参数修改器
    处理分布在多个文件中的参数
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.modifications: List[Dict[str, Any]] = []
        self.backup_files: List[Path] = []
    
    def find_parameter_locations(self, param_name: str, 
                                  param_value: Any = None) -> List[Dict[str, Any]]:
        """
        查找参数在项目中的所有位置
        
        Args:
            param_name: 参数名
            param_value: 参数值（可选，用于确认）
            
        Returns:
            List[Dict]: 参数位置列表
            [{
                "file": "path/to/file.py",
                "line": 10,
                "type": "dict_key/variable/keyword_arg",
                "context": "...",
                "current_value": "..."
            }]
        """
        locations = []
        
        for py_file in self.project_path.rglob("*.py"):
            # 排除常见目录
            if any(part.startswith('.') or part in ['__pycache__', 'venv', 'env'] 
                   for part in py_file.parts):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找参数定义
                file_locations = self._find_in_python_file(
                    py_file, content, param_name, param_value
                )
                locations.extend(file_locations)
                
            except Exception as e:
                continue
        
        # 查找 YAML 配置文件
        for yaml_file in self.project_path.rglob("*.yaml"):
            if any(part.startswith('.') for part in yaml_file.parts):
                continue
            try:
                yaml_locations = self._find_in_yaml_file(yaml_file, param_name)
                locations.extend(yaml_locations)
            except:
                pass
        
        # 查找 JSON 配置文件
        for json_file in self.project_path.rglob("*.json"):
            if any(part.startswith('.') for part in json_file.parts):
                continue
            try:
                json_locations = self._find_in_json_file(json_file, param_name)
                locations.extend(json_locations)
            except:
                pass
        
        return locations
    
    def _find_in_python_file(self, file_path: Path, content: str, 
                              param_name: str, param_value: Any) -> List[Dict]:
        """在 Python 文件中查找参数"""
        locations = []
        lines = content.split('\n')
        
        # 模式1: 字典键值对
        # "param_name": value 或 'param_name': value
        dict_pattern = rf'["\']?{re.escape(param_name)}["\']?\s*:\s*([^,\n]+)'
        for i, line in enumerate(lines, 1):
            match = re.search(dict_pattern, line)
            if match:
                locations.append({
                    "file": str(file_path),
                    "line": i,
                    "type": "dict_key",
                    "context": line.strip(),
                    "current_value": match.group(1).strip()
                })
        
        # 模式2: 变量赋值
        # param_name = value
        var_pattern = rf'^{re.escape(param_name)}\s*=\s*(.+)'
        for i, line in enumerate(lines, 1):
            match = re.search(var_pattern, line.strip())
            if match:
                locations.append({
                    "file": str(file_path),
                    "line": i,
                    "type": "variable",
                    "context": line.strip(),
                    "current_value": match.group(1).strip()
                })
        
        # 模式3: 函数关键字参数
        # func(param_name=value)
        kwarg_pattern = rf'{re.escape(param_name)}\s*=\s*([^,)]+)'
        for i, line in enumerate(lines, 1):
            match = re.search(kwarg_pattern, line)
            if match:
                locations.append({
                    "file": str(file_path),
                    "line": i,
                    "type": "keyword_arg",
                    "context": line.strip(),
                    "current_value": match.group(1).strip()
                })
        
        return locations
    
    def _find_in_yaml_file(self, file_path: Path, param_name: str) -> List[Dict]:
        """在 YAML 文件中查找参数"""
        locations = []
        
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # 简单查找参数名
        pattern = rf'^{re.escape(param_name)}\s*:'
        for i, line in enumerate(lines, 1):
            match = re.search(pattern, line)
            if match:
                locations.append({
                    "file": str(file_path),
                    "line": i,
                    "type": "yaml_key",
                    "context": line.strip(),
                    "current_value": line.split(':', 1)[1].strip() if ':' in line else ""
                })
        
        return locations
    
    def _find_in_json_file(self, file_path: Path, param_name: str) -> List[Dict]:
        """在 JSON 文件中查找参数"""
        locations = []
        
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # 查找 JSON 键
        pattern = rf'"{re.escape(param_name)}"\s*:'
        for i, line in enumerate(lines, 1):
            match = re.search(pattern, line)
            if match:
                locations.append({
                    "file": str(file_path),
                    "line": i,
                    "type": "json_key",
                    "context": line.strip(),
                    "current_value": ""
                })
        
        return locations
    
    def modify_parameter(self, location: Dict[str, Any], 
                         new_value: Union[str, List]) -> bool:
        """
        修改指定位置的参数
        
        Args:
            location: 参数位置信息
            new_value: 新值（ValueSpace 或 LayerSpace 表达式）
            
        Returns:
            bool: 是否成功
        """
        file_path = Path(location["file"])
        line_num = location["line"]
        param_type = location["type"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 获取要修改的行
            line_idx = line_num - 1
            if line_idx >= len(lines):
                return False
            
            original_line = lines[line_idx]
            
            # 根据类型修改
            if param_type == "dict_key":
                # 修改字典键值对
                new_line = re.sub(
                    rf'(["\']?{re.escape(location.get("param_name", ""))}["\']?\s*:\s*)[^,\n]+',
                    rf'\g<1>{new_value}',
                    original_line
                )
            elif param_type == "variable":
                # 修改变量赋值
                param_name = location.get("param_name", "")
                new_line = re.sub(
                    rf'^({re.escape(param_name)}\s*=\s*).+$',
                    rf'\g<1>{new_value}',
                    original_line
                )
            elif param_type == "keyword_arg":
                # 修改关键字参数
                param_name = location.get("param_name", "")
                new_line = re.sub(
                    rf'({re.escape(param_name)}\s*=\s*)[^,)]+',
                    rf'\g<1>{new_value}',
                    original_line
                )
            else:
                return False
            
            # 应用修改
            lines[line_idx] = new_line
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return True
            
        except Exception as e:
            print(f"[CrossFileModifier] Error modifying {file_path}:{line_num}: {e}")
            return False
    
    def modify_python_dict_in_file(self, file_path: Path, 
                                    dict_name: str,
                                    key: str, 
                                    new_value: str) -> bool:
        """
        修改 Python 文件中的字典值
        
        Args:
            file_path: 文件路径
            dict_name: 字典变量名
            key: 要修改的键
            new_value: 新值
            
        Returns:
            bool: 是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式查找并修改字典键值对
            # 匹配 dict_name = { ... key: value ... }
            pattern = rf'({re.escape(dict_name)}\s*=\s*\{{[\s\S]*?["\']?{re.escape(key)}["\']?\s*:\s*)[^,\n\}}]+'
            
            def replace_value(match):
                return f"{match.group(1)}{new_value}"
            
            new_content = re.sub(pattern, replace_value, content)
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            
            return False
            
        except Exception as e:
            print(f"[CrossFileModifier] Error modifying dict in {file_path}: {e}")
            return False
    
    def add_import_to_file(self, file_path: Path, 
                           import_statement: str) -> bool:
        """
        向文件添加 import 语句
        
        Args:
            file_path: 文件路径
            import_statement: import 语句（如 "from archmind import ValueSpace"）
            
        Returns:
            bool: 是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已存在
            if import_statement in content:
                return True
            
            # 在文件开头添加 import
            lines = content.split('\n')
            
            # 找到最后一个 import 语句的位置
            last_import_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    last_import_idx = i
            
            # 在最后一个 import 后添加
            if last_import_idx >= 0:
                lines.insert(last_import_idx + 1, import_statement)
            else:
                # 在文件开头添加（跳过 docstring）
                insert_idx = 0
                if lines and lines[0].strip().startswith('"""'):
                    for i, line in enumerate(lines[1:], 1):
                        if '"""' in line:
                            insert_idx = i + 1
                            break
                lines.insert(insert_idx, import_statement)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            return True
            
        except Exception as e:
            print(f"[CrossFileModifier] Error adding import to {file_path}: {e}")
            return False


class ConfigFileHandler:
    """
    配置文件处理器
    处理 YAML、JSON、Python 配置文件
    """
    
    @staticmethod
    def load_config(file_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif suffix in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        elif suffix == '.py':
            return ConfigFileHandler._load_python_config(file_path)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
    
    @staticmethod
    def _load_python_config(file_path: Path) -> Dict[str, Any]:
        """加载 Python 配置文件（执行并提取字典）"""
        config = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 安全地执行 Python 代码
        try:
            exec(content, config)
        except Exception as e:
            print(f"[ConfigHandler] Error executing {file_path}: {e}")
        
        # 过滤掉内置变量
        return {k: v for k, v in config.items() if not k.startswith('_')}
    
    @staticmethod
    def modify_config_value(file_path: Path, 
                            key_path: List[str], 
                            new_value: Any) -> bool:
        """
        修改配置文件中的值
        
        Args:
            file_path: 配置文件路径
            key_path: 键路径（如 ['training', 'lr']）
            new_value: 新值
            
        Returns:
            bool: 是否成功
        """
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.json':
                return ConfigFileHandler._modify_json(file_path, key_path, new_value)
            elif suffix in ['.yaml', '.yml']:
                return ConfigFileHandler._modify_yaml(file_path, key_path, new_value)
            elif suffix == '.py':
                return ConfigFileHandler._modify_python_config(file_path, key_path, new_value)
            else:
                return False
        except Exception as e:
            print(f"[ConfigHandler] Error modifying {file_path}: {e}")
            return False
    
    @staticmethod
    def _modify_json(file_path: Path, key_path: List[str], new_value: Any) -> bool:
        """修改 JSON 文件"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 导航到目标位置
        current = data
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[key_path[-1]] = new_value
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    @staticmethod
    def _modify_yaml(file_path: Path, key_path: List[str], new_value: Any) -> bool:
        """修改 YAML 文件"""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 导航到目标位置
        current = data
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[key_path[-1]] = new_value
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        return True
    
    @staticmethod
    def _modify_python_config(file_path: Path, key_path: List[str], new_value: Any) -> bool:
        """修改 Python 配置文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 构建要查找的键路径
        dict_name = key_path[0]
        inner_keys = key_path[1:]
        
        if len(inner_keys) == 0:
            # 直接修改变量
            pattern = rf'^({re.escape(dict_name)}\s*=\s*).+$'
            new_content = re.sub(pattern, rf'\g<1>{new_value}', content, flags=re.MULTILINE)
        else:
            # 修改嵌套字典
            # 这是一个简化的实现，可能需要更复杂的解析
            key = inner_keys[0]
            pattern = rf'(["\']?{re.escape(key)}["\']?\s*:\s*)[^,\n\}}]+'
            new_content = re.sub(pattern, rf'\g<1>{new_value}', content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
