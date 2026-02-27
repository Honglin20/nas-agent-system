"""
MAS Core - Search Space Expander (v1.2.0)
寻优空间张开器：
- 处理条件代码如：if activation == 'relu': self.act = nn.ReLU()
- 修改模型代码，确保寻优空间张开后所有选项能正确实例化
- 将条件分支转换为 LayerSpace
"""
import ast
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import libcst as cst


class ConditionalLayerAnalyzer:
    """
    条件层分析器
    识别代码中的条件层选择模式
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def analyze_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        分析文件中的条件层选择
        
        Returns:
            List[Dict]: 条件层信息列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用 AST 分析
            tree = ast.parse(content)
            conditional_layers = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    layer_info = self._analyze_if_statement(node, content)
                    if layer_info:
                        conditional_layers.append(layer_info)
            
            # 使用 LLM 进行更深入的分析
            if self.llm_client and conditional_layers:
                snippet = self._extract_relevant_snippet(content, conditional_layers)
                llm_result = self.llm_client.analyze_conditional_layers(snippet)
                
                # 合并 LLM 结果
                for i, layer_info in enumerate(conditional_layers):
                    if i < len(llm_result):
                        layer_info.update(llm_result[i])
            
            return conditional_layers
            
        except Exception as e:
            print(f"[SearchSpaceExpander] Error analyzing {file_path}: {e}")
            return []
    
    def _analyze_if_statement(self, node: ast.If, content: str) -> Optional[Dict[str, Any]]:
        """分析 if 语句是否是条件层选择"""
        # 检查条件是否是比较字符串
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.ops[0], ast.Eq):
                # 检查是否是 activation == 'xxx' 模式
                if isinstance(node.test.left, ast.Name):
                    var_name = node.test.left.id
                    if any(keyword in var_name.lower() for keyword in 
                           ['activation', 'act', 'norm', 'normalization', 'optimizer', 'layer']):
                        
                        # 提取赋值的变量
                        assigned_var = None
                        assigned_value = None
                        
                        for stmt in node.body:
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute):
                                        assigned_var = f"{self._get_attr_name(target.value)}.{target.attr}"
                                    elif isinstance(target, ast.Name):
                                        assigned_var = target.id
                                    
                                    if isinstance(stmt.value, ast.Call):
                                        assigned_value = self._get_call_repr(stmt.value)
                                    elif isinstance(stmt.value, ast.Name):
                                        assigned_value = stmt.value.id
                        
                        if assigned_var:
                            return {
                                "variable_name": assigned_var,
                                "condition_variable": var_name,
                                "line_start": node.lineno,
                                "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                                "assigned_value": assigned_value
                            }
        
        return None
    
    def _get_attr_name(self, node) -> str:
        """获取属性节点的完整名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attr_name(node.value)}.{node.attr}"
        return ""
    
    def _get_call_repr(self, node: ast.Call) -> str:
        """获取调用节点的字符串表示"""
        if isinstance(node.func, ast.Name):
            return f"{node.func.id}()"
        elif isinstance(node.func, ast.Attribute):
            return f"{self._get_attr_name(node.func)}()"
        return ""
    
    def _extract_relevant_snippet(self, content: str, 
                                   conditional_layers: List[Dict]) -> str:
        """提取相关代码片段"""
        lines = content.split('\n')
        
        # 找到所有相关的行范围
        all_lines = set()
        for layer in conditional_layers:
            start = layer.get("line_start", 1) - 1
            end = layer.get("line_end", start + 1)
            all_lines.update(range(start, end + 3))  # 包含一些上下文
        
        # 提取并排序
        sorted_lines = sorted(all_lines)
        if not sorted_lines:
            return ""
        
        # 获取连续的行块
        snippets = []
        current_start = sorted_lines[0]
        current_end = sorted_lines[0]
        
        for line_num in sorted_lines[1:]:
            if line_num == current_end + 1:
                current_end = line_num
            else:
                snippets.extend(lines[current_start:current_end + 1])
                snippets.append("\n# ...\n")
                current_start = line_num
                current_end = line_num
        
        snippets.extend(lines[current_start:current_end + 1])
        
        return '\n'.join(snippets[:100])  # 限制长度


class SearchSpaceExpander:
    """
    寻优空间张开器
    将条件层选择转换为 LayerSpace
    """
    
    def __init__(self, llm_client=None):
        self.analyzer = ConditionalLayerAnalyzer(llm_client)
        self.llm_client = llm_client
    
    def expand_file(self, file_path: Path) -> bool:
        """
        对文件进行寻优空间张开
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否成功
        """
        print(f"[SearchSpaceExpander] Expanding {file_path}...")
        
        # 1. 分析条件层
        conditional_layers = self.analyzer.analyze_file(file_path)
        
        if not conditional_layers:
            print(f"[SearchSpaceExpander] No conditional layers found in {file_path}")
            return False
        
        print(f"[SearchSpaceExpander] Found {len(conditional_layers)} conditional layers")
        
        # 2. 对每个条件层进行转换
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified_content = content
            
            for layer_info in conditional_layers:
                modified_content = self._convert_to_layer_space(
                    modified_content, layer_info
                )
            
            # 3. 添加必要的 import
            if modified_content != content:
                modified_content = self._add_import(modified_content)
                
                # 4. 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                print(f"[SearchSpaceExpander] Successfully expanded {file_path}")
                return True
            
            return False
            
        except Exception as e:
            print(f"[SearchSpaceExpander] Error expanding {file_path}: {e}")
            return False
    
    def _convert_to_layer_space(self, content: str, 
                                 layer_info: Dict[str, Any]) -> str:
        """
        将条件层选择转换为 LayerSpace
        
        Args:
            content: 文件内容
            layer_info: 条件层信息
            
        Returns:
            str: 修改后的内容
        """
        lines = content.split('\n')
        
        line_start = layer_info.get("line_start", 1) - 1
        line_end = layer_info.get("line_end", line_start + 1)
        variable_name = layer_info.get("variable_name", "")
        options = layer_info.get("options", [])
        
        if not options:
            # 使用默认选项
            condition_var = layer_info.get("condition_variable", "")
            if "activation" in condition_var.lower():
                options = ["nn.ReLU()", "nn.Sigmoid()", "nn.Tanh()", "nn.GELU()"]
            elif "norm" in condition_var.lower():
                options = ["nn.BatchNorm1d(dim)", "nn.LayerNorm(dim)"]
            else:
                return content
        
        # 构建 LayerSpace 赋值
        formatted_options = [f'"{opt}"' for opt in options]
        layer_space_line = f"{variable_name} = LayerSpace([{', '.join(formatted_options)}])"
        
        # 替换条件代码块
        new_lines = lines[:line_start]
        new_lines.append(layer_space_line)
        new_lines.extend(lines[line_end:])
        
        return '\n'.join(new_lines)
    
    def _add_import(self, content: str) -> str:
        """添加 LayerSpace import"""
        import_stmt = "from archmind import LayerSpace"
        
        if import_stmt in content:
            return content
        
        lines = content.split('\n')
        
        # 找到最后一个 import 语句的位置
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                last_import_idx = i
        
        # 在最后一个 import 后添加
        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, import_stmt)
        else:
            # 在文件开头添加（跳过 docstring）
            insert_idx = 0
            if lines and lines[0].strip().startswith('"""'):
                for i, line in enumerate(lines[1:], 1):
                    if '"""' in line:
                        insert_idx = i + 1
                        break
            lines.insert(insert_idx, import_stmt)
        
        return '\n'.join(lines)
    
    def expand_project(self, project_path: str) -> List[str]:
        """
        对整个项目进行寻优空间张开
        
        Args:
            project_path: 项目路径
            
        Returns:
            List[str]: 修改的文件列表
        """
        project = Path(project_path)
        modified_files = []
        
        for py_file in project.rglob("*.py"):
            if any(part.startswith('.') or part in ['__pycache__', 'venv', 'env'] 
                   for part in py_file.parts):
                continue
            
            # 快速检查是否包含条件层选择
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否包含激活函数或归一化层的条件选择
                if not re.search(r'if\s+\w+\s*==\s*[\'"]\w+[\'"]', content):
                    continue
                
                if not any(keyword in content for keyword in 
                          ['activation', 'act', 'norm', 'ReLU', 'Sigmoid', 'BatchNorm']):
                    continue
                
                if self.expand_file(py_file):
                    modified_files.append(str(py_file))
                    
            except Exception as e:
                continue
        
        return modified_files


class LayerSpaceTransformer(cst.CSTTransformer):
    """
    使用 libcst 的 LayerSpace 转换器
    更精准的 AST 级别转换
    """
    
    def __init__(self, variable_name: str, options: List[str]):
        self.variable_name = variable_name
        self.options = options
        self.modified = False
    
    def leave_If(self, original_node: cst.If, 
                 updated_node: cst.If) -> cst.FlattenSentinel:
        """将 if 语句转换为 LayerSpace 赋值"""
        # 检查是否是目标条件
        if self._is_target_condition(original_node):
            # 构建 LayerSpace 赋值
            formatted_options = [cst.SimpleString(f'"{opt}"') for opt in self.options]
            layer_space_call = cst.Assign(
                targets=[cst.AssignTarget(target=cst.Name(self.variable_name))],
                value=cst.Call(
                    func=cst.Name("LayerSpace"),
                    args=[
                        cst.Arg(
                            value=cst.List(
                                elements=[cst.Element(value=opt) for opt in formatted_options]
                            )
                        )
                    ]
                )
            )
            
            self.modified = True
            return cst.FlattenSentinel([layer_space_call])
        
        return updated_node
    
    def _is_target_condition(self, node: cst.If) -> bool:
        """检查是否是目标条件"""
        # 简化检查：检查条件是否包含比较
        if isinstance(node.test, cst.Comparison):
            return True
        return False


def expand_search_space_in_project(project_path: str, 
                                    llm_client=None) -> List[str]:
    """
    便捷函数：对项目进行寻优空间张开
    
    Args:
        project_path: 项目路径
        llm_client: LLM 客户端
        
    Returns:
        List[str]: 修改的文件列表
    """
    expander = SearchSpaceExpander(llm_client)
    return expander.expand_project(project_path)
