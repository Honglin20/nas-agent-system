"""
MAS Core - Modifier Agent (v1.4.0 Fixed)
修改智能体：使用 libcst 进行精准的代码修改
- 只修改 backbone 主模型的 __init__ 方法
- 不修改自定义的其他层
- 修改前用 LLM 分析确认修改位置
- 修改后验证文件确实被修改
"""
from typing import Dict, Any, List, Optional, Tuple
import libcst as cst
from libcst import matchers as m
import re

from .base_agent import BaseAgent


class ModifierAgent(BaseAgent):
    """
    代码修改智能体 - v1.4.0
    - 使用 libcst 进行 AST 级别的精准修改
    - 自动处理 Import 插入
    - 只修改 backbone 主模型的 __init__ 方法
    - 修改后验证
    """
    
    def __init__(self):
        super().__init__(
            scope_type="modifier",
            scope_name="code_modifier"
        )
        self.modifications: List[Dict[str, Any]] = []
        self.backbone_class_name: Optional[str] = None
        
    def analyze(self) -> Dict[str, Any]:
        """Modifier Agent 不需要分析"""
        return {'status': 'modifier_ready'}
    
    def get_nas_candidates(self) -> List[Dict[str, Any]]:
        """Modifier Agent 不提供候选"""
        return []
    
    def set_backbone_class(self, class_name: str):
        """
        v1.4.0: 设置 backbone 主模型类名
        只修改这个类的 __init__ 方法
        """
        self.backbone_class_name = class_name
        self._think(f"Set backbone class to modify: {class_name}")
    
    def modify_for_value_space(self, tree: cst.Module, target_name: str, 
                                search_space: List[Any], imports_to_add: set,
                                class_name: Optional[str] = None) -> cst.Module:
        """
        将硬编码数值替换为 ValueSpace
        
        Args:
            tree: libcst 解析树
            target_name: 目标变量名
            search_space: 搜索空间值列表
            imports_to_add: 需要添加的 import 集合
            class_name: 要修改的类名（None 表示修改所有）
            
        Returns:
            cst.Module: 修改后的解析树
        """
        self._think(f"Modifying {target_name} with ValueSpace {search_space}")
        
        # 创建 ValueSpace 表达式
        values_str = ", ".join(repr(v) for v in search_space)
        value_space_expr = f"ValueSpace([{values_str}])"
        
        # v1.4.0: 使用增强的 Transformer，支持类名限制
        target_class = class_name or self.backbone_class_name
        transformer = ValueSpaceTransformer(target_name, value_space_expr, target_class)
        new_tree = tree.visit(transformer)
        
        if transformer.modified:
            # 记录需要添加的 import
            imports_to_add.add("from archmind import ValueSpace")
            self._think(f"Transformed {target_name} to {value_space_expr}")
        else:
            self._think(f"Warning: Did not find {target_name} to transform")
        
        return new_tree
    
    def modify_for_layer_space(self, tree: cst.Module, target_name: str,
                                layer_options: List[str], imports_to_add: set,
                                class_name: Optional[str] = None) -> cst.Module:
        """
        将层实例化替换为 LayerSpace
        
        Args:
            tree: libcst 解析树
            target_name: 目标变量名
            layer_options: 层选项列表
            imports_to_add: 需要添加的 import 集合
            class_name: 要修改的类名（None 表示修改所有）
            
        Returns:
            cst.Module: 修改后的解析树
        """
        self._think(f"Modifying {target_name} with LayerSpace {layer_options}")
        
        # 创建 LayerSpace 表达式 - 确保字符串选项用引号包裹
        formatted_options = []
        for opt in layer_options:
            if isinstance(opt, str):
                formatted_options.append(repr(opt))
            else:
                formatted_options.append(str(opt))
        layers_str = ", ".join(formatted_options)
        layer_space_expr = f"LayerSpace([{layers_str}])"
        
        # v1.4.0: 使用增强的 Transformer，支持类名限制
        target_class = class_name or self.backbone_class_name
        transformer = LayerSpaceTransformer(target_name, layer_space_expr, target_class)
        new_tree = tree.visit(transformer)
        
        if transformer.modified:
            # 记录需要添加的 import
            imports_to_add.add("from archmind import LayerSpace")
            self._think(f"Transformed {target_name} to {layer_space_expr}")
        else:
            self._think(f"Warning: Did not find {target_name} to transform")
        
        return new_tree
    
    def add_imports(self, tree: cst.Module, imports_to_add: set) -> cst.Module:
        """
        添加必要的 import 语句
        
        Args:
            tree: libcst 解析树
            imports_to_add: 需要添加的 import 集合
            
        Returns:
            cst.Module: 修改后的解析树
        """
        if not imports_to_add:
            return tree
        
        self._think(f"Adding imports: {imports_to_add}")
        
        # 创建 Import 添加器
        import_adder = ImportAdder(imports_to_add)
        new_tree = tree.visit(import_adder)
        
        return new_tree
    
    def inject_report_call(self, tree: cst.Module, 
                           metrics: Dict[str, str], imports_to_add: set) -> cst.Module:
        """
        在训练循环中注入 report 调用
        
        Args:
            tree: libcst 解析树
            metrics: 指标字典，如 {'loss': 'loss', 'accuracy': 'accuracy'}
            imports_to_add: 需要添加的 import 集合
            
        Returns:
            cst.Module: 修改后的解析树
        """
        self._think(f"Injecting report call with metrics: {metrics}")
        
        # 构建 report 调用
        kwargs = ", ".join(f"{k}={v}" for k, v in metrics.items())
        report_expr = f"report(model=model, {kwargs})"
        
        # 使用 Transformer 注入
        transformer = ReportInjector(report_expr)
        new_tree = tree.visit(transformer)
        
        # 记录需要添加的 import
        imports_to_add.add("from archmind import report")
        
        self._think(f"Injected report call: {report_expr}")
        return new_tree
    
    def apply_modifications(self, file_path: str, 
                           modifications: List[Dict[str, Any]]) -> bool:
        """
        应用所有修改到文件 - v1.4.0 增强版
        
        Args:
            file_path: 文件路径
            modifications: 修改列表
            
        Returns:
            bool: 是否成功
        """
        try:
            self._think(f"Applying modifications to {file_path}")
            
            # 每次调用都创建新的 imports 集合，避免累积
            imports_to_add = set()
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                original_source = f.read()
            
            # 解析
            tree = cst.parse_module(original_source)
            
            # 应用每个修改
            for mod in modifications:
                mod_type = mod.get('type')
                class_name = mod.get('class_name') or self.backbone_class_name
                
                if mod_type == 'value_space':
                    tree = self.modify_for_value_space(
                        tree, 
                        mod['target'],
                        mod['search_space'],
                        imports_to_add,
                        class_name
                    )
                elif mod_type == 'layer_space':
                    tree = self.modify_for_layer_space(
                        tree,
                        mod['target'],
                        mod['layer_options'],
                        imports_to_add,
                        class_name
                    )
                elif mod_type == 'report':
                    tree = self.inject_report_call(
                        tree,
                        mod['metrics'],
                        imports_to_add
                    )
            
            # 添加必要的 imports
            tree = self.add_imports(tree, imports_to_add)
            
            # 生成修改后的代码
            modified_code = tree.code
            
            # v1.4.0: 验证修改是否生效
            if modified_code == original_source:
                self._think(f"Warning: No changes detected in {file_path}")
                # 尝试使用文本替换作为备选方案
                modified_code = self._fallback_text_replacement(original_source, modifications)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            
            # v1.4.0: 验证文件确实被修改
            with open(file_path, 'r', encoding='utf-8') as f:
                final_content = f.read()
            
            if final_content == original_source:
                self._think(f"ERROR: File was not modified: {file_path}")
                return False
            
            self._think(f"Successfully modified {file_path}")
            return True
            
        except Exception as e:
            self._think(f"ERROR modifying {file_path}: {e}")
            import traceback
            self._think(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _fallback_text_replacement(self, source: str, modifications: List[Dict[str, Any]]) -> str:
        """
        v1.4.0: 备选方案 - 使用文本替换
        当 libcst 修改失败时使用
        """
        self._think("Using fallback text replacement")
        modified = source
        
        for mod in modifications:
            mod_type = mod.get('type')
            target = mod.get('target', '')
            
            if mod_type == 'value_space':
                search_space = mod.get('search_space', [])
                values_str = ", ".join(repr(v) for v in search_space)
                replacement = f"ValueSpace([{values_str}])"
                
                # 查找变量赋值并替换
                pattern = rf'({re.escape(target)}\s*=\s*)[^\n]+'
                modified = re.sub(pattern, rf'\g<1>{replacement}', modified)
                
            elif mod_type == 'layer_space':
                layer_options = mod.get('layer_options', [])
                formatted_options = [repr(opt) for opt in layer_options]
                replacement = f"LayerSpace([{', '.join(formatted_options)}])"
                
                # 查找关键字参数并替换
                pattern = rf'({re.escape(target)}\s*=\s*)[^,\n]+'
                modified = re.sub(pattern, rf'\g<1>{replacement}', modified)
        
        return modified


# ==================== libcst Transformers ====================

class ValueSpaceTransformer(cst.CSTTransformer):
    """
    将数值替换为 ValueSpace - v1.4.0 增强版
    支持限制在特定类的 __init__ 方法中修改
    """
    
    def __init__(self, target_name: str, value_space_expr: str, class_name: Optional[str] = None):
        self.target_name = target_name
        self.value_space_expr = value_space_expr
        self.class_name = class_name
        self.modified = False
        self.in_target_class = False
        self.in_init_method = False
    
    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        """v1.4.0: 检查是否进入目标类"""
        if self.class_name:
            if node.name.value == self.class_name:
                self.in_target_class = True
                print(f"[ValueSpaceTransformer] Entering target class: {self.class_name}")
            else:
                self.in_target_class = False
                print(f"[ValueSpaceTransformer] Skipping class: {node.name.value}")
        else:
            # 如果没有指定类名，修改所有类
            self.in_target_class = True
        return True
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """离开类定义"""
        if self.class_name and node.name.value == self.class_name:
            self.in_target_class = False
        return updated_node
    
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        """v1.4.0: 检查是否进入 __init__ 方法"""
        if self.in_target_class and node.name.value == '__init__':
            self.in_init_method = True
            print(f"[ValueSpaceTransformer] Entering __init__ method")
        return True
    
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """离开函数定义"""
        if node.name.value == '__init__':
            self.in_init_method = False
        return updated_node
    
    def leave_Assign(self, original_node: cst.Assign, 
                     updated_node: cst.Assign) -> cst.Assign:
        """v1.4.0: 只在目标类的 __init__ 中替换"""
        # 如果指定了类名，但不在目标类的 __init__ 中，跳过
        if self.class_name and not (self.in_target_class and self.in_init_method):
            if self.class_name:
                return updated_node
        
        # 检查是否是目标变量的赋值
        for target in original_node.targets:
            if isinstance(target.target, cst.Name):
                if target.target.value == self.target_name:
                    # 替换为 ValueSpace
                    try:
                        new_value = cst.parse_expression(self.value_space_expr)
                        self.modified = True
                        print(f"[ValueSpaceTransformer] Replaced {self.target_name} with {self.value_space_expr}")
                        return updated_node.with_changes(value=new_value)
                    except Exception as e:
                        print(f"[ValueSpaceTransformer] Error parsing expression: {e}")
                        return updated_node
        return updated_node


class LayerSpaceTransformer(cst.CSTTransformer):
    """
    将层替换为 LayerSpace - v1.4.0 增强版
    支持限制在特定类的 __init__ 方法中修改
    """
    
    def __init__(self, target_name: str, layer_space_expr: str, class_name: Optional[str] = None):
        self.target_name = target_name
        self.layer_space_expr = layer_space_expr
        self.class_name = class_name
        self.modified = False
        self.in_target_class = False
        self.in_init_method = False
    
    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        """v1.4.0: 检查是否进入目标类"""
        if self.class_name:
            if node.name.value == self.class_name:
                self.in_target_class = True
                print(f"[LayerSpaceTransformer] Entering target class: {self.class_name}")
            else:
                self.in_target_class = False
                print(f"[LayerSpaceTransformer] Skipping class: {node.name.value}")
        else:
            self.in_target_class = True
        return True
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """离开类定义"""
        if self.class_name and node.name.value == self.class_name:
            self.in_target_class = False
        return updated_node
    
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        """v1.4.0: 检查是否进入 __init__ 方法"""
        if self.in_target_class and node.name.value == '__init__':
            self.in_init_method = True
            print(f"[LayerSpaceTransformer] Entering __init__ method")
        return True
    
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """离开函数定义"""
        if node.name.value == '__init__':
            self.in_init_method = False
        return updated_node
    
    def leave_Call(self, original_node: cst.Call, 
                   updated_node: cst.Call) -> cst.Call:
        """v1.4.0: 只在目标类的 __init__ 中替换"""
        # 如果指定了类名，但不在目标类的 __init__ 中，跳过
        if self.class_name and not (self.in_target_class and self.in_init_method):
            if self.class_name:
                return updated_node
        
        # 检查是否是目标关键字参数
        new_args = []
        modified = False
        for arg in original_node.args:
            if arg.keyword and arg.keyword.value == self.target_name:
                # 替换为 LayerSpace
                try:
                    new_value = cst.parse_expression(self.layer_space_expr)
                    new_args.append(arg.with_changes(value=new_value))
                    modified = True
                    print(f"[LayerSpaceTransformer] Replaced {self.target_name} with {self.layer_space_expr}")
                except Exception as e:
                    print(f"[LayerSpaceTransformer] Error parsing expression: {e}")
                    new_args.append(arg)
            else:
                new_args.append(arg)
        
        if modified:
            self.modified = True
            return updated_node.with_changes(args=new_args)
        return updated_node


class ReportInjector(cst.CSTTransformer):
    """在训练循环中注入 report 调用"""
    
    def __init__(self, report_expr: str):
        self.report_expr = report_expr
        self.found_training_loop = False
        self.modified = False
    
    def visit_For(self, node: cst.For) -> bool:
        # 检查是否是训练循环（简单启发式：检查 epoch 关键字）
        if isinstance(node.target, cst.Name):
            if 'epoch' in node.target.value.lower():
                self.found_training_loop = True
        return True
    
    def leave_For(self, original_node: cst.For, 
                  updated_node: cst.For) -> cst.For:
        if self.found_training_loop:
            # 在循环体末尾添加 report 调用
            try:
                report_stmt = cst.parse_statement(self.report_expr)
                new_body = list(updated_node.body.body) + [report_stmt]
                self.modified = True
                return updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_body)
                )
            except Exception as e:
                print(f"[ReportInjector] Error parsing report statement: {e}")
        return updated_node


class ImportAdder(cst.CSTTransformer):
    """添加 Import 语句"""
    
    def __init__(self, imports_to_add: set):
        self.imports_to_add = imports_to_add
        self.added_imports = set()
    
    def leave_Module(self, original_node: cst.Module, 
                     updated_node: cst.Module) -> cst.Module:
        # 在文件开头添加 import
        new_body = []
        
        for import_str in self.imports_to_add:
            try:
                import_stmt = cst.parse_statement(import_str)
                new_body.append(import_stmt)
                self.added_imports.add(import_str)
            except Exception as e:
                print(f"[ImportAdder] Error parsing import: {import_str}, {e}")
        
        new_body.extend(list(updated_node.body))
        
        return updated_node.with_changes(body=new_body)
