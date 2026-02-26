"""
MAS Core - Modifier Agent
修改智能体：使用 libcst 进行精准的代码修改
"""
from typing import Dict, Any, List, Optional, Tuple
import libcst as cst
from libcst import matchers as m

from .base_agent import BaseAgent


class ModifierAgent(BaseAgent):
    """
    代码修改智能体
    - 使用 libcst 进行 AST 级别的精准修改
    - 自动处理 Import 插入
    - 保证代码修改的安全性
    """
    
    def __init__(self):
        super().__init__(
            scope_type="modifier",
            scope_name="code_modifier"
        )
        self.modifications: List[Dict[str, Any]] = []
        self.imports_to_add: set = set()
        
    def analyze(self) -> Dict[str, Any]:
        """Modifier Agent 不需要分析"""
        return {'status': 'modifier_ready'}
    
    def get_nas_candidates(self) -> List[Dict[str, Any]]:
        """Modifier Agent 不提供候选"""
        return []
    
    def modify_for_value_space(self, tree: cst.Module, target_name: str, 
                                search_space: List[Any]) -> cst.Module:
        """
        将硬编码数值替换为 ValueSpace
        
        Args:
            tree: libcst 解析树
            target_name: 目标变量名
            search_space: 搜索空间值列表
            
        Returns:
            cst.Module: 修改后的解析树
        """
        self._think(f"Modifying {target_name} with ValueSpace {search_space}")
        
        # 创建 ValueSpace 表达式
        values_str = ", ".join(str(v) for v in search_space)
        value_space_expr = f"ValueSpace([{values_str}])"
        
        # 使用 Transformer 进行替换
        transformer = ValueSpaceTransformer(target_name, value_space_expr)
        new_tree = tree.visit(transformer)
        
        # 记录需要添加的 import
        self.imports_to_add.add("from archmind import ValueSpace")
        
        self._think(f"Transformed {target_name} to {value_space_expr}")
        return new_tree
    
    def modify_for_layer_space(self, tree: cst.Module, target_name: str,
                                layer_options: List[str]) -> cst.Module:
        """
        将层实例化替换为 LayerSpace
        
        Args:
            tree: libcst 解析树
            target_name: 目标变量名
            layer_options: 层选项列表
            
        Returns:
            cst.Module: 修改后的解析树
        """
        self._think(f"Modifying {target_name} with LayerSpace {layer_options}")
        
        # 创建 LayerSpace 表达式
        layers_str = ", ".join(layer_options)
        layer_space_expr = f"LayerSpace([{layers_str}])"
        
        # 使用 Transformer 进行替换
        transformer = LayerSpaceTransformer(target_name, layer_space_expr)
        new_tree = tree.visit(transformer)
        
        # 记录需要添加的 import
        self.imports_to_add.add("from archmind import LayerSpace")
        
        self._think(f"Transformed {target_name} to {layer_space_expr}")
        return new_tree
    
    def add_imports(self, tree: cst.Module) -> cst.Module:
        """
        添加必要的 import 语句
        
        Args:
            tree: libcst 解析树
            
        Returns:
            cst.Module: 修改后的解析树
        """
        if not self.imports_to_add:
            return tree
        
        self._think(f"Adding imports: {self.imports_to_add}")
        
        # 创建 Import 添加器
        import_adder = ImportAdder(self.imports_to_add)
        new_tree = tree.visit(import_adder)
        
        return new_tree
    
    def inject_report_call(self, tree: cst.Module, 
                           metrics: Dict[str, str]) -> cst.Module:
        """
        在训练循环中注入 report 调用
        
        Args:
            tree: libcst 解析树
            metrics: 指标字典，如 {'loss': 'loss', 'accuracy': 'accuracy'}
            
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
        self.imports_to_add.add("from archmind import report")
        
        self._think(f"Injected report call: {report_expr}")
        return new_tree
    
    def apply_modifications(self, file_path: str, 
                           modifications: List[Dict[str, Any]]) -> bool:
        """
        应用所有修改到文件
        
        Args:
            file_path: 文件路径
            modifications: 修改列表
            
        Returns:
            bool: 是否成功
        """
        try:
            self._think(f"Applying modifications to {file_path}")
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # 解析
            tree = cst.parse_module(source)
            
            # 应用每个修改
            for mod in modifications:
                mod_type = mod.get('type')
                
                if mod_type == 'value_space':
                    tree = self.modify_for_value_space(
                        tree, 
                        mod['target'],
                        mod['search_space']
                    )
                elif mod_type == 'layer_space':
                    tree = self.modify_for_layer_space(
                        tree,
                        mod['target'],
                        mod['layer_options']
                    )
                elif mod_type == 'report':
                    tree = self.inject_report_call(
                        tree,
                        mod['metrics']
                    )
            
            # 添加必要的 imports
            tree = self.add_imports(tree)
            
            # 写回文件
            modified_code = tree.code
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            
            self._think(f"Successfully modified {file_path}")
            return True
            
        except Exception as e:
            self._think(f"ERROR modifying {file_path}: {e}")
            return False


# ==================== libcst Transformers ====================

class ValueSpaceTransformer(cst.CSTTransformer):
    """将数值替换为 ValueSpace"""
    
    def __init__(self, target_name: str, value_space_expr: str):
        self.target_name = target_name
        self.value_space_expr = value_space_expr
    
    def leave_Assign(self, original_node: cst.Assign, 
                     updated_node: cst.Assign) -> cst.Assign:
        # 检查是否是目标变量的赋值
        for target in original_node.targets:
            if isinstance(target.target, cst.Name):
                if target.target.value == self.target_name:
                    # 替换为 ValueSpace
                    new_value = cst.parse_expression(self.value_space_expr)
                    return updated_node.with_changes(value=new_value)
        return updated_node


class LayerSpaceTransformer(cst.CSTTransformer):
    """将层替换为 LayerSpace"""
    
    def __init__(self, target_name: str, layer_space_expr: str):
        self.target_name = target_name
        self.layer_space_expr = layer_space_expr
    
    def leave_Call(self, original_node: cst.Call, 
                   updated_node: cst.Call) -> cst.Call:
        # 检查是否是目标关键字参数
        new_args = []
        for arg in original_node.args:
            if arg.keyword and arg.keyword.value == self.target_name:
                # 替换为 LayerSpace
                new_value = cst.parse_expression(self.layer_space_expr)
                new_args.append(arg.with_changes(value=new_value))
            else:
                new_args.append(arg)
        
        return updated_node.with_changes(args=new_args)


class ReportInjector(cst.CSTTransformer):
    """在训练循环中注入 report 调用"""
    
    def __init__(self, report_expr: str):
        self.report_expr = report_expr
        self.found_training_loop = False
    
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
            report_stmt = cst.parse_statement(self.report_expr)
            new_body = list(updated_node.body.body) + [report_stmt]
            return updated_node.with_changes(
                body=updated_node.body.with_changes(body=new_body)
            )
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
            import_stmt = cst.parse_statement(import_str)
            new_body.append(import_stmt)
        
        new_body.extend(list(updated_node.body))
        
        return updated_node.with_changes(body=new_body)
