"""
MAS Core - Report Injector (v1.2.0)
LLM 驱动的 Report 插入器：
- 分析代码结构（函数、类、作用）
- 聚焦训练函数/类
- 找到 model 实例化位置，插入 report(model=model)
- 识别训练循环中的 metrics（loss, accuracy, gain, reward 等）
- 通过代码片段分析而非全文，避免 context 过长
"""
import ast
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import libcst as cst


class ReportInjector:
    """
    LLM 驱动的 Report 插入器
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.report_call_template = "report(model={model_var}{metrics})"
    
    def analyze_file_for_training(self, file_path: Path) -> Dict[str, Any]:
        """
        分析文件，找到训练相关的代码
        
        Returns:
            Dict: {
                "has_training_loop": bool,
                "training_functions": [...],
                "model_instantiation": {...},
                "metrics": {...}
            }
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            result = {
                "has_training_loop": False,
                "training_functions": [],
                "model_instantiation": None,
                "metrics": {}
            }
            
            # 查找训练函数/类
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node, content)
                    if func_info.get("is_training_function"):
                        result["training_functions"].append(func_info)
                        result["has_training_loop"] = True
                
                elif isinstance(node, ast.ClassDef):
                    # 检查类中是否有训练方法
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = self._analyze_function(item, content, class_name=node.name)
                            if method_info.get("is_training_function"):
                                method_info["class_name"] = node.name
                                result["training_functions"].append(method_info)
                                result["has_training_loop"] = True
            
            # 查找模型实例化
            result["model_instantiation"] = self._find_model_instantiation(tree, content)
            
            return result
            
        except Exception as e:
            print(f"[ReportInjector] Error analyzing {file_path}: {e}")
            return {}
    
    def _analyze_function(self, node: ast.FunctionDef, content: str, 
                          class_name: str = None) -> Dict[str, Any]:
        """分析函数是否是训练函数"""
        func_info = {
            "name": node.name,
            "line": node.lineno,
            "is_training_function": False,
            "has_loop": False,
            "metrics": [],
            "model_variable": None
        }
        
        if class_name:
            func_info["class_name"] = class_name
        
        # 检查函数名是否包含训练相关关键词
        training_keywords = ['train', 'fit', 'epoch', 'step', 'optimize']
        if any(kw in node.name.lower() for kw in training_keywords):
            func_info["is_training_function"] = True
        
        # 查找循环
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                func_info["has_loop"] = True
                
                # 检查循环变量是否包含 epoch
                if isinstance(child, ast.For) and isinstance(child.target, ast.Name):
                    if 'epoch' in child.target.id.lower():
                        func_info["is_training_function"] = True
            
            # 查找 metrics 相关代码
            if isinstance(child, ast.Name):
                metric_keywords = ['loss', 'accuracy', 'acc', 'reward', 'gain', 'score', 'metric']
                if any(kw in child.id.lower() for kw in metric_keywords):
                    if child.id not in func_info["metrics"]:
                        func_info["metrics"].append(child.id)
            
            # 查找模型变量
            if isinstance(child, ast.Name):
                if child.id.lower() in ['model', 'net', 'network', 'classifier']:
                    func_info["model_variable"] = child.id
        
        return func_info
    
    def _find_model_instantiation(self, tree: ast.AST, content: str) -> Optional[Dict]:
        """查找模型实例化"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # 检查赋值右侧是否是类实例化
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        class_name = node.value.func.id
                        # 检查是否是模型类（首字母大写）
                        if class_name[0].isupper():
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    return {
                                        "variable": target.id,
                                        "class_name": class_name,
                                        "line": node.lineno
                                    }
        return None
    
    def extract_training_snippet(self, file_path: Path, 
                                  function_info: Dict) -> str:
        """
        提取训练函数的代码片段
        
        Args:
            file_path: 文件路径
            function_info: 函数信息
            
        Returns:
            str: 代码片段
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_line = function_info.get("line", 1) - 1
            
            # 提取函数体（简化处理，提取最多 50 行）
            end_line = min(start_line + 50, len(lines))
            
            snippet_lines = lines[start_line:end_line]
            return ''.join(snippet_lines)
            
        except Exception as e:
            print(f"[ReportInjector] Error extracting snippet: {e}")
            return ""
    
    def llm_analyze_training_snippet(self, snippet: str) -> Dict[str, Any]:
        """
        使用 LLM 分析训练代码片段
        
        Args:
            snippet: 代码片段
            
        Returns:
            Dict: LLM 分析结果
        """
        if not self.llm_client:
            return {}
        
        return self.llm_client.find_training_function_and_metrics(snippet)
    
    def generate_report_call(self, model_variable: str, 
                             metrics: Dict[str, str]) -> str:
        """
        生成 report 调用代码
        
        Args:
            model_variable: 模型变量名
            metrics: 指标字典，如 {"loss": "loss", "accuracy": "acc"}
            
        Returns:
            str: report 调用代码
        """
        metrics_str = ""
        for key, value in metrics.items():
            metrics_str += f", {key}={value}"
        
        return f"report(model={model_variable}{metrics_str})"
    
    def find_insertion_point(self, file_path: Path, 
                             function_info: Dict) -> Optional[int]:
        """
        找到 report 插入位置
        
        Args:
            file_path: 文件路径
            function_info: 函数信息
            
        Returns:
            int: 插入行号
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            func_start = function_info.get("line", 1) - 1
            
            # 查找 epoch 循环的结束位置
            # 简化处理：在函数中查找 print 语句或循环结束后的位置
            for i in range(func_start, min(func_start + 100, len(lines))):
                line = lines[i]
                # 在 epoch 循环结束后插入
                if 'print' in line and ('epoch' in line.lower() or 'loss' in line.lower()):
                    return i + 1  # 在 print 语句后插入
            
            # 默认在函数结束前插入
            return func_start + min(30, len(lines) - func_start)
            
        except Exception as e:
            print(f"[ReportInjector] Error finding insertion point: {e}")
            return None
    
    def inject_report(self, file_path: Path, 
                      function_info: Dict,
                      llm_analysis: Dict[str, Any]) -> bool:
        """
        在文件中注入 report 调用
        
        Args:
            file_path: 文件路径
            function_info: 函数信息
            llm_analysis: LLM 分析结果
            
        Returns:
            bool: 是否成功
        """
        try:
            # 获取模型变量名
            model_var = llm_analysis.get("model_variable") or function_info.get("model_variable", "model")
            
            # 获取 metrics
            metrics = llm_analysis.get("metrics", {})
            if not metrics:
                # 使用默认 metrics
                metrics = {"loss": "avg_loss", "accuracy": "accuracy"}
            
            # 生成 report 调用
            report_call = self.generate_report_call(model_var, metrics)
            
            # 找到插入位置
            insert_line = self.find_insertion_point(file_path, function_info)
            if insert_line is None:
                return False
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 插入 report 调用
            indent = self._get_indent(lines[insert_line - 1]) if insert_line <= len(lines) else "        "
            lines.insert(insert_line, f"{indent}{report_call}\n")
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print(f"[ReportInjector] Injected report call at line {insert_line + 1}")
            return True
            
        except Exception as e:
            print(f"[ReportInjector] Error injecting report: {e}")
            return False
    
    def _get_indent(self, line: str) -> str:
        """获取行的缩进"""
        match = re.match(r'^(\s*)', line)
        return match.group(1) if match else "        "
    
    def inject_report_to_file(self, file_path: Path) -> bool:
        """
        对文件进行完整的 report 注入流程
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否成功
        """
        print(f"[ReportInjector] Analyzing {file_path}...")
        
        # 1. 分析文件
        analysis = self.analyze_file_for_training(file_path)
        
        if not analysis.get("has_training_loop"):
            print(f"[ReportInjector] No training loop found in {file_path}")
            return False
        
        # 2. 对每个训练函数使用 LLM 分析
        for func_info in analysis.get("training_functions", []):
            print(f"[ReportInjector] Analyzing function: {func_info['name']}")
            
            # 提取代码片段
            snippet = self.extract_training_snippet(file_path, func_info)
            
            # LLM 分析
            if self.llm_client:
                llm_result = self.llm_analyze_training_snippet(snippet)
                print(f"[ReportInjector] LLM analysis: {llm_result}")
            else:
                llm_result = {}
            
            # 注入 report
            success = self.inject_report(file_path, func_info, llm_result)
            if success:
                print(f"[ReportInjector] Successfully injected report in {func_info['name']}")
            
            return success  # 只处理第一个训练函数
        
        return False


class ReportInjectorCST(cst.CSTTransformer):
    """
    使用 libcst 的 Report 注入器
    更精准的 AST 级别修改
    """
    
    def __init__(self, model_variable: str = "model", 
                 metrics: Dict[str, str] = None):
        self.model_variable = model_variable
        self.metrics = metrics or {}
        self.modified = False
        self.found_training_loop = False
    
    def visit_For(self, node: cst.For) -> bool:
        """检查是否是训练循环"""
        if isinstance(node.target, cst.Name):
            if 'epoch' in node.target.value.lower():
                self.found_training_loop = True
        return True
    
    def leave_For(self, original_node: cst.For, 
                  updated_node: cst.For) -> cst.For:
        """在训练循环中插入 report 调用"""
        if self.found_training_loop and not self.modified:
            # 构建 report 调用
            args = [
                cst.Arg(
                    keyword=cst.Name("model"),
                    value=cst.Name(self.model_variable),
                    equal=cst.AssignEqual(
                        whitespace_before=cst.SimpleWhitespace(""),
                        whitespace_after=cst.SimpleWhitespace("")
                    )
                )
            ]
            
            # 添加 metrics
            for key, value in self.metrics.items():
                args.append(
                    cst.Arg(
                        keyword=cst.Name(key),
                        value=cst.Name(value),
                        equal=cst.AssignEqual(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.SimpleWhitespace("")
                        )
                    )
                )
            
            report_call = cst.SimpleStatementLine(
                body=[
                    cst.Expr(
                        value=cst.Call(
                            func=cst.Name("report"),
                            args=args
                        )
                    )
                ]
            )
            
            # 在循环体末尾添加
            new_body = list(updated_node.body.body) + [report_call]
            self.modified = True
            
            return updated_node.with_changes(
                body=updated_node.body.with_changes(body=new_body)
            )
        
        return updated_node


def inject_report_to_project(project_path: str, 
                              entry_file: str,
                              llm_client=None) -> List[str]:
    """
    对整个项目进行 report 注入
    
    Args:
        project_path: 项目路径
        entry_file: 入口文件
        llm_client: LLM 客户端
        
    Returns:
        List[str]: 修改的文件列表
    """
    injector = ReportInjector(llm_client)
    modified_files = []
    
    project = Path(project_path)
    entry = project / entry_file
    
    # 分析入口文件
    if entry.exists():
        if injector.inject_report_to_file(entry):
            modified_files.append(str(entry))
    
    # 查找其他可能包含训练代码的文件
    for py_file in project.rglob("*.py"):
        if py_file == entry:
            continue
        if any(part.startswith('.') or part in ['__pycache__', 'venv'] 
               for part in py_file.parts):
            continue
        
        # 快速检查是否包含训练相关代码
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'def train' in content or 'for epoch' in content:
                if injector.inject_report_to_file(py_file):
                    modified_files.append(str(py_file))
        except:
            pass
    
    return modified_files
