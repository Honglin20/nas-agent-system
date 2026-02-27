"""
NAS CLI - 交互式智能 NAS 寻优空间注入工具 v1.2.0
增强版：
- 智能模型识别
- 跨文件参数修改
- LLM 驱动的 Report 插入
- 寻优空间张开
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# 导入 MAS 核心
sys.path.insert(0, str(Path(__file__).parent.parent))
from mas_core import (
    NASOrchestrator, ScopeAgent, CentralRegistry, 
    init_llm, get_llm_client, ModifierAgent,
    # v1.2.0 新增
    ModelDiscoveryAnalyzer,
    CrossFileParameterModifier,
    SearchSpaceExpander,
    inject_report_to_project
)

console = Console()

__version__ = "1.2.0"


@dataclass
class NASCandidate:
    """NAS 候选参数"""
    name: str
    param_type: str  # 'value' or 'layer'
    current_value: Any
    location: str
    line: int
    recommended: bool = True
    reason: str = ""
    search_space: List[Any] = field(default_factory=list)
    selected: bool = True
    is_config: bool = False  # v1.2.0: 是否是配置文件中的参数


class InteractiveNASCLI:
    """交互式 NAS CLI v1.2.0"""
    
    def __init__(self):
        self.current_dir = Path.cwd()
        self.target_dir: Optional[Path] = None
        self.entry_file: Optional[str] = None
        self.candidates: List[NASCandidate] = []
        self.scanned_files: List[str] = []
        self.console = Console()
        self.llm = None
        self.modifier_agent = ModifierAgent()
        
        # v1.2.0 新增组件
        self.model_discovery: Optional[ModelDiscoveryAnalyzer] = None
        self.cross_file_modifier: Optional[CrossFileParameterModifier] = None
        self.search_space_expander: Optional[SearchSpaceExpander] = None
        
    def show_banner(self):
        """显示欢迎界面"""
        banner = f"""
╭────────────────────────────────────────────────────────────╮
│                                                            │
│   🧠 NAS-CLI 智能神经网络架构搜索工具 v{__version__}               │
│                                                            │
│   增强功能:                                                │
│   • 智能模型识别 (动态反射解析)                           │
│   • 跨文件参数修改                                        │
│   • LLM 驱动的 Report 插入                                │
│   • 寻优空间张开                                          │
│                                                            │
╰────────────────────────────────────────────────────────────╯
        """
        self.console.print(Panel(banner, style="bold blue"))
    
    def ask_directory(self) -> Path:
        """询问目标目录"""
        self.console.print("\n[bold cyan]📁 步骤 1: 选择目标项目目录[/bold cyan]")
        self.console.print(f"当前目录: [dim]{self.current_dir}[/dim]\n")
        
        while True:
            completer = PathCompleter(only_directories=True)
            path_input = prompt(
                "请输入项目目录路径 (支持 Tab 补全): ",
                completer=completer,
                default=str(self.current_dir)
            ).strip()
            
            target = Path(path_input).expanduser().resolve()
            
            if not target.exists():
                self.console.print(f"[red]❌ 目录不存在: {target}[/red]")
                continue
            
            if not target.is_dir():
                self.console.print(f"[red]❌ 这不是一个目录: {target}[/red]")
                continue
            
            self.console.print(f"\n[green]✓ 已选择目录:[/green] {target}")
            self.show_directory_preview(target)
            
            if Confirm.ask("确认使用此目录?", default=True):
                self.target_dir = target
                os.chdir(target)
                
                # v1.2.0: 初始化跨文件修改器
                self.cross_file_modifier = CrossFileParameterModifier(str(target))
                
                return target
    
    def show_directory_preview(self, path: Path):
        """显示目录预览"""
        tree = Tree(f"📂 {path.name}")
        
        try:
            items = list(path.iterdir())[:20]
            for item in items:
                if item.is_dir():
                    if not item.name.startswith('.') and item.name not in ['__pycache__', 'venv', 'env']:
                        tree.add(f"📁 {item.name}/")
                elif item.suffix == '.py':
                    tree.add(f"🐍 {item.name}")
                elif item.suffix in ['.yaml', '.yml', '.json']:
                    tree.add(f"⚙️  {item.name}")
            
            if len(list(path.iterdir())) > 20:
                tree.add("...")
                
        except PermissionError:
            tree.add("[red]权限不足[/red]")
        
        self.console.print(tree)
    
    def ask_entry_file(self) -> str:
        """询问入口文件"""
        self.console.print("\n[bold cyan]📄 步骤 2: 选择入口脚本[/bold cyan]\n")
        
        py_files = []
        for f in self.target_dir.rglob("*.py"):
            if not any(part.startswith('.') or part in ['__pycache__', 'venv', 'env'] 
                      for part in f.parts):
                py_files.append(f)
        
        priority_names = ['main.py', 'train.py', 'run.py', 'app.py', 'server.py']
        py_files.sort(key=lambda x: (0 if x.name in priority_names else 1, x.name))
        
        if not py_files:
            self.console.print("[red]❌ 未找到 Python 文件[/red]")
            return ""
        
        table = Table(title="发现的 Python 文件")
        table.add_column("序号", style="cyan", justify="center")
        table.add_column("文件名", style="green")
        table.add_column("路径", style="dim")
        table.add_column("推荐", style="yellow")
        
        for i, f in enumerate(py_files[:15], 1):
            rel_path = f.relative_to(self.target_dir)
            is_recommended = "⭐" if f.name in priority_names else ""
            table.add_row(str(i), f.name, str(rel_path), is_recommended)
        
        self.console.print(table)
        
        while True:
            choice = Prompt.ask(
                "\n请选择入口文件 (输入序号或完整路径)",
                default="1"
            )
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(py_files[:15]):
                    selected = py_files[idx]
                    self.entry_file = str(selected.relative_to(self.target_dir))
                    break
            except ValueError:
                file_path = self.target_dir / choice
                if file_path.exists():
                    self.entry_file = choice
                    break
            
            self.console.print("[red]❌ 无效选择，请重试[/red]")
        
        self.console.print(f"[green]✓ 已选择入口文件:[/green] {self.entry_file}")
        return self.entry_file
    
    def scan_project(self):
        """扫描项目 - v1.2.0 增强版"""
        self.console.print("\n[bold cyan]🔍 步骤 3: 扫描项目架构[/bold cyan]\n")
        
        # 初始化 LLM
        self.llm = get_llm_client()
        
        # 发现 Python 文件
        self.console.print("[yellow]📂 发现 Python 文件...[/yellow]")
        py_files = []
        for f in self.target_dir.rglob("*.py"):
            if not any(part.startswith('.') or part in ['__pycache__', 'venv', 'env', '.git'] 
                      for part in f.parts):
                py_files.append(f)
        self.console.print(f"[green]✓ 发现 {len(py_files)} 个 Python 文件[/green]\n")
        
        self.scanned_files = [str(f.relative_to(self.target_dir)) for f in py_files]
        
        # v1.2.0: 智能模型发现
        if self.entry_file:
            self.console.print("[yellow]🤖 正在进行智能模型发现...[/yellow]")
            self.model_discovery = ModelDiscoveryAnalyzer(
                str(self.target_dir), 
                self.llm
            )
            entry_path = self.target_dir / self.entry_file
            discovery_result = self.model_discovery.run_full_discovery(entry_path)
            
            if discovery_result.get("instantiated_model"):
                model_info = discovery_result["instantiated_model"]
                self.console.print(f"[green]✓ 识别到实际被实例化的模型:[/green]")
                self.console.print(f"  • 模型: [cyan]{model_info.get('instantiated_model')}[/cyan]")
                self.console.print(f"  • 变量: [cyan]{model_info.get('model_variable')}[/cyan]")
                self.console.print(f"  • 置信度: [cyan]{model_info.get('confidence')}[/cyan]\n")
        
        # 分析所有文件
        all_agents = {}
        for f in py_files:
            rel_path = str(f.relative_to(self.target_dir))
            self.console.print(f"[yellow]🤖 LLM 正在分析: {rel_path}[/yellow]")
            
            agent = ScopeAgent(str(f))
            if agent.load_file():
                analysis = agent.analyze()
                all_agents[rel_path] = agent
                
                # 实时打印该文件的候选
                candidates = analysis.get('nas_candidates', [])
                if candidates:
                    self.console.print(f"[green]  ↳ 发现 {len(candidates)} 个候选[/green]")
                    for cand in candidates[:3]:
                        self.console.print(f"    • [cyan]{cand.get('name')}[/cyan] = [yellow]{cand.get('current_value')}[/yellow]")
                    if len(candidates) > 3:
                        self.console.print(f"    ... 还有 {len(candidates) - 3} 个")
        
        self.console.print()
        
        # v1.2.0: 也查找配置文件中的参数
        self.console.print("[yellow]📂 查找配置文件...[/yellow]")
        config_candidates = self._scan_config_files()
        if config_candidates:
            self.console.print(f"[green]✓ 从配置文件发现 {len(config_candidates)} 个候选[/green]\n")
        
        # 收集所有候选
        self.console.print("[yellow]📊 收集所有 NAS 候选...[/yellow]")
        self.candidates = []
        
        for file_path, agent in all_agents.items():
            for cand in agent.get_nas_candidates():
                # 使用 LLM 生成搜索空间
                search_space = self.llm.generate_search_space(
                    cand['name'],
                    cand['current_value'],
                    cand['type']
                )
                
                nas_cand = NASCandidate(
                    name=cand['name'],
                    param_type=cand['type'],
                    current_value=cand['current_value'],
                    location=file_path,
                    line=cand.get('line', 0),
                    recommended=True,
                    reason=cand.get('reason', ''),
                    search_space=search_space
                )
                self.candidates.append(nas_cand)
        
        # 添加配置文件候选
        for cand in config_candidates:
            self.candidates.append(NASCandidate(
                name=cand['name'],
                param_type=cand['type'],
                current_value=cand['current_value'],
                location=cand['source_file'],
                line=0,
                recommended=True,
                reason=cand.get('reason', ''),
                search_space=cand.get('search_space', [cand['current_value']]),
                is_config=True
            ))
        
        # 使用 LLM 推荐
        if self.candidates:
            self.console.print(f"[yellow]🤖 LLM 正在评估 {len(self.candidates)} 个候选的推荐优先级...[/yellow]")
            cand_dicts = [
                {
                    'name': c.name,
                    'type': c.param_type,
                    'current_value': str(c.current_value),
                    'reason': c.reason
                }
                for c in self.candidates
            ]
            recommendations = self.llm.recommend_injection(cand_dicts)
            
            rec_map = {r['name']: r for r in recommendations}
            for cand in self.candidates:
                if cand.name in rec_map:
                    cand.recommended = rec_map[cand.name].get('recommended', True)
                    cand.reason = rec_map[cand.name].get('reason', cand.reason)
            
            self.console.print(f"[green]✓ LLM 推荐 {sum(1 for c in self.candidates if c.recommended)}/{len(self.candidates)} 个参数[/green]")
        
        self.console.print()
        self.show_scan_results(all_agents, config_candidates)
    
    def _scan_config_files(self) -> List[Dict]:
        """v1.2.0: 扫描配置文件"""
        from mas_core.cross_file_modifier import ConfigFileHandler
        
        candidates = []
        
        # 查找 Python 配置文件
        for config_file in self.target_dir.rglob("*_config.py"):
            if any(part.startswith('.') for part in config_file.parts):
                continue
            
            try:
                config = ConfigFileHandler.load_config(config_file)
                rel_path = str(config_file.relative_to(self.target_dir))
                
                # 递归查找数值参数
                self._extract_from_dict(config, rel_path, candidates)
            except Exception as e:
                pass
        
        return candidates
    
    def _extract_from_dict(self, data: Dict, file_path: str, 
                           candidates: List, prefix: str = ""):
        """从字典中提取候选参数"""
        nas_keywords = [
            'lr', 'learning_rate', 'batch_size', 'epoch', 'dropout', 
            'dim', 'hidden', 'layer', 'head', 'rate', 'weight_decay',
            'momentum', 'beta', 'gamma', 'alpha'
        ]
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._extract_from_dict(value, file_path, candidates, full_key)
            elif isinstance(value, (int, float)):
                if any(kw in key.lower() for kw in nas_keywords):
                    search_space = self._generate_search_space(value, key)
                    candidates.append({
                        'name': full_key,
                        'type': 'value',
                        'current_value': value,
                        'source_file': file_path,
                        'search_space': search_space,
                        'reason': f'Configuration parameter: {key}'
                    })
            elif isinstance(value, str):
                if key.lower() in ['activation', 'optimizer', 'norm', 'loss']:
                    candidates.append({
                        'name': full_key,
                        'type': 'layer',
                        'current_value': value,
                        'source_file': file_path,
                        'search_space': self._generate_layer_options(key, value),
                        'reason': f'Layer/optimizer selection: {key}'
                    })
    
    def _generate_search_space(self, value, name):
        """生成搜索空间"""
        if isinstance(value, (int, float)):
            if 'lr' in name.lower() or 'rate' in name.lower():
                if value < 1:
                    return [value / 10, value, value * 10]
            return [max(1, int(value / 2)), value, value * 2]
        return [value]
    
    def _generate_layer_options(self, name, value):
        """生成层选项"""
        if 'activation' in name.lower():
            return ['relu', 'sigmoid', 'tanh', 'gelu']
        elif 'optimizer' in name.lower():
            return ['Adam', 'SGD', 'RMSprop']
        elif 'norm' in name.lower():
            return ['batchnorm', 'layernorm']
        return [value]
    
    def show_scan_results(self, all_agents: Dict, config_candidates: List):
        """显示扫描结果"""
        self.console.print("\n[bold green]✓ 扫描完成![/bold green]\n")
        
        tree = Tree("📂 项目结构")
        for file_path in sorted(self.scanned_files)[:20]:
            tree.add(f"🐍 {file_path}")
        
        if len(self.scanned_files) > 20:
            tree.add(f"... 还有 {len(self.scanned_files) - 20} 个文件")
        
        self.console.print(tree)
        
        stats = Table(title="扫描统计")
        stats.add_column("指标", style="cyan")
        stats.add_column("数值", style="green")
        stats.add_row("Python 文件数", str(len(self.scanned_files)))
        stats.add_row("代码候选数", str(len(self.candidates) - len(config_candidates)))
        stats.add_row("配置候选数", str(len(config_candidates)))
        stats.add_row("总候选数", str(len(self.candidates)))
        self.console.print(stats)
        
        # 显示候选
        if self.candidates:
            self.console.print("\n[bold]识别的 NAS 候选:[/bold]")
            for cand in self.candidates[:5]:
                rec = "⭐" if cand.recommended else ""
                config_mark = "⚙️ " if cand.is_config else ""
                self.console.print(f"  • {config_mark}{cand.name} = {cand.current_value} {rec}")
                self.console.print(f"    [dim]{cand.reason}[/dim]")
            if len(self.candidates) > 5:
                self.console.print(f"  ... 还有 {len(self.candidates) - 5} 个")
    
    def select_candidates(self) -> bool:
        """让用户选择候选参数"""
        self.console.print("\n[bold cyan]⚙️  步骤 4: 配置 NAS 寻优空间[/bold cyan]\n")
        
        if not self.candidates:
            self.console.print("[yellow]⚠️  未发现 NAS 候选参数[/yellow]")
            return False
        
        # 显示候选表格
        table = Table(title=f"共发现 {len(self.candidates)} 个 NAS 寻优候选")
        table.add_column("序号", style="cyan", justify="center")
        table.add_column("参数名", style="green")
        table.add_column("当前值", style="yellow")
        table.add_column("类型", style="blue")
        table.add_column("位置", style="dim")
        table.add_column("来源", style="magenta")
        
        for i, cand in enumerate(self.candidates, 1):
            source = "⚙️ 配置" if cand.is_config else "🐍 代码"
            rec_mark = "⭐ 推荐" if cand.recommended else ""
            table.add_row(
                str(i),
                cand.name,
                str(cand.current_value),
                cand.param_type,
                cand.location,
                f"{source} {rec_mark}"
            )
        
        self.console.print(table)
        
        # 选择方式
        self.console.print("\n[bold]选择方式:[/bold]")
        self.console.print("  [1] 使用 LLM 推荐参数 (带⭐标记)")
        self.console.print("  [2] 全选所有参数")
        self.console.print("  [3] 手动逐个选择")
        self.console.print("  [4] 输入序号范围选择 (如: 1,3,5-7)")
        
        choice = Prompt.ask("请选择", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            for cand in self.candidates:
                cand.selected = cand.recommended
        elif choice == "2":
            for cand in self.candidates:
                cand.selected = True
        elif choice == "3":
            for cand in self.candidates:
                default = "y" if cand.recommended else "n"
                cand.selected = Confirm.ask(
                    f"选择 '{cand.name}' = {cand.current_value}?",
                    default=(default == "y")
                )
        elif choice == "4":
            range_input = Prompt.ask("请输入序号 (如: 1,3,5-7,10)")
            selected_indices = self._parse_range(range_input)
            for i, cand in enumerate(self.candidates, 1):
                cand.selected = i in selected_indices
        
        selected = [c for c in self.candidates if c.selected]
        self.console.print(f"\n[green]✓ 已选择 {len(selected)}/{len(self.candidates)} 个参数[/green]")
        
        # 自定义搜索空间
        if selected and Confirm.ask("\n是否自定义寻优空间?", default=False):
            self._customize_search_space(selected)
        
        return len(selected) > 0
    
    def _parse_range(self, range_str: str) -> set:
        """解析序号范围字符串"""
        result = set()
        parts = range_str.replace(' ', '').split(',')
        for part in parts:
            if '-' in part:
                start, end = part.split('-')
                result.update(range(int(start), int(end) + 1))
            else:
                result.add(int(part))
        return result
    
    def _customize_search_space(self, selected: List[NASCandidate]):
        """让用户自定义搜索空间"""
        self.console.print("\n[bold cyan]🔧 自定义寻优空间[/bold cyan]")
        self.console.print("[dim]提示: 直接回车保持默认，或输入自定义值 (如: 32,64,128,256)[/dim]\n")
        
        for cand in selected:
            self.console.print(f"\n[bold]{cand.name}[/bold]")
            self.console.print(f"  当前值: [yellow]{cand.current_value}[/yellow]")
            self.console.print(f"  默认搜索空间: [dim]{cand.search_space}[/dim]")
            
            custom = Prompt.ask("  自定义搜索空间 (回车跳过)", default="")
            if custom.strip():
                try:
                    if ',' in custom:
                        values = [v.strip() for v in custom.split(',')]
                        parsed = []
                        for v in values:
                            try:
                                if '.' in v:
                                    parsed.append(float(v))
                                else:
                                    parsed.append(int(v))
                            except ValueError:
                                parsed.append(v)
                        cand.search_space = parsed
                        self.console.print(f"  [green]✓ 已设置为: {parsed}[/green]")
                    else:
                        try:
                            if '.' in custom:
                                cand.search_space = [float(custom)]
                            else:
                                cand.search_space = [int(custom)]
                        except ValueError:
                            cand.search_space = [custom]
                        self.console.print(f"  [green]✓ 已设置为: {cand.search_space}[/green]")
                except Exception as e:
                    self.console.print(f"  [red]✗ 解析失败，保持默认: {e}[/red]")
    
    def show_diff_and_confirm(self) -> bool:
        """显示修改差异并确认"""
        self.console.print("\n[bold cyan]📝 步骤 5: 确认修改[/bold cyan]\n")
        
        selected = [c for c in self.candidates if c.selected]
        
        by_file: Dict[str, List[NASCandidate]] = {}
        for cand in selected:
            if cand.location not in by_file:
                by_file[cand.location] = []
            by_file[cand.location].append(cand)
        
        for file_path, cands in by_file.items():
            self.console.print(f"\n[bold]文件: {file_path}[/bold]")
            
            for cand in cands:
                before = f"{cand.name} = {cand.current_value}"
                
                if cand.param_type == 'value':
                    after = f"{cand.name} = ValueSpace({cand.search_space})"
                else:
                    after = f"{cand.name} = LayerSpace({cand.search_space})"
                
                self.console.print(f"  [red]- {before}[/red]")
                self.console.print(f"  [green]+ {after}[/green]")
                self.console.print(f"    [dim]{cand.reason}[/dim]\n")
        
        return Confirm.ask("\n确认执行以上修改?", default=True)
    
    def create_backup(self):
        """创建备份"""
        self.console.print("\n[bold cyan]💾 创建备份...[/bold cyan]")
        
        backup_dir = self.target_dir / ".nas_backup"
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        backup_dir.mkdir(exist_ok=True)
        
        # 备份所有 Python 文件
        for py_file in self.target_dir.rglob("*.py"):
            if any(part.startswith('.') or part in ['__pycache__', 'venv'] 
                   for part in py_file.parts):
                continue
            
            rel_path = py_file.relative_to(self.target_dir)
            backup_path = backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, backup_path)
        
        self.console.print(f"[green]✓ 备份已创建: {backup_dir}[/green]")
        return backup_dir
    
    def apply_modifications(self):
        """应用修改 - v1.2.0 增强版"""
        self.console.print("\n[bold cyan]🔧 步骤 6: 应用修改[/bold cyan]\n")
        
        selected = [c for c in self.candidates if c.selected]
        
        # 按文件分组
        by_file: Dict[str, List[NASCandidate]] = {}
        for cand in selected:
            if cand.location not in by_file:
                by_file[cand.location] = []
            by_file[cand.location].append(cand)
        
        success_count = 0
        fail_count = 0
        
        with Progress(console=self.console) as progress:
            task = progress.add_task("[yellow]修改文件中...", total=len(by_file))
            
            for file_path, cands in by_file.items():
                full_path = self.target_dir / file_path
                
                # 分离代码修改和配置修改
                code_mods = [c for c in cands if not c.is_config]
                config_mods = [c for c in cands if c.is_config]
                
                success = True
                
                # 应用代码修改
                if code_mods:
                    modifications = []
                    for cand in code_mods:
                        if cand.param_type == 'value':
                            mod = {
                                'type': 'value_space',
                                'target': cand.name,
                                'search_space': cand.search_space,
                                'line': cand.line
                            }
                        else:
                            mod = {
                                'type': 'layer_space',
                                'target': cand.name,
                                'layer_options': [str(v) for v in cand.search_space],
                                'line': cand.line
                            }
                        modifications.append(mod)
                    
                    try:
                        if not self.modifier_agent.apply_modifications(
                            str(full_path), modifications
                        ):
                            success = False
                    except Exception as e:
                        self.console.print(f"[red]  ✗ 修改失败 {file_path}: {e}[/red]")
                        success = False
                
                # 应用配置修改
                for cand in config_mods:
                    try:
                        from mas_core.cross_file_modifier import ConfigFileHandler
                        key_path = cand.name.split('.')
                        
                        if cand.param_type == 'value':
                            new_value = f"ValueSpace({cand.search_space})"
                        else:
                            new_value = f"LayerSpace({cand.search_space})"
                        
                        if not ConfigFileHandler.modify_config_value(
                            full_path, key_path, new_value
                        ):
                            success = False
                    except Exception as e:
                        self.console.print(f"[red]  ✗ 配置修改失败 {cand.name}: {e}[/red]")
                        success = False
                
                if success:
                    success_count += 1
                    self.console.print(f"[green]  ✓ 已修改: {file_path}[/green]")
                else:
                    fail_count += 1
                
                progress.advance(task)
        
        self.console.print(f"\n[green]✓ 修改完成![/green] 成功: {success_count}, 失败: {fail_count}")
    
    def run_search_space_expansion(self):
        """v1.2.0: 运行寻优空间张开"""
        self.console.print("\n[bold cyan]🌐 步骤 7: 寻优空间张开[/bold cyan]\n")
        
        self.search_space_expander = SearchSpaceExpander(self.llm)
        expanded_files = self.search_space_expander.expand_project(str(self.target_dir))
        
        if expanded_files:
            self.console.print(f"[green]✓ 已张开 {len(expanded_files)} 个文件:[/green]")
            for f in expanded_files:
                self.console.print(f"  • {f}")
        else:
            self.console.print("[dim]未发现需要张开的条件层选择[/dim]")
    
    def run_report_injection(self):
        """v1.2.0: 运行 Report 注入"""
        self.console.print("\n[bold cyan]📊 步骤 8: Report 注入[/bold cyan]\n")
        
        if not self.entry_file:
            self.console.print("[yellow]⚠️  未指定入口文件，跳过 report 注入[/yellow]")
            return
        
        modified_files = inject_report_to_project(
            str(self.target_dir),
            self.entry_file,
            self.llm
        )
        
        if modified_files:
            self.console.print(f"[green]✓ 已注入 report 到 {len(modified_files)} 个文件:[/green]")
            for f in modified_files:
                self.console.print(f"  • {f}")
        else:
            self.console.print("[dim]未发现需要注入 report 的文件[/dim]")
    
    def run(self):
        """运行完整流程 v1.2.0"""
        self.show_banner()
        
        if self.target_dir is None:
            self.ask_directory()
        else:
            self.console.print(f"\n[green]✓ 使用指定目录:[/green] {self.target_dir}")
            self.show_directory_preview(self.target_dir)
        
        if self.entry_file is None:
            self.ask_entry_file()
        else:
            self.console.print(f"\n[green]✓ 使用指定入口文件:[/green] {self.entry_file}")
        
        if not Confirm.ask("\n确认开始扫描?", default=True):
            self.console.print("[yellow]已取消[/yellow]")
            return
        
        self.scan_project()
        
        if not self.select_candidates():
            self.console.print("[yellow]未选择任何参数，退出[/yellow]")
            return
        
        if not self.show_diff_and_confirm():
            self.console.print("[yellow]已取消修改[/yellow]")
            return
        
        # 创建备份
        self.create_backup()
        
        # 应用修改
        self.apply_modifications()
        
        # v1.2.0: 寻优空间张开
        self.run_search_space_expansion()
        
        # v1.2.0: Report 注入
        self.run_report_injection()
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold green]🎉 NAS 寻优空间注入完成![/bold green]")
        self.console.print("="*60)


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="NAS-CLI 智能神经网络架构搜索工具 v1.2.0 (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  nas-cli              启动交互式界面
  nas-cli --version    显示版本信息
  
环境变量:
  OPENAI_API_KEY       LLM API Key
  OPENAI_BASE_URL      LLM API URL
        """
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--dir', '-d', help='目标项目目录')
    parser.add_argument('--entry', '-e', help='入口文件')
    
    args = parser.parse_args()
    
    # 初始化 LLM
    api_key = os.getenv('OPENAI_API_KEY', 'sk-IA0OXgtva7EmahBVdzkCJgcJxnmo4ja6O0M0M146HniteI3m')
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.moonshot.cn/v1')
    
    try:
        init_llm(api_key, base_url)
        console.print("[dim]✓ LLM 客户端初始化成功[/dim]")
    except Exception as e:
        console.print(f"[red]✗ LLM 初始化失败: {e}[/red]")
        sys.exit(1)
    
    cli = InteractiveNASCLI()
    
    if args.dir:
        cli.target_dir = Path(args.dir).expanduser().resolve()
        os.chdir(cli.target_dir)
    if args.entry:
        cli.entry_file = args.entry
    
    try:
        cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]错误: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
