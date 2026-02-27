"""
NAS CLI - 交互式智能 NAS 寻优空间注入工具 v1.3.1
增强版：
- 智能模型识别
- 跨文件参数修改
- LLM 驱动的 Report 插入
- 寻优空间张开
- 完善的错误处理
- 配置持久化
- 撤销/重做功能
- 代理支持
- 备份增强和切换
- 完成后流程优化
"""
import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
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
    inject_report_to_project,
    # v1.3.0 新增
    Config, ConfigManager, get_config, load_config,
    BackupManager, Operation,
    NASCLIError, ErrorCode, get_user_friendly_message,
    is_llm_available,
)

console = Console()

__version__ = "1.3.1"


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
    """交互式 NAS CLI v1.3.1"""
    
    def __init__(self, config: Optional[Config] = None):
        self.current_dir = Path.cwd()
        self.target_dir: Optional[Path] = None
        self.entry_file: Optional[str] = None
        self.candidates: List[NASCandidate] = []
        self.scanned_files: List[str] = []
        self.console = Console()
        self.llm = None
        self.modifier_agent = ModifierAgent()
        
        # v1.3.0: 配置
        self.config = config or get_config()
        
        # v1.2.0 新增组件
        self.model_discovery: Optional[ModelDiscoveryAnalyzer] = None
        self.cross_file_modifier: Optional[CrossFileParameterModifier] = None
        self.search_space_expander: Optional[SearchSpaceExpander] = None
        
        # v1.3.0: 备份管理器
        self.backup_manager: Optional[BackupManager] = None
        self.current_operation: Optional[Operation] = None
        
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
│   • 完善的错误处理与重试机制                              │
│   • 配置持久化                                            │
│   • 撤销/重做功能                                         │
│   • 代理支持                                              │
│   • 备份增强与快速切换                                    │
│                                                            │
╰────────────────────────────────────────────────────────────╯
        """
        self.console.print(Panel(banner, style="bold blue"))
    
    def ask_directory(self) -> Path:
        """询问目标目录"""
        self.console.print("\n[bold cyan]📁 步骤 1: 选择目标项目目录[/bold cyan]")
        self.console.print(f"当前目录: [dim]{self.current_dir}[/dim]\n")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
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
                
                # 检查目录权限
                if not os.access(target, os.R_OK):
                    self.console.print(f"[red]❌ 没有读取权限: {target}[/red]")
                    continue
                
                self.console.print(f"\n[green]✓ 已选择目录:[/green] {target}")
                self.show_directory_preview(target)
                
                if Confirm.ask("确认使用此目录?", default=True):
                    self.target_dir = target
                    os.chdir(target)
                    
                    # v1.2.0: 初始化跨文件修改器
                    self.cross_file_modifier = CrossFileParameterModifier(str(target))
                    
                    # v1.3.0: 初始化备份管理器
                    self.backup_manager = BackupManager(str(target))
                    
                    # v1.3.1: 显示现有备份列表
                    self._show_existing_backups()
                    
                    # v1.3.0: 加载项目配置
                    project_config = load_config(target)
                    if project_config:
                        self.config = project_config
                    
                    return target
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.console.print(f"[red]❌ 错误: {e}[/red]")
                if attempt == max_attempts - 1:
                    raise
        
        raise NASCLIError(ErrorCode.INVALID_INPUT, "无法获取有效的目录路径")
    
    def _show_existing_backups(self):
        """v1.3.1: 显示现有备份列表"""
        if not self.backup_manager:
            return
        
        backups = self.backup_manager.list_backups_with_info()
        if backups:
            self.console.print(f"\n[yellow]📦 发现 {len(backups)} 个现有备份:[/yellow]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ID", style="cyan", width=10)
            table.add_column("时间", style="dim", width=20)
            table.add_column("描述", style="green")
            table.add_column("文件数", style="yellow", justify="right")
            
            for backup in backups[-5:]:  # 只显示最近5个
                status = "[strikethrough]" if backup['undone'] else ""
                table.add_row(
                    backup['short_id'],
                    backup['formatted_time'],
                    f"{status}{backup['description'][:30]}{status}",
                    str(backup['file_count'])
                )
            
            self.console.print(table)
            
            # 询问是否要切换到某个备份
            if Confirm.ask("\n是否要切换到某个备份版本?", default=False):
                self._handle_backup_switch()
    
    def _handle_backup_switch(self):
        """v1.3.1: 处理备份切换"""
        if not self.backup_manager:
            return
        
        backups = self.backup_manager.list_backups_with_info()
        if not backups:
            return
        
        self.console.print("\n[bold cyan]可用备份列表:[/bold cyan]")
        for i, backup in enumerate(backups, 1):
            status = " (已撤销)" if backup['undone'] else ""
            self.console.print(f"  [{i}] {backup['short_id']} - {backup['formatted_time']} - {backup['description'][:40]}{status}")
        
        try:
            choice = IntPrompt.ask("请选择要恢复的备份序号 (0 取消)", default=0)
            if choice > 0 and choice <= len(backups):
                selected = backups[choice - 1]
                if Confirm.ask(f"确认切换到备份 {selected['short_id']} ?"):
                    # 先备份当前状态，然后切换
                    if self.backup_manager.switch_to_backup(selected['id']):
                        self.console.print(f"[green]✓ 已成功切换到备份 {selected['short_id']}[/green]")
                    else:
                        self.console.print(f"[red]✗ 切换失败[/red]")
            else:
                self.console.print("[dim]已取消切换[/dim]")
        except Exception as e:
            self.console.print(f"[red]切换出错: {e}[/red]")
    
    def show_directory_preview(self, path: Path):
        """显示目录预览"""
        tree = Tree(f"📂 {path.name}")
        
        try:
            items = list(path.iterdir())[:20]
            for item in items:
                if item.is_dir():
                    if not item.name.startswith('.') and item.name not in self.config.analysis.exclude_patterns:
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
            if not any(part.startswith('.') or part in self.config.analysis.exclude_patterns 
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
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
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
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.console.print(f"[red]❌ 错误: {e}[/red]")
        
        if not self.entry_file:
            raise NASCLIError(ErrorCode.INVALID_INPUT, "未选择有效的入口文件")
        
        self.console.print(f"[green]✓ 已选择入口文件:[/green] {self.entry_file}")
        return self.entry_file
    
    def scan_project(self):
        """扫描项目 - v1.3.1 增强版"""
        self.console.print("\n[bold cyan]🔍 步骤 3: 扫描项目架构[/bold cyan]\n")
        
        # v1.3.1: 检查 LLM 可用性，不再使用 Mock 模式
        if not is_llm_available():
            self.console.print("[yellow]⚠️  LLM 客户端未初始化，尝试自动初始化...[/yellow]")
            try:
                init_llm()
            except Exception as e:
                self.console.print(f"[red]❌ LLM 初始化失败: {e}[/red]")
                raise NASCLIError(
                    ErrorCode.LLM_NOT_INITIALIZED,
                    f"LLM 初始化失败: {e}"
                )
        
        self.llm = get_llm_client()
        
        # 发现 Python 文件
        self.console.print("[yellow]📂 发现 Python 文件...[/yellow]")
        py_files = []
        for f in self.target_dir.rglob("*.py"):
            if not any(part.startswith('.') or part in self.config.analysis.exclude_patterns 
                      for part in f.parts):
                # 检查文件大小
                try:
                    if f.stat().st_size > self.config.analysis.max_file_size:
                        self.console.print(f"[dim]  跳过超大文件: {f.name}[/dim]")
                        continue
                    py_files.append(f)
                except:
                    pass
        
        self.console.print(f"[green]✓ 发现 {len(py_files)} 个 Python 文件[/green]\n")
        
        self.scanned_files = [str(f.relative_to(self.target_dir)) for f in py_files]
        
        # v1.2.0: 智能模型发现
        if self.entry_file:
            try:
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
            except Exception as e:
                if self.config.ui.verbose:
                    self.console.print(f"[dim]模型发现失败: {e}[/dim]")
        
        # 分析所有文件
        all_agents = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[yellow]分析文件中...", total=len(py_files))
            
            for f in py_files:
                rel_path = str(f.relative_to(self.target_dir))
                progress.update(task, description=f"[yellow]分析: {rel_path}[/yellow]")
                
                try:
                    agent = ScopeAgent(str(f))
                    if agent.load_file():
                        analysis = agent.analyze()
                        all_agents[rel_path] = agent
                        
                        # 实时打印该文件的候选
                        candidates = analysis.get('nas_candidates', [])
                        if candidates and self.config.ui.verbose:
                            self.console.print(f"[green]  ↳ 发现 {len(candidates)} 个候选[/green]")
                            for cand in candidates[:3]:
                                self.console.print(f"    • [cyan]{cand.get('name')}[/cyan] = [yellow]{cand.get('current_value')}[/yellow]")
                            if len(candidates) > 3:
                                self.console.print(f"    ... 还有 {len(candidates) - 3} 个")
                except Exception as e:
                    if self.config.ui.verbose:
                        self.console.print(f"[dim]  分析失败 {rel_path}: {e}[/dim]")
                
                progress.advance(task)
        
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
            try:
                for cand in agent.get_nas_candidates():
                    # 使用 LLM 生成搜索空间
                    try:
                        search_space = self.llm.generate_search_space(
                            cand['name'],
                            cand['current_value'],
                            cand['type']
                        )
                    except Exception:
                        search_space = [cand['current_value']]
                    
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
            except Exception as e:
                if self.config.ui.verbose:
                    self.console.print(f"[dim]  收集候选失败 {file_path}: {e}[/dim]")
        
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
            try:
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
            except Exception as e:
                if self.config.ui.verbose:
                    self.console.print(f"[dim]LLM 推荐失败: {e}[/dim]")
        
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
                if self.config.ui.verbose:
                    pass
        
        return candidates
    
    def _extract_from_dict(self, data: Dict, file_path: str, 
                           candidates: List, prefix: str = ""):
        """从字典中提取候选参数"""
        nas_keywords = self.config.nas.value_keywords
        layer_keywords = self.config.nas.layer_keywords
        
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
                if any(kw in key.lower() for kw in layer_keywords):
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
                try:
                    start, end = part.split('-')
                    result.update(range(int(start), int(end) + 1))
                except:
                    pass
            else:
                try:
                    result.add(int(part))
                except:
                    pass
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
        
        if not self.config.ui.confirm_before_modify:
            return True
        
        return Confirm.ask("\n确认执行以上修改?", default=True)
    
    def create_backup(self):
        """创建备份 - v1.3.1 使用 BackupManager 并添加描述"""
        self.console.print("\n[bold cyan]💾 创建备份...[/bold cyan]")
        
        if not self.backup_manager:
            self.backup_manager = BackupManager(str(self.target_dir))
        
        try:
            # v1.3.1: 生成详细的备份描述
            selected_count = len([c for c in self.candidates if c.selected])
            description = f"NAS v{__version__} - {selected_count} 个参数 - {self.entry_file or 'unknown'}"
            
            operation = self.backup_manager.create_backup(
                description=description,
                metadata={
                    'version': __version__,
                    'entry_file': self.entry_file,
                    'candidate_count': selected_count,
                    'scan_mode': 'full'
                }
            )
            self.current_operation = operation
            self.console.print(f"[green]✓ 备份已创建: {operation.id}[/green]")
            self.console.print(f"[dim]  描述: {description}[/dim]")
            return operation
        except Exception as e:
            self.console.print(f"[red]✗ 备份创建失败: {e}[/red]")
            if not Confirm.ask("是否继续而不创建备份?", default=False):
                raise
            return None
    
    def apply_modifications(self):
        """应用修改 - v1.3.0 增强版"""
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
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
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
                    if self.config.ui.verbose:
                        self.console.print(f"[green]  ✓ 已修改: {file_path}[/green]")
                else:
                    fail_count += 1
                
                progress.advance(task)
        
        self.console.print(f"\n[green]✓ 修改完成![/green] 成功: {success_count}, 失败: {fail_count}")
        
        if fail_count > 0 and self.backup_manager and self.current_operation:
            if Confirm.ask("部分修改失败，是否撤销所有修改?", default=True):
                self.undo_modifications()
    
    def undo_modifications(self):
        """v1.3.0: 撤销修改"""
        if not self.backup_manager or not self.current_operation:
            self.console.print("[yellow]⚠️  没有可撤销的操作[/yellow]")
            return
        
        self.console.print("\n[bold cyan]↩️  撤销修改...[/bold cyan]")
        
        try:
            success = self.backup_manager.undo(self.current_operation.id)
            if success:
                self.console.print("[green]✓ 修改已撤销[/green]")
            else:
                self.console.print("[red]✗ 撤销失败[/red]")
        except Exception as e:
            self.console.print(f"[red]✗ 撤销出错: {e}[/red]")
    
    def run_search_space_expansion(self):
        """v1.2.0: 运行寻优空间张开"""
        self.console.print("\n[bold cyan]🌐 步骤 7: 寻优空间张开[/bold cyan]\n")
        
        self.search_space_expander = SearchSpaceExpander(self.llm)
        
        try:
            expanded_files = self.search_space_expander.expand_project(str(self.target_dir))
            
            if expanded_files:
                self.console.print(f"[green]✓ 已张开 {len(expanded_files)} 个文件:[/green]")
                for f in expanded_files:
                    self.console.print(f"  • {f}")
            else:
                self.console.print("[dim]未发现需要张开的条件层选择[/dim]")
        except Exception as e:
            if self.config.ui.verbose:
                self.console.print(f"[dim]寻优空间张开失败: {e}[/dim]")
    
    def run_report_injection(self):
        """v1.2.0: 运行 Report 注入"""
        self.console.print("\n[bold cyan]📊 步骤 8: Report 注入[/bold cyan]\n")
        
        if not self.entry_file:
            self.console.print("[yellow]⚠️  未指定入口文件，跳过 report 注入[/yellow]")
            return
        
        try:
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
        except Exception as e:
            if self.config.ui.verbose:
                self.console.print(f"[dim]Report 注入失败: {e}[/dim]")
    
    def _handle_post_completion(self):
        """
        v1.3.1: 处理完成后的流程
        给用户两个选项：
        1. 继续执行 nas-start 命令
        2. 回退到原来的版本
        """
        self.console.print("\n" + "=" * 60)
        self.console.print("[bold green]🎉 NAS 寻优空间注入完成![/bold green]")
        if self.current_operation:
            self.console.print(f"[dim]备份 ID: {self.current_operation.id} (可用于撤销)[/dim]")
        self.console.print("=" * 60)
        
        self.console.print("\n[bold cyan]请选择接下来的操作:[/bold cyan]")
        self.console.print("  [1] 继续执行 nas-start 命令（启动 NAS 训练）")
        self.console.print("  [2] 回退到原来的版本")
        self.console.print("  [3] 退出")
        
        choice = Prompt.ask("请选择", choices=["1", "2", "3"], default="1")
        
        if choice == "1":
            self._run_nas_start()
        elif choice == "2":
            self._rollback()
        else:
            self.console.print("[dim]已退出[/dim]")
    
    def _run_nas_start(self):
        """v1.3.1: 执行 nas-start 命令"""
        self.console.print("\n[bold cyan]🚀 启动 nas-start...[/bold cyan]")
        
        try:
            # 检查 nas-start 是否可用
            result = subprocess.run(
                ["which", "nas-start"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.console.print("[yellow]⚠️  nas-start 命令未找到[/yellow]")
                self.console.print("[dim]请确保 nas-start 已安装并在 PATH 中[/dim]")
                return
            
            # 执行 nas-start
            self.console.print("[dim]执行: nas-start[/dim]")
            subprocess.run(["nas-start"], cwd=self.target_dir)
            
        except Exception as e:
            self.console.print(f"[red]启动 nas-start 失败: {e}[/red]")
    
    def _rollback(self):
        """v1.3.1: 回退到原来的版本"""
        self.console.print("\n[bold cyan]↩️  回退到原版本...[/bold cyan]")
        
        if not self.backup_manager or not self.current_operation:
            self.console.print("[yellow]⚠️  没有可回退的备份[/yellow]")
            return
        
        try:
            success = self.backup_manager.undo(self.current_operation.id)
            if success:
                self.console.print("[green]✓ 已成功回退到原版本[/green]")
            else:
                self.console.print("[red]✗ 回退失败[/red]")
        except Exception as e:
            self.console.print(f"[red]回退出错: {e}[/red]")
    
    def run(self):
        """运行完整流程 v1.3.1"""
        self.show_banner()
        
        if self.target_dir is None:
            self.ask_directory()
        else:
            self.console.print(f"\n[green]✓ 使用指定目录:[/green] {self.target_dir}")
            self.show_directory_preview(self.target_dir)
            
            # v1.3.1: 显示现有备份
            if self.backup_manager:
                self._show_existing_backups()
        
        if self.entry_file is None:
            self.ask_entry_file()
        else:
            self.console.print(f"\n[green]✓ 使用指定入口文件:[/green] {self.entry_file}")
        
        if not Confirm.ask("\n确认开始扫描?", default=True):
            self.console.print("[yellow]已取消[/yellow]")
            return
        
        try:
            self.scan_project()
        except NASCLIError as e:
            self.console.print(f"\n[red]扫描失败: {get_user_friendly_message(e)}[/red]")
            if self.config.ui.verbose:
                self.console.print(f"[dim]详情: {e}[/dim]")
            return
        except Exception as e:
            self.console.print(f"\n[red]扫描出错: {e}[/red]")
            return
        
        if not self.select_candidates():
            self.console.print("[yellow]未选择任何参数，退出[/yellow]")
            return
        
        if not self.show_diff_and_confirm():
            self.console.print("[yellow]已取消修改[/yellow]")
            return
        
        # 创建备份
        backup_op = self.create_backup()
        
        # 应用修改
        try:
            self.apply_modifications()
        except Exception as e:
            self.console.print(f"\n[red]修改失败: {e}[/red]")
            if backup_op and Confirm.ask("是否撤销修改?", default=True):
                self.undo_modifications()
            return
        
        # v1.2.0: 寻优空间张开
        try:
            self.run_search_space_expansion()
        except Exception as e:
            if self.config.ui.verbose:
                self.console.print(f"[dim]寻优空间张开出错: {e}[/dim]")
        
        # v1.2.0: Report 注入
        try:
            self.run_report_injection()
        except Exception as e:
            if self.config.ui.verbose:
                self.console.print(f"[dim]Report 注入出错: {e}[/dim]")
        
        # v1.3.1: 完成后流程
        self._handle_post_completion()


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="NAS-CLI 智能神经网络架构搜索工具 v1.3.1 (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  nas-cli              启动交互式界面
  nas-cli --version    显示版本信息
  nas-cli --dir ./project --entry main.py  指定目录和入口文件
  nas-cli --undo       撤销上次修改
  nas-cli --config     编辑配置文件
  nas-cli --backups    列出所有备份
  
环境变量:
  OPENAI_API_KEY       LLM API Key
  OPENAI_BASE_URL      LLM API URL
  http_proxy           HTTP 代理 (e.g., http://127.0.0.1:7890)
  https_proxy          HTTPS 代理 (e.g., http://127.0.0.1:7890)
  NAS_CLI_VERBOSE      详细输出模式 (1/true/yes)
  NAS_CLI_LANGUAGE     界面语言 (zh/en)
        """
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--dir', '-d', help='目标项目目录')
    parser.add_argument('--entry', '-e', help='入口文件')
    parser.add_argument('--undo', action='store_true', help='撤销上次修改')
    parser.add_argument('--config', action='store_true', help='编辑配置文件')
    parser.add_argument('--backups', '-b', action='store_true', help='列出所有备份')
    parser.add_argument('--switch', '-s', help='切换到指定备份 ID')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    if args.verbose:
        config.ui.verbose = True
    
    # 处理 --config
    if args.config:
        console.print("[bold cyan]编辑配置文件[/bold cyan]")
        config_path = ConfigManager.DEFAULT_CONFIG_FILE
        console.print(f"配置文件路径: {config_path}")
        if not config_path.exists():
            ConfigManager().save_user_config(config)
            console.print(f"[green]已创建默认配置文件[/green]")
        console.print(f"请使用文本编辑器修改: {config_path}")
        return
    
    # 处理 --backups
    if args.backups:
        if args.dir:
            target_dir = Path(args.dir)
            backup_manager = BackupManager(str(target_dir))
            backup_manager.display_backup_list()
        else:
            console.print("[red]请使用 --dir 指定项目目录[/red]")
        return
    
    # 处理 --switch
    if args.switch:
        if args.dir:
            target_dir = Path(args.dir)
            backup_manager = BackupManager(str(target_dir))
            if backup_manager.switch_to_backup(args.switch):
                console.print(f"[green]✓ 已切换到备份 {args.switch[:8]}[/green]")
            else:
                console.print(f"[red]✗ 切换失败[/red]")
        else:
            console.print("[red]请使用 --dir 指定项目目录[/red]")
        return
    
    # 处理 --undo
    if args.undo:
        if args.dir:
            target_dir = Path(args.dir)
            backup_manager = BackupManager(str(target_dir))
            operations = backup_manager.list_operations()
            if operations:
                backup_manager.undo()
            else:
                console.print("[yellow]没有可撤销的操作[/yellow]")
        else:
            console.print("[red]请使用 --dir 指定项目目录[/red]")
        return
    
    # v1.3.1: 初始化 LLM（不再支持 Mock 模式）
    try:
        init_llm()
        if config.ui.verbose:
            console.print("[dim]✓ LLM 客户端初始化成功[/dim]")
    except Exception as e:
        console.print(f"[red]❌ LLM 初始化失败: {e}[/red]")
        console.print("[yellow]请检查 API Key 和代理配置后重试[/yellow]")
        sys.exit(1)
    
    cli = InteractiveNASCLI(config)
    
    if args.dir:
        cli.target_dir = Path(args.dir).expanduser().resolve()
        if not cli.target_dir.exists():
            console.print(f"[red]目录不存在: {cli.target_dir}[/red]")
            sys.exit(1)
        os.chdir(cli.target_dir)
        cli.backup_manager = BackupManager(str(cli.target_dir))
    if args.entry:
        cli.entry_file = args.entry
    
    try:
        cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
        sys.exit(0)
    except NASCLIError as e:
        console.print(f"\n[red]错误: {get_user_friendly_message(e)}[/red]")
        if config.ui.verbose:
            console.print(f"[dim]详情: {e}[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]错误: {e}[/red]")
        if config.ui.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
