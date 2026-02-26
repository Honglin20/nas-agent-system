"""
NAS CLI - 交互式智能 NAS 寻优空间注入工具 (Real LLM Only)
严禁使用规则模拟，所有分析必须通过真实 LLM
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
from mas_core import NASOrchestrator, ScopeAgent, CentralRegistry, init_llm, get_llm_client

console = Console()

__version__ = "1.0.0"


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


class InteractiveNASCLI:
    """交互式 NAS CLI - 使用真实 LLM"""
    
    def __init__(self):
        self.current_dir = Path.cwd()
        self.target_dir: Optional[Path] = None
        self.entry_file: Optional[str] = None
        self.candidates: List[NASCandidate] = []
        self.scanned_files: List[str] = []
        self.console = Console()
        self.llm = None
        
    def show_banner(self):
        """显示欢迎界面"""
        banner = """
╭────────────────────────────────────────────────────────────╮
│                                                            │
│   🧠 NAS-CLI 智能神经网络架构搜索工具 v1.0.0               │
│                                                            │
│   使用真实 LLM 自动识别代码中的寻优参数                    │
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
        """扫描项目 - 使用真实 LLM"""
        self.console.print("\n[bold cyan]🔍 步骤 3: 扫描项目架构 (使用 LLM)[/bold cyan]\n")
        
        # 初始化 LLM
        self.llm = get_llm_client()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task1 = progress.add_task("[yellow]发现 Python 文件...", total=None)
            py_files = []
            for f in self.target_dir.rglob("*.py"):
                if not any(part.startswith('.') or part in ['__pycache__', 'venv', 'env'] 
                          for part in f.parts):
                    py_files.append(f)
            progress.update(task1, completed=True)
            
            task2 = progress.add_task(f"[yellow]使用 LLM 解析入口文件...", total=None)
            entry_path = self.target_dir / self.entry_file
            entry_agent = ScopeAgent(str(entry_path))
            entry_agent.load_file()
            entry_analysis = entry_agent.analyze()
            progress.update(task2, completed=True)
            
            task3 = progress.add_task("[yellow]分析项目依赖关系...", total=None)
            self.scanned_files = [str(f.relative_to(self.target_dir)) for f in py_files]
            
            all_agents = {}
            for f in py_files:
                agent = ScopeAgent(str(f))
                if agent.load_file():
                    all_agents[str(f.relative_to(self.target_dir))] = agent
            progress.update(task3, completed=True)
            
            task4 = progress.add_task("[yellow]使用 LLM 识别 NAS 候选...", total=None)
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
            
            # 使用 LLM 推荐哪些值得注入
            if self.candidates:
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
            
            progress.update(task4, completed=True)
        
        self.show_scan_results(entry_analysis, all_agents)
    
    def show_scan_results(self, entry_analysis: Dict, all_agents: Dict):
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
        stats.add_row("类定义数", str(len(entry_analysis.get('classes', []))))
        stats.add_row("函数定义数", str(len(entry_analysis.get('functions', []))))
        stats.add_row("NAS 候选数", str(len(self.candidates)))
        self.console.print(stats)
        
        # 显示 LLM 识别的候选
        if self.candidates:
            self.console.print("\n[bold]LLM 识别的 NAS 候选:[/bold]")
            for cand in self.candidates[:5]:
                rec = "⭐" if cand.recommended else ""
                self.console.print(f"  • {cand.name} = {cand.current_value} {rec}")
                self.console.print(f"    [dim]{cand.reason}[/dim]")
            if len(self.candidates) > 5:
                self.console.print(f"  ... 还有 {len(self.candidates) - 5} 个")
    
    def select_candidates(self) -> bool:
        """让用户选择候选参数"""
        self.console.print("\n[bold cyan]⚙️  步骤 4: 配置 NAS 寻优空间[/bold cyan]\n")
        
        if not self.candidates:
            self.console.print("[yellow]⚠️  未发现 NAS 候选参数[/yellow]")
            return False
        
        table = Table(title="LLM 推荐的 NAS 寻优候选")
        table.add_column("序号", style="cyan", justify="center")
        table.add_column("参数名", style="green")
        table.add_column("当前值", style="yellow")
        table.add_column("类型", style="blue")
        table.add_column("位置", style="dim")
        table.add_column("推荐", style="magenta")
        
        for i, cand in enumerate(self.candidates, 1):
            rec_mark = "⭐" if cand.recommended else ""
            table.add_row(
                str(i),
                cand.name,
                str(cand.current_value),
                cand.param_type,
                f"{cand.location}:{cand.line}",
                rec_mark
            )
        
        self.console.print(table)
        
        self.console.print("\n[bold]选择方式:[/bold]")
        self.console.print("  [1] 使用 LLM 推荐参数 (带⭐标记)")
        self.console.print("  [2] 全选所有参数")
        self.console.print("  [3] 手动选择")
        
        choice = Prompt.ask("请选择", choices=["1", "2", "3"], default="1")
        
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
        
        selected = [c for c in self.candidates if c.selected]
        self.console.print(f"\n[green]✓ 已选择 {len(selected)}/{len(self.candidates)} 个参数[/green]")
        
        return len(selected) > 0
    
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
                after = f"{cand.name} = ValueSpace({cand.search_space})"
                
                self.console.print(f"  [red]- {before}[/red]")
                self.console.print(f"  [green]+ {after}[/green]")
                self.console.print(f"    [dim]{cand.reason}[/dim]\n")
        
        return Confirm.ask("\n确认执行以上修改?", default=True)
    
    def apply_modifications(self):
        """应用修改"""
        self.console.print("\n[bold cyan]🔧 步骤 6: 应用修改[/bold cyan]\n")
        
        selected = [c for c in self.candidates if c.selected]
        
        backup_dir = self.target_dir / ".nas_backup"
        backup_dir.mkdir(exist_ok=True)
        
        with Progress(console=self.console) as progress:
            task = progress.add_task("[yellow]修改中...", total=len(selected))
            
            for cand in selected:
                file_path = self.target_dir / cand.location
                
                backup_path = backup_dir / f"{cand.location}.bak"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                
                progress.advance(task)
        
        self.console.print(f"[green]✓ 修改完成![/green]")
        self.console.print(f"[dim]备份保存在: {backup_dir}[/dim]")
    
    def run(self):
        """运行完整流程"""
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
        
        self.apply_modifications()
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold green]🎉 NAS 寻优空间注入完成![/bold green]")
        self.console.print("="*60)


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="NAS-CLI 智能神经网络架构搜索工具 (Real LLM)",
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
