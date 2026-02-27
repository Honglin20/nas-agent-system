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
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# 导入 MAS 核心
sys.path.insert(0, str(Path(__file__).parent.parent))
from mas_core import NASOrchestrator, ScopeAgent, CentralRegistry, init_llm, get_llm_client, ModifierAgent

console = Console()

__version__ = "1.1.0"


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
        self.modifier_agent = ModifierAgent()
        
    def show_banner(self):
        """显示欢迎界面"""
        banner = """
╭────────────────────────────────────────────────────────────╮
│                                                            │
│   🧠 NAS-CLI 智能神经网络架构搜索工具 v1.1.0               │
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
        """扫描项目 - 使用真实 LLM，带实时打印"""
        self.console.print("\n[bold cyan]🔍 步骤 3: 扫描项目架构 (使用 LLM)[/bold cyan]\n")
        
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
        
        # 分析入口文件
        self.console.print(f"[yellow]🤖 LLM 正在分析入口文件: {self.entry_file}[/yellow]")
        entry_path = self.target_dir / self.entry_file
        entry_agent = ScopeAgent(str(entry_path))
        entry_agent.load_file()
        entry_analysis = entry_agent.analyze()
        
        # 打印 LLM 识别的候选
        entry_candidates = entry_analysis.get('nas_candidates', [])
        if entry_candidates:
            self.console.print(f"[green]✓ LLM 在入口文件发现 {len(entry_candidates)} 个候选:[/green]")
            for cand in entry_candidates:
                self.console.print(f"  • [cyan]{cand.get('name')}[/cyan] = [yellow]{cand.get('current_value')}[/yellow] - [dim]{cand.get('reason', '')[:50]}...[/dim]")
        self.console.print()
        
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
                    for cand in candidates[:3]:  # 只显示前3个
                        self.console.print(f"    • [cyan]{cand.get('name')}[/cyan] = [yellow]{cand.get('current_value')}[/yellow]")
                    if len(candidates) > 3:
                        self.console.print(f"    ... 还有 {len(candidates) - 3} 个")
        
        self.console.print()
        
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
        
        # 使用 LLM 推荐哪些值得注入
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
        """让用户选择候选参数 - 改进的交互"""
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
        table.add_column("LLM推荐", style="magenta")
        
        for i, cand in enumerate(self.candidates, 1):
            rec_mark = "⭐ 推荐" if cand.recommended else ""
            table.add_row(
                str(i),
                cand.name,
                str(cand.current_value),
                cand.param_type,
                f"{cand.location}:{cand.line}",
                rec_mark
            )
        
        self.console.print(table)
        
        # 改进的选择方式
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
        
        # 让用户自定义搜索空间
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
                    # 尝试解析为列表
                    if ',' in custom:
                        values = [v.strip() for v in custom.split(',')]
                        # 尝试转换为数字
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
                        # 单个值
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
                after = f"{cand.name} = ValueSpace({cand.search_space})"
                
                self.console.print(f"  [red]- {before}[/red]")
                self.console.print(f"  [green]+ {after}[/green]")
                self.console.print(f"    [dim]{cand.reason}[/dim]\n")
        
        return Confirm.ask("\n确认执行以上修改?", default=True)
    
    def apply_modifications(self):
        """应用修改 - 修复：真正修改文件"""
        self.console.print("\n[bold cyan]🔧 步骤 6: 应用修改[/bold cyan]\n")
        
        selected = [c for c in self.candidates if c.selected]
        
        # 创建备份目录
        backup_dir = self.target_dir / ".nas_backup"
        backup_dir.mkdir(exist_ok=True)
        
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
                
                # 创建备份
                backup_path = backup_dir / f"{file_path}.bak"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(full_path, backup_path)
                    self.console.print(f"[dim]  📦 已备份: {file_path}[/dim]")
                except Exception as e:
                    self.console.print(f"[red]  ⚠️ 备份失败 {file_path}: {e}[/red]")
                
                # 准备修改列表
                modifications = []
                for cand in cands:
                    if cand.param_type == 'value':
                        mod = {
                            'type': 'value_space',
                            'target': cand.name,
                            'search_space': cand.search_space,
                            'line': cand.line
                        }
                    elif cand.param_type == 'layer':
                        mod = {
                            'type': 'layer_space',
                            'target': cand.name,
                            'layer_options': [str(v) for v in cand.search_space],
                            'line': cand.line
                        }
                    else:
                        continue
                    modifications.append(mod)
                
                # 使用 ModifierAgent 应用修改
                try:
                    result = self.modifier_agent.apply_modifications(
                        str(full_path),
                        modifications
                    )
                    if result:
                        self.console.print(f"[green]  ✓ 已修改: {file_path}[/green]")
                        success_count += 1
                    else:
                        self.console.print(f"[red]  ✗ 修改失败: {file_path}[/red]")
                        fail_count += 1
                except Exception as e:
                    self.console.print(f"[red]  ✗ 修改失败 {file_path}: {e}[/red]")
                    fail_count += 1
                
                progress.advance(task)
        
        self.console.print(f"\n[green]✓ 修改完成![/green] 成功: {success_count}, 失败: {fail_count}")
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
