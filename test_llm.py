#!/usr/bin/env python3
"""
使用真实 LLM 测试 NAS-CLI - 三个 Level 靶机
严禁使用规则模拟
"""
import os
import sys
from pathlib import Path

# 设置 LLM API
os.environ['OPENAI_API_KEY'] = 'sk-IA0OXgtva7EmahBVdzkCJgcJxnmo4ja6O0M0M146HniteI3m'
os.environ['OPENAI_BASE_URL'] = 'https://api.moonshot.cn/v1'

sys.path.insert(0, str(Path(__file__).parent))

from mas_core import init_llm, ScopeAgent, get_llm_client
from rich.console import Console
from rich.table import Table

console = Console()

def test_llm_connection():
    """测试 LLM 连接"""
    console.print("[bold cyan]测试 LLM 连接...[/bold cyan]")
    
    try:
        init_llm(
            api_key='sk-IA0OXgtva7EmahBVdzkCJgcJxnmo4ja6O0M0M146HniteI3m',
            base_url='https://api.moonshot.cn/v1'
        )
        console.print("[green]✓ LLM 连接成功[/green]\n")
        return True
    except Exception as e:
        console.print(f"[red]✗ LLM 连接失败: {e}[/red]")
        return False

def test_level(level: int, target_file: str):
    """测试指定 Level"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]🎯 测试 Level {level}: {target_file}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    target_path = Path(__file__).parent / "target_projects" / f"level{level}" / target_file
    
    if not target_path.exists():
        console.print(f"[red]✗ 文件不存在: {target_path}[/red]")
        return False
    
    # 读取代码
    with open(target_path, 'r') as f:
        code = f.read()
    
    # 使用 ScopeAgent 分析（内部使用 LLM）
    agent = ScopeAgent(str(target_path))
    if not agent.load_file():
        console.print("[red]✗ 文件加载失败[/red]")
        return False
    
    analysis = agent.analyze()
    candidates = analysis['nas_candidates']
    
    console.print(f"[green]✓ LLM 识别了 {len(candidates)} 个 NAS 候选[/green]\n")
    
    # 显示候选表格
    table = Table(title=f"Level {level} NAS 候选参数")
    table.add_column("参数名", style="cyan")
    table.add_column("类型", style="green")
    table.add_column("当前值", style="yellow")
    table.add_column("推荐理由", style="dim")
    
    for cand in candidates:
        table.add_row(
            cand.get('name', 'N/A'),
            cand.get('type', 'N/A'),
            str(cand.get('current_value', 'N/A'))[:30],
            cand.get('reason', 'N/A')[:40]
        )
    
    console.print(table)
    
    # 使用 LLM 生成搜索空间
    if candidates:
        console.print("\n[bold]使用 LLM 生成搜索空间...[/bold]")
        llm = get_llm_client()
        
        count = 0
        for cand in candidates[:3]:  # 只显示前3个
            try:
                search_space = llm.generate_search_space(
                    cand['name'],
                    cand['current_value'],
                    cand['type']
                )
                console.print(f"  • {cand['name']}: {search_space}")
                count += 1
            except Exception as e:
                console.print(f"  • {cand['name']}: [red]生成失败 {e}[/red]")
    
    return len(candidates) > 0

def test_dynamic_reflection():
    """测试动态反射解析 (Level 3)"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]🧠 测试动态反射解析 (Level 3)[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    target_path = Path(__file__).parent / "target_projects" / "level3" / "main.py"
    
    with open(target_path, 'r') as f:
        code = f.read()
    
    llm = get_llm_client()
    
    console.print("[bold]使用 LLM 解析动态引用 (getattr)...[/bold]")
    result = llm.resolve_dynamic_reference(code, "model_class")
    
    console.print(f"[green]✓ LLM 解析结果:[/green]")
    console.print(result[:500])
    
    return True

def main():
    console.print("="*60)
    console.print("[bold]NAS-CLI 真实 LLM 测试 - 三个 Level 靶机[/bold]")
    console.print("="*60)
    
    # 测试连接
    if not test_llm_connection():
        return 1
    
    results = {}
    
    # 测试 Level 1
    results[1] = test_level(1, "train.py")
    
    # 测试 Level 2
    results[2] = test_level(2, "main.py")
    
    # 测试 Level 3
    results[3] = test_level(3, "main.py")
    
    # 测试动态反射
    test_dynamic_reflection()
    
    # 汇总
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold cyan]📊 测试汇总[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    for level, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        console.print(f"Level {level}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        console.print("\n[bold green]🎉 所有 Level 测试通过！靶机项目已被 LLM 成功攻克[/bold green]")
    else:
        console.print("\n[bold red]⚠️ 部分测试失败[/bold red]")
    
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
