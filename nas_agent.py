#!/usr/bin/env python3
"""
NAS Agent System - CLI Entry Point
智能 NAS 注入 CLI 工具

Usage:
    nas-agent inject <project_path> [--api-key KEY] [--base-url URL]
    nas-agent analyze <project_path>
    nas-agent test --level {1,2,3}
"""
import sys
import argparse
from pathlib import Path

# 添加 mas_core 到路径
sys.path.insert(0, str(Path(__file__).parent))

from mas_core import NASOrchestrator, init_llm


def cmd_inject(args):
    """执行 NAS 注入命令"""
    project_path = args.project_path
    
    # 初始化 LLM
    api_key = args.api_key or "sk-IA0OXgtva7EmahBVdzkCJgcJxnmo4ja6O0M0M146HniteI3m"
    base_url = args.base_url or "https://api.moonshot.cn/v1"
    
    print(f"Initializing LLM client...")
    print(f"API URL: {base_url}")
    init_llm(api_key, base_url)
    
    # 创建协调器并运行
    orchestrator = NASOrchestrator(project_path)
    orchestrator.run()


def cmd_analyze(args):
    """执行代码分析命令"""
    from mas_core import ScopeAgent
    
    project_path = Path(args.project_path)
    
    print(f"Analyzing project: {project_path}")
    print("="*70)
    
    for py_file in project_path.rglob("*.py"):
        if any(part.startswith('.') for part in py_file.parts):
            continue
        
        print(f"\n📄 {py_file.relative_to(project_path)}")
        print("-"*70)
        
        agent = ScopeAgent(str(py_file))
        if agent.load_file():
            analysis = agent.analyze()
            
            print(f"  Classes: {len(analysis['classes'])}")
            for cls in analysis['classes']:
                print(f"    - {cls['name']} (line {cls['line']})")
            
            print(f"  Functions: {len(analysis['functions'])}")
            for func in analysis['functions'][:5]:  # 只显示前5个
                print(f"    - {func['name']} (line {func['line']})")
            
            print(f"  NAS Candidates: {len(analysis['nas_candidates'])}")
            for cand in analysis['nas_candidates'][:3]:  # 只显示前3个
                print(f"    - {cand['name']}: {cand['current_value']} -> {cand['suggestion']}")


def cmd_test(args):
    """运行靶机测试"""
    from tests.test_runner import run_level_test
    
    level = args.level
    use_llm = not args.no_llm
    
    # 初始化 LLM
    if use_llm:
        api_key = "sk-IA0OXgtva7EmahBVdzkCJgcJxnmo4ja6O0M0M146HniteI3m"
        base_url = "https://api.moonshot.cn/v1"
        init_llm(api_key, base_url)
    
    run_level_test(level, use_llm=use_llm)


def main():
    parser = argparse.ArgumentParser(
        description="NAS Agent System - Intelligent NAS Injection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nas-agent inject ./my_project
  nas-agent analyze ./my_project
  nas-agent test --level 1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # inject 命令
    inject_parser = subparsers.add_parser('inject', help='Inject NAS search spaces into project')
    inject_parser.add_argument('project_path', help='Path to the project')
    inject_parser.add_argument('--api-key', help='OpenAI API Key')
    inject_parser.add_argument('--base-url', help='API Base URL')
    inject_parser.set_defaults(func=cmd_inject)
    
    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='Analyze project for NAS candidates')
    analyze_parser.add_argument('project_path', help='Path to the project')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # test 命令
    test_parser = subparsers.add_parser('test', help='Run tests on target projects')
    test_parser.add_argument('--level', type=int, choices=[1, 2, 3], required=True,
                            help='Target project level to test')
    test_parser.add_argument('--no-llm', action='store_true',
                            help='Run without LLM (mock mode)')
    test_parser.set_defaults(func=cmd_test)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
