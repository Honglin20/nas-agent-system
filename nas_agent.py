#!/usr/bin/env python3
"""
NAS Agent System - CLI Entry Point
智能 NAS 注入 CLI 工具 v1.3.0

Usage:
    nas-agent inject <project_path> [--api-key KEY] [--base-url URL]
    nas-agent analyze <project_path>
    nas-agent test --level {1,2,3}
    nas-agent config
    nas-agent undo <project_path>
"""
import sys
import argparse
from pathlib import Path

# 添加 mas_core 到路径
sys.path.insert(0, str(Path(__file__).parent))

from mas_core import (
    NASOrchestrator, init_llm, get_config, load_config, 
    ConfigManager, BackupManager, __version__
)


def cmd_inject(args):
    """执行 NAS 注入命令"""
    project_path = args.project_path
    
    # 加载配置
    config = load_config(Path(project_path))
    
    # 初始化 LLM
    api_key = args.api_key or config.llm.api_key
    base_url = args.base_url or config.llm.base_url
    
    if not api_key:
        print("错误: 未设置 API Key。请设置 OPENAI_API_KEY 环境变量或使用 --api-key")
        sys.exit(1)
    
    print(f"Initializing LLM client...")
    print(f"API URL: {base_url}")
    
    try:
        init_llm(api_key, base_url)
    except Exception as e:
        print(f"LLM 初始化失败: {e}")
        sys.exit(1)
    
    # 创建协调器并运行
    orchestrator = NASOrchestrator(project_path)
    orchestrator.run()


def cmd_analyze(args):
    """执行代码分析命令"""
    from mas_core import ScopeAgent
    
    project_path = Path(args.project_path)
    
    print(f"Analyzing project: {project_path}")
    print("="*70)
    
    # 加载配置以获取排除模式
    config = load_config(project_path)
    
    for py_file in project_path.rglob("*.py"):
        if any(part.startswith('.') or part in config.analysis.exclude_patterns 
               for part in py_file.parts):
            continue
        
        print(f"\n📄 {py_file.relative_to(project_path)}")
        print("-"*70)
        
        try:
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
                    print(f"    - {cand['name']}: {cand['current_value']} -> {cand.get('suggestion', 'N/A')}")
        except Exception as e:
            print(f"  分析失败: {e}")


def cmd_test(args):
    """运行靶机测试"""
    from tests.test_runner import run_level_test
    
    level = args.level
    use_llm = not args.no_llm
    
    # 加载配置
    config = load_config()
    
    # 初始化 LLM
    if use_llm:
        api_key = config.llm.api_key
        base_url = config.llm.base_url
        
        if not api_key:
            print("警告: 未设置 API Key，将使用 Mock 模式")
            init_llm(use_mock=True)
        else:
            try:
                init_llm(api_key, base_url)
            except Exception as e:
                print(f"LLM 初始化失败: {e}，将使用 Mock 模式")
                init_llm(use_mock=True)
    else:
        init_llm(use_mock=True)
    
    run_level_test(level, use_llm=use_llm)


def cmd_config(args):
    """管理配置"""
    config_manager = ConfigManager()
    
    if args.show:
        config = get_config()
        print("当前配置:")
        print(f"  LLM Base URL: {config.llm.base_url}")
        print(f"  LLM Models: {config.llm.models}")
        print(f"  LLM Timeout: {config.llm.timeout}s")
        print(f"  UI Language: {config.ui.language}")
        print(f"  Auto Backup: {config.ui.auto_backup}")
        return
    
    if args.edit:
        config_path = ConfigManager.DEFAULT_CONFIG_FILE
        print(f"配置文件路径: {config_path}")
        if not config_path.exists():
            config = Config()
            config_manager.save_user_config(config)
            print("已创建默认配置文件")
        print(f"请使用文本编辑器修改: {config_path}")
        return
    
    if args.reset:
        config = Config()
        if config_manager.save_user_config(config):
            print("配置已重置为默认值")
        else:
            print("重置配置失败")
        return


def cmd_undo(args):
    """撤销修改"""
    project_path = Path(args.project_path)
    
    if not project_path.exists():
        print(f"错误: 项目路径不存在: {project_path}")
        sys.exit(1)
    
    backup_manager = BackupManager(str(project_path))
    operations = backup_manager.list_operations()
    
    if not operations:
        print("没有可撤销的操作")
        return
    
    print(f"找到 {len(operations)} 个操作记录:")
    for i, op in enumerate(operations[-5:], 1):  # 显示最近5个
        from datetime import datetime
        timestamp = datetime.fromtimestamp(op.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        undone = " (已撤销)" if op.metadata.get('undone') else ""
        print(f"  {i}. [{timestamp}] {op.description}{undone}")
    
    if args.operation_id:
        success = backup_manager.undo(args.operation_id)
    else:
        success = backup_manager.undo()
    
    if success:
        print("撤销成功")
    else:
        print("撤销失败")


def main():
    parser = argparse.ArgumentParser(
        description=f"NAS Agent System - Intelligent NAS Injection CLI v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nas-agent inject ./my_project
  nas-agent analyze ./my_project
  nas-agent test --level 1
  nas-agent config --show
  nas-agent undo ./my_project
        """
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
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
    
    # config 命令
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--edit', action='store_true', help='Edit configuration file')
    config_parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    config_parser.set_defaults(func=cmd_config)
    
    # undo 命令
    undo_parser = subparsers.add_parser('undo', help='Undo last modification')
    undo_parser.add_argument('project_path', help='Path to the project')
    undo_parser.add_argument('--operation-id', help='Specific operation ID to undo')
    undo_parser.set_defaults(func=cmd_undo)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
