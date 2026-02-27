"""
Test Runner for NAS Agent System
Phase 3: 真实闭环测试
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mas_core import (
    NASOrchestrator, 
    ScopeAgent, 
    CentralRegistry,
    init_llm,
    get_llm_client
)


def run_level_test(level: int, use_llm: bool = True):
    """
    运行指定 Level 的靶机测试
    
    Args:
        level: 1, 2, or 3
        use_llm: 是否使用真实 LLM
    """
    print("\n" + "="*80)
    print(f"🎯 PHASE 3 TEST: Level {level} Target Project")
    print("="*80)
    
    # 确定靶机路径
    target_path = Path(__file__).parent.parent / "target_projects" / f"level{level}"
    
    if not target_path.exists():
        print(f"❌ Target project not found: {target_path}")
        return False
    
    print(f"\n📁 Target: {target_path}")
    print(f"🤖 LLM Enabled: {use_llm}")
    
    # 清理之前的注册
    registry = CentralRegistry()
    # 重置单例
    CentralRegistry._instance = None
    registry = CentralRegistry()
    
    try:
        # 创建协调器
        orchestrator = NASOrchestrator(str(target_path))
        
        # Phase 1: 扫描
        files = orchestrator.scan_project()
        print(f"\n✓ Found {len(files)} Python files")
        for f in files:
            print(f"  - {Path(f).name}")
        
        # Phase 2: 创建 Agents
        orchestrator.create_scope_agents(files)
        print(f"\n✓ Created {len(orchestrator.scope_agents)} Scope Agents")
        
        # 打印 Agent 日志
        print("\n📋 Agent Analysis Logs:")
        print("-"*80)
        for agent_id, agent in orchestrator.scope_agents.items():
            print(f"\n🔹 Agent: {agent_id}")
            print(f"   File: {agent.file_path}")
            for log in agent.get_cot()[:5]:  # 只显示前5条思考
                print(f"   {log}")
        
        # Phase 3: P2P 解析
        orchestrator.run_p2p_resolution()
        
        # Phase 4: 收集候选
        candidates = orchestrator.collect_nas_candidates()
        print(f"\n✓ Found {len(candidates)} NAS candidates")
        
        for cand in candidates:
            print(f"  - {cand['name']} ({cand['type']}): {cand['current_value']}")
        
        # 注册表摘要
        print("\n📊 Registry Summary:")
        registry.print_summary()
        
        # Phase 5-7: 交互和修改（模拟）
        if candidates:
            print("\n⚠️  Skipping actual modification (dry run mode)")
            print("   Candidates that would be injected:")
            for cand in candidates:
                print(f"   • {cand['name']} -> {cand['suggestion']}")
        
        print("\n" + "="*80)
        print(f"✅ Level {level} Test PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ Level {level} Test FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有 Level 的测试"""
    print("\n" + "="*80)
    print("🚀 RUNNING ALL LEVEL TESTS")
    print("="*80)
    
    results = {}
    
    for level in [1, 2, 3]:
        results[level] = run_level_test(level, use_llm=False)
        print("\n" + "-"*80)
    
    # 汇总
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for level, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"Level {level}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("🎉 ALL TESTS PASSED!" if all_passed else "⚠️  SOME TESTS FAILED"))
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NAS Agent Test Runner")
    parser.add_argument('--level', type=int, choices=[1, 2, 3], 
                       help='Run specific level test')
    parser.add_argument('--all', action='store_true',
                       help='Run all level tests')
    parser.add_argument('--no-llm', action='store_true',
                       help='Run without LLM')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_tests()
    elif args.level:
        run_level_test(args.level, use_llm=not args.no_llm)
    else:
        parser.print_help()
