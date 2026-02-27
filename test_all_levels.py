#!/usr/bin/env python3
"""
NAS CLI v1.2.0 测试脚本
测试所有 4 个 level 的靶机
"""
import os
import sys
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from mas_core import NASOrchestrator, init_llm


def test_level(level_name: str, project_path: str, entry_file: str):
    """测试单个 level"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing {level_name}")
    print(f"{'='*70}")
    print(f"Project: {project_path}")
    print(f"Entry: {entry_file}")
    
    # 清理之前的备份
    backup_dir = Path(project_path) / ".nas_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
        print(f"✓ Cleaned up old backup")
    
    # 创建 orchestrator
    orchestrator = NASOrchestrator(project_path, entry_file)
    
    # 运行扫描
    files = orchestrator.scan_project()
    print(f"✓ Found {len(files)} Python files")
    
    # 创建 Scope Agents
    orchestrator.create_scope_agents(files)
    print(f"✓ Created {len(orchestrator.scope_agents)} Scope Agents")
    
    # 运行模型发现
    discovery_result = orchestrator.run_model_discovery()
    if discovery_result:
        print(f"✓ Model discovery completed")
        if discovery_result.get('instantiated_model'):
            print(f"  - Instantiated model: {discovery_result['instantiated_model'].get('instantiated_model')}")
    
    # 收集候选参数
    candidates = orchestrator.collect_nas_candidates()
    print(f"✓ Found {len(candidates)} NAS candidates:")
    for cand in candidates[:5]:
        print(f"  - {cand['name']} ({cand['type']}) = {cand['current_value']}")
    if len(candidates) > 5:
        print(f"  ... and {len(candidates) - 5} more")
    
    # 创建备份
    orchestrator.create_backup()
    print(f"✓ Backup created at: {backup_dir}")
    
    # 验证备份
    if backup_dir.exists():
        backup_files = list(backup_dir.rglob("*.py"))
        print(f"✓ Backup contains {len(backup_files)} files")
    
    print(f"\n✅ {level_name} test completed successfully!")
    return True


def main():
    """运行所有 level 的测试"""
    print("🚀 NAS CLI v1.2.0 - Level Testing")
    print("="*70)
    
    # 初始化 LLM
    api_key = os.getenv('OPENAI_API_KEY', 'sk-IA0OXgtva7EmahBVdzkCJgcJxnmo4ja6O0M0M146HniteI3m')
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.moonshot.cn/v1')
    
    try:
        init_llm(api_key, base_url)
        print("✓ LLM client initialized")
    except Exception as e:
        print(f"✗ LLM initialization failed: {e}")
        return 1
    
    base_path = Path(__file__).parent / "target_projects"
    
    # 测试 Level 1
    test_level(
        "Level 1 - Static Single File",
        str(base_path / "level1"),
        "train.py"
    )
    
    # 测试 Level 2
    test_level(
        "Level 2 - Cross-file Static",
        str(base_path / "level2"),
        "main.py"
    )
    
    # 测试 Level 3
    test_level(
        "Level 3 - Dynamic Reflection + YAML",
        str(base_path / "level3"),
        "main.py"
    )
    
    # 测试 Level 4
    test_level(
        "Level 4 - Complex Project Structure",
        str(base_path / "level4"),
        "main.py"
    )
    
    print(f"\n{'='*70}")
    print("🎉 All levels tested successfully!")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
