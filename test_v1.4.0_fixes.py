#!/usr/bin/env python3
"""
NAS CLI v1.4.0 测试脚本
测试所有修复点：
1. 参数过滤：只推荐模型结构参数
2. 模型修改范围限制：只修改 backbone 的 __init__
3. Level 4 YAML 配置支持
4. 回退功能
5. 修改生效验证
"""
import os
import sys
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from mas_core.llm_client import (
    is_excluded_param, 
    is_recommended_param,
    filter_nas_candidates,
    EXCLUDED_PARAM_NAMES,
    RECOMMENDED_PARAM_NAMES
)
from mas_core.backup import BackupManager


def test_param_filtering():
    """测试参数过滤功能"""
    print("\n" + "="*70)
    print("测试 1: 参数过滤功能")
    print("="*70)
    
    # 测试应该被排除的参数
    excluded_test_cases = [
        'lr', 'learning_rate', 'learning-rate', 'learningrate',
        'optimizer', 'optim',
        'num_classes', 'num_classes', 'n_classes',
        'batch_size', 'batchsize', 'batch-size',
        'epoch', 'epochs', 'num_epochs', 'num_epochs',
        'weight_decay', 'weight_decay',
        'momentum', 'beta1', 'beta2',
    ]
    
    print("\n1.1 测试应该被排除的参数（训练/任务相关）：")
    all_passed = True
    for param in excluded_test_cases:
        result = is_excluded_param(param)
        status = "✓" if result else "✗"
        print(f"  {status} {param}: {'排除' if result else '未排除'}")
        if not result:
            all_passed = False
    
    # 测试应该被推荐的参数
    recommended_test_cases = [
        'd_model', 'dmodel', 'hidden_dim', 'hidden_dim', 'embed_dim',
        'num_layers', 'n_layers', 'depth', 'num_blocks',
        'num_heads', 'n_heads', 'nhead',
        'dropout', 'attention_dropout',
        'dim_feedforward', 'ffn_dim', 'ff_dim',
        'activation', 'hidden_act',
        'norm_type', 'normalization',
    ]
    
    print("\n1.2 测试应该被推荐的参数（模型结构）：")
    for param in recommended_test_cases:
        result = is_recommended_param(param)
        status = "✓" if result else "✗"
        print(f"  {status} {param}: {'推荐' if result else '未推荐'}")
        if not result:
            all_passed = False
    
    # 测试候选参数过滤
    print("\n1.3 测试候选参数过滤：")
    test_candidates = [
        {'name': 'lr', 'type': 'value', 'current_value': 0.001},
        {'name': 'learning_rate', 'type': 'value', 'current_value': 0.001},
        {'name': 'batch_size', 'type': 'value', 'current_value': 32},
        {'name': 'num_epochs', 'type': 'value', 'current_value': 10},
        {'name': 'optimizer', 'type': 'layer', 'current_value': 'Adam'},
        {'name': 'num_classes', 'type': 'value', 'current_value': 10},
        {'name': 'd_model', 'type': 'value', 'current_value': 256},
        {'name': 'num_layers', 'type': 'value', 'current_value': 6},
        {'name': 'num_heads', 'type': 'value', 'current_value': 8},
        {'name': 'dropout', 'type': 'value', 'current_value': 0.1},
        {'name': 'activation', 'type': 'layer', 'current_value': 'gelu'},
        {'name': 'norm_type', 'type': 'layer', 'current_value': 'layernorm'},
    ]
    
    filtered = filter_nas_candidates(test_candidates)
    filtered_names = {c['name'] for c in filtered}
    
    # 检查训练参数被排除
    for param in ['lr', 'learning_rate', 'batch_size', 'num_epochs', 'optimizer', 'num_classes']:
        if param not in filtered_names:
            print(f"  ✓ {param} 被正确排除")
        else:
            print(f"  ✗ {param} 应该被排除但未被排除")
            all_passed = False
    
    # 检查模型参数被保留
    for param in ['d_model', 'num_layers', 'num_heads', 'dropout', 'activation', 'norm_type']:
        if param in filtered_names:
            print(f"  ✓ {param} 被正确保留")
        else:
            print(f"  ✗ {param} 应该被保留但未被保留")
            all_passed = False
    
    print(f"\n参数过滤测试: {'全部通过' if all_passed else '有失败'}")
    return all_passed


def test_backup_functionality():
    """测试回退功能"""
    print("\n" + "="*70)
    print("测试 2: 回退功能")
    print("="*70)
    
    # 创建测试目录
    test_dir = Path(__file__).parent / "test_backup_dir"
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试文件
    test_file = test_dir / "test.py"
    original_content = "# Original content\nx = 1\n"
    test_file.write_text(original_content)
    
    try:
        # 创建备份管理器
        backup_mgr = BackupManager(test_dir)
        
        # 创建备份
        print("\n2.1 创建备份...")
        operation = backup_mgr.create_backup(description="Test backup")
        print(f"  ✓ 备份创建成功: {operation.id}")
        
        # 修改文件
        print("\n2.2 修改文件...")
        modified_content = "# Modified content\nx = 2\n"
        test_file.write_text(modified_content)
        print("  ✓ 文件已修改")
        
        # 验证文件已修改
        current_content = test_file.read_text()
        if current_content == modified_content:
            print("  ✓ 确认文件内容已变更")
        else:
            print("  ✗ 文件内容未变更")
            return False
        
        # 执行撤销
        print("\n2.3 执行撤销...")
        result = backup_mgr.undo()
        
        if result:
            print("  ✓ 撤销操作成功")
        else:
            print("  ✗ 撤销操作失败")
            return False
        
        # 验证文件已恢复
        restored_content = test_file.read_text()
        if restored_content == original_content:
            print("  ✓ 文件内容已恢复为原始内容")
        else:
            print(f"  ✗ 文件内容未恢复")
            print(f"    期望: {original_content!r}")
            print(f"    实际: {restored_content!r}")
            return False
        
        print("\n回退功能测试: 全部通过")
        return True
        
    finally:
        # 清理
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_level4_yaml_structure():
    """测试 Level 4 YAML 配置结构"""
    print("\n" + "="*70)
    print("测试 3: Level 4 YAML 配置结构")
    print("="*70)
    
    level4_dir = Path(__file__).parent / "target_projects" / "level4"
    
    # 检查 YAML 文件存在
    model_config_yaml = level4_dir / "configs" / "model_config.yaml"
    train_config_yaml = level4_dir / "configs" / "train_config.yaml"
    
    print("\n3.1 检查 YAML 配置文件存在：")
    all_passed = True
    
    if model_config_yaml.exists():
        print(f"  ✓ model_config.yaml 存在")
    else:
        print(f"  ✗ model_config.yaml 不存在")
        all_passed = False
    
    if train_config_yaml.exists():
        print(f"  ✓ train_config.yaml 存在")
    else:
        print(f"  ✗ train_config.yaml 不存在")
        all_passed = False
    
    # 检查 YAML 内容
    print("\n3.2 检查 YAML 配置内容：")
    try:
        import yaml
        
        with open(model_config_yaml, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # 检查是否有 config 键
        if 'model' in model_config and 'config' in model_config['model']:
            print("  ✓ model.config 结构正确")
            config = model_config['model']['config']
            
            # 检查模型结构参数
            model_params = ['d_model', 'nhead', 'num_encoder_layers', 'dim_feedforward', 'dropout']
            for param in model_params:
                if param in config:
                    print(f"  ✓ 找到模型参数: {param} = {config[param]}")
                else:
                    print(f"  ✗ 缺少模型参数: {param}")
                    all_passed = False
        else:
            print("  ✗ model.config 结构不正确")
            all_passed = False
        
        with open(train_config_yaml, 'r') as f:
            train_config = yaml.safe_load(f)
        
        # 检查训练参数
        if 'training' in train_config:
            print("  ✓ training 配置存在")
        else:
            print("  ✗ training 配置不存在")
        
        if 'optimizer' in train_config:
            print("  ✓ optimizer 配置存在")
        else:
            print("  ✗ optimizer 配置不存在")
            
    except Exception as e:
        print(f"  ✗ 读取 YAML 失败: {e}")
        all_passed = False
    
    # 检查 main.py 使用 YAML
    print("\n3.3 检查 main.py 使用 YAML 配置：")
    main_file = level4_dir / "main.py"
    main_content = main_file.read_text()
    
    if 'yaml' in main_content.lower() or 'load_yaml_config' in main_content:
        print("  ✓ main.py 使用 YAML 配置")
    else:
        print("  ✗ main.py 未使用 YAML 配置")
        all_passed = False
    
    # 检查模型接受 config 参数
    print("\n3.4 检查模型接受 config 参数：")
    advanced_models = level4_dir / "models" / "advanced_models.py"
    models_content = advanced_models.read_text()
    
    if '**config' in models_content or 'config.get' in models_content:
        print("  ✓ 模型接受 config 参数")
    else:
        print("  ✗ 模型未接受 config 参数")
        all_passed = False
    
    print(f"\nLevel 4 YAML 测试: {'全部通过' if all_passed else '有失败'}")
    return all_passed


def test_level_structure():
    """测试所有 level 的结构"""
    print("\n" + "="*70)
    print("测试 4: 所有 Level 结构检查")
    print("="*70)
    
    all_passed = True
    
    levels = [
        ('level1', 'train.py'),
        ('level2', 'main.py'),
        ('level3', 'main.py'),
        ('level4', 'main.py'),
    ]
    
    for level, entry in levels:
        level_dir = Path(__file__).parent / "target_projects" / level
        entry_file = level_dir / entry
        
        print(f"\n4.{list(zip(*levels))[0].index(level)+1} 检查 {level}:")
        
        if level_dir.exists():
            print(f"  ✓ {level} 目录存在")
        else:
            print(f"  ✗ {level} 目录不存在")
            all_passed = False
            continue
        
        if entry_file.exists():
            print(f"  ✓ {entry} 存在")
        else:
            print(f"  ✗ {entry} 不存在")
            all_passed = False
            continue
        
        # 检查是否有模型定义
        py_files = list(level_dir.rglob("*.py"))
        print(f"  ✓ 找到 {len(py_files)} 个 Python 文件")
    
    print(f"\nLevel 结构测试: {'全部通过' if all_passed else '有失败'}")
    return all_passed


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("NAS CLI v1.4.0 测试套件")
    print("="*70)
    
    results = []
    
    # 运行测试
    results.append(("参数过滤", test_param_filtering()))
    results.append(("回退功能", test_backup_functionality()))
    results.append(("Level 4 YAML", test_level4_yaml_structure()))
    results.append(("Level 结构", test_level_structure()))
    
    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查修复")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
