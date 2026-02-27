"""
Level 4 靶机：复杂项目结构 + YAML 配置 (v1.4.0)
- 多个模型文件，但 getattr 只获取其中一个
- 模型参数通过 YAML 配置文件管理
- 模型只接受 config 参数
- 通过 config.d_model 方式获取参数
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 动态反射导入模型
import models.advanced_models as adv_models
import models.simple_models as simple_models


def load_yaml_config(config_path: Path) -> dict:
    """v1.4.0: 从 YAML 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数 - 复杂项目结构 + YAML 配置"""
    
    # v1.4.0: 从 YAML 文件加载配置
    config_dir = Path(__file__).parent / "configs"
    
    train_config = load_yaml_config(config_dir / "train_config.yaml")
    model_config = load_yaml_config(config_dir / "model_config.yaml")
    
    # 动态反射：根据配置选择模型类
    # 注意：虽然有两个模型模块，但只会实例化其中一个
    model_type = model_config.get('model_type', 'advanced')
    model_name = model_config.get('model_name', 'TransformerModel')
    
    if model_type == 'advanced':
        model_class = getattr(adv_models, model_name)
    else:
        model_class = getattr(simple_models, model_name)
    
    # v1.4.0: 模型只接受 config 参数
    # 通过 config.d_model 方式获取参数
    model_kwargs = model_config.get('config', {})
    model = model_class(**model_kwargs)
    
    # 创建数据
    data_config = train_config.get('data', {})
    X = torch.randn(
        data_config.get('num_samples', 1000),
        data_config.get('input_dim', 50)
    )
    y = torch.randint(
        0,
        data_config.get('num_classes', 10),
        (data_config.get('num_samples', 1000),)
    )
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.get('training', {}).get('batch_size', 64),
        shuffle=True
    )
    
    # 动态选择优化器
    optimizer_config = train_config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'Adam')
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            weight_decay=optimizer_config.get('weight_decay', 1e-4),
            betas=(
                optimizer_config.get('beta1', 0.9),
                optimizer_config.get('beta2', 0.999)
            )
        )
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # 动态选择损失函数
    loss_name = train_config.get('loss', 'CrossEntropyLoss')
    criterion = getattr(nn, loss_name)()
    
    # 训练循环
    num_epochs = train_config.get('training', {}).get('num_epochs', 20)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
