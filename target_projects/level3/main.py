"""
Level 3 靶机：动态反射与 YAML
使用 getattr 动态加载模型，配置完全来自 YAML
"""
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 导入模型模块（用于动态反射）
import models


def load_config(config_path="config.yaml"):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数 - 使用动态反射加载模型"""
    
    # 加载配置
    config = load_config()
    model_config = config['model']
    train_config = config['training']
    
    # 动态反射：通过字符串获取模型类
    model_name = model_config.pop('model_name')  # 例如: 'DynamicModel'
    model_class = getattr(models, model_name)  # 动态获取类
    
    # 动态实例化模型
    model = model_class(**model_config['model_kwargs'])
    
    # 创建模拟数据
    data_config = config['data']
    X = torch.randn(
        data_config['num_samples'], 
        data_config['input_dim']
    )
    y = torch.randint(
        0, 
        data_config['num_classes'], 
        (data_config['num_samples'],)
    )
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True
    )
    
    # 动态选择优化器
    optimizer_name = train_config.pop('optimizer_name')
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['lr'],
            weight_decay=train_config.get('weight_decay', 0)
        )
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config['lr'],
            momentum=train_config.get('momentum', 0),
            weight_decay=train_config.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # 动态选择损失函数
    loss_name = train_config.get('loss', 'CrossEntropyLoss')
    criterion = getattr(nn, loss_name)()
    
    # 训练循环
    num_epochs = train_config['num_epochs']
    
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
