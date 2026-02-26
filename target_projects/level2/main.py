"""
Level 2 靶机：跨文件静态传参
main.py 通过字典定义参数，传递给 model.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 从 model.py 导入模型
from model import ConfigurableModel


def main():
    """主函数，通过字典配置参数"""
    
    # 在 main.py 中定义的配置字典
    model_config = {
        "input_dim": 10,
        "d_model": 64,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout_rate": 0.3,
        "activation": "relu",  # 可选: relu, sigmoid, tanh
        "output_dim": 2
    }
    
    training_config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "num_epochs": 20,
        "weight_decay": 1e-5,
        "optimizer": "adam",  # 可选: adam, sgd
        "momentum": 0.9  # 仅用于 SGD
    }
    
    # 创建模拟数据
    X = torch.randn(2000, model_config["input_dim"])
    y = torch.randint(0, model_config["output_dim"], (2000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset, 
        batch_size=training_config["batch_size"], 
        shuffle=True
    )
    
    # 实例化模型 - 参数从 main.py 传递
    model = ConfigurableModel(**model_config)
    
    # 根据配置选择优化器
    if training_config["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"]
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config["learning_rate"],
            momentum=training_config["momentum"],
            weight_decay=training_config["weight_decay"]
        )
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(training_config["num_epochs"]):
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
        
        print(f"Epoch [{epoch+1}/{training_config['num_epochs']}], "
              f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
