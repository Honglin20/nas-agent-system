from archmind import ValueSpace
"""
Level 1 靶机：静态单文件
所有配置硬编码在 train.py 中
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(nn.Module):
    """简单的神经网络，所有参数硬编码"""
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 硬编码的维度参数
        d_model = 64
        hidden_dim = 128
        num_layers = 2
        dropout_rate = 0.2
        
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(nn.Linear(10, d_model))
        self.layers.append(nn.ReLU())  # 硬编码激活函数
        self.layers.append(nn.Dropout(dropout_rate))
        
        # 中间层
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(d_model, hidden_dim))
            self.layers.append(nn.ReLU())  # 硬编码激活函数
            self.layers.append(nn.Dropout(dropout_rate))
            d_model = hidden_dim
        
        # 输出层
        self.output = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def train():
    """训练函数，所有超参数硬编码"""
    # 硬编码的训练参数
    batch_size = ValueSpace([16, 32, 64, 128])
    learning_rate = ValueSpace([0.0001, 0.001, 0.01])
    num_epochs = 10
    weight_decay = 1e-4
    
    # 创建模拟数据
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 模型、损失函数、优化器
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 训练循环
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("Training completed!")


if __name__ == "__main__":
    train()
