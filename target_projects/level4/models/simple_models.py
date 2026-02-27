"""
Level 4 靶机 - 简单模型（不会被实例化的模型）
这些模型虽然存在，但根据配置不会被使用
"""
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    简单 MLP - 不会被实例化
    存在作为干扰项
    """
    
    def __init__(self, input_dim=50, hidden_dims=[128, 64], 
                 output_dim=10, activation='relu', dropout=0.2):
        super(SimpleMLP, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # 条件代码：根据 activation 参数选择激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class BasicClassifier(nn.Module):
    """
    基础分类器 - 不会被实例化
    存在作为干扰项
    """
    
    def __init__(self, input_dim=50, hidden_dim=128, num_classes=10):
        super(BasicClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
