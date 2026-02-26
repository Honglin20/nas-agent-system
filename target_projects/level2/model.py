"""
Level 2 靶机：跨文件静态传参 - 模型定义
"""
import torch
import torch.nn as nn


class ConfigurableModel(nn.Module):
    """可配置的神经网络模型"""
    
    def __init__(self, input_dim=10, d_model=64, hidden_dim=128, 
                 num_layers=2, dropout_rate=0.2, activation="relu", output_dim=2):
        super(ConfigurableModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # 根据字符串选择激活函数
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "sigmoid":
            act_layer = nn.Sigmoid()
        elif activation == "tanh":
            act_layer = nn.Tanh()
        else:
            act_layer = nn.ReLU()
        
        # 第一层
        self.layers.append(nn.Linear(input_dim, d_model))
        self.layers.append(act_layer)
        self.layers.append(nn.Dropout(dropout_rate))
        
        # 中间层
        current_dim = d_model
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(act_layer)
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # 输出层
        self.output = nn.Linear(current_dim, output_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
