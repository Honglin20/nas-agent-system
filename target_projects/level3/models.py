"""
Level 3 靶机：动态反射 - 模型定义模块
"""
import torch
import torch.nn as nn


class DynamicModel(nn.Module):
    """
    完全通过配置动态构建的模型
    所有架构参数都来自外部配置
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 activation='ReLU', dropout_rate=0.2, use_batchnorm=True):
        """
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表，如 [128, 64, 32]
            output_dim: 输出维度
            activation: 激活函数名称字符串
            dropout_rate: Dropout 比率
            use_batchnorm: 是否使用 BatchNorm
        """
        super(DynamicModel, self).__init__()
        
        # 动态获取激活函数
        act_class = getattr(nn, activation)
        
        self.layers = nn.ModuleList()
        
        # 构建层
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(dims[i+1]))
            
            self.layers.append(act_class())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        self.output = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class ResidualBlock(nn.Module):
    """残差块，用于更复杂的架构"""
    
    def __init__(self, dim, activation='ReLU'):
        super(ResidualBlock, self).__init__()
        act_class = getattr(nn, activation)
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            act_class(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = act_class()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class ResidualModel(nn.Module):
    """使用残差连接的模型"""
    
    def __init__(self, input_dim, hidden_dim, num_blocks, output_dim,
                 activation='ReLU', dropout_rate=0.2):
        super(ResidualModel, self).__init__()
        
        act_class = getattr(nn, activation)
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation)
            for _ in range(num_blocks)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.output(x)
