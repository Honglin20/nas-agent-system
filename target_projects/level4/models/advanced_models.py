"""
Level 4 靶机 - 高级模型（实际被实例化的模型）
包含复杂的架构参数
"""
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Transformer 模型 - 实际被实例化的模型之一
    包含复杂的架构参数
    """
    
    def __init__(self, input_dim=50, d_model=256, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, num_classes=10, activation='gelu',
                 norm_type='layernorm'):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 根据 activation 参数选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # 根据 norm_type 选择归一化层
        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(d_model)
        else:
            self.norm = nn.Identity()
        
        # 输出层
        self.output = nn.Linear(d_model, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        # x shape: (batch, input_dim)
        x = self.input_projection(x)  # (batch, d_model)
        x = x.unsqueeze(1)  # (batch, 1, d_model) - 添加序列维度
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # (batch, d_model)
        
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x)
        else:
            x = self.norm(x)
        
        x = self.output(x)
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CNNModel(nn.Module):
    """
    CNN 模型 - 另一个可能被实例化的模型
    不会被实际使用（根据配置）
    """
    
    def __init__(self, input_dim=50, num_channels=[64, 128, 256], 
                 kernel_sizes=[3, 3, 3], dropout=0.3, num_classes=10):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        
        in_channels = 1  # 将输入视为 1D 信号
        for out_channels, kernel_size in zip(num_channels, kernel_sizes):
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            )
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm1d(out_channels))
            self.conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        # 计算输出维度
        self.fc_input_dim = num_channels[-1] * input_dim
        self.fc = nn.Linear(self.fc_input_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        for layer in self.conv_layers:
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
