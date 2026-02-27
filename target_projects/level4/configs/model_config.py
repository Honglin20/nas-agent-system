"""
Level 4 靶机 - 模型配置
参数分布在 configs 文件夹中
"""

MODEL_CONFIG = {
    "model_type": "advanced",  # 可选: advanced, simple
    "model_name": "TransformerModel",  # 可选: TransformerModel, CNNModel (advanced), SimpleMLP, BasicClassifier (simple)
    
    "model_kwargs": {
        # Transformer 参数
        "input_dim": 50,
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "num_classes": 10,
        
        # 激活函数选择
        "activation": "gelu",  # 可选: relu, gelu, tanh
        
        # 归一化选择
        "norm_type": "layernorm",  # 可选: layernorm, batchnorm
    }
}
