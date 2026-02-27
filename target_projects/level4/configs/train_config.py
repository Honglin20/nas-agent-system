"""
Level 4 靶机 - 训练配置
参数分布在 configs 文件夹中
"""

TRAIN_CONFIG = {
    # 数据配置
    "data": {
        "num_samples": 5000,
        "input_dim": 50,
        "num_classes": 10,
    },
    
    # 训练超参数
    "batch_size": 128,
    "num_epochs": 50,
    
    # 优化器配置
    "optimizer": {
        "name": "Adam",  # 可选: Adam, SGD
        "lr": 0.001,  # 学习率
        "weight_decay": 1e-4,
        "beta1": 0.9,  # Adam 参数
        "beta2": 0.999,  # Adam 参数
        "momentum": 0.9,  # SGD 参数
    },
    
    # 学习率调度
    "lr_scheduler": {
        "type": "step",  # 可选: step, cosine, plateau
        "step_size": 10,
        "gamma": 0.5,
    },
    
    # 损失函数
    "loss": "CrossEntropyLoss",  # 可选: CrossEntropyLoss, MSELoss
    
    # 早停配置
    "early_stopping": {
        "enabled": True,
        "patience": 5,
        "min_delta": 0.001,
    },
}
