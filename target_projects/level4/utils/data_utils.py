"""
Level 4 靶机 - 数据加载工具（冗余文件）
"""
import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """虚拟数据集"""
    
    def __init__(self, num_samples=1000, input_dim=50, num_classes=10):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
