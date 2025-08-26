import torch
from torch import nn
from torchvision import transforms

class Args():
    def __init__(self):
        self.buffer_size = 10000
        self.get_size = int(self.buffer_size / 10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.CrossEntropyLoss()
        self.lr = 0.001
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # 专门用于缓冲区采样的转换，只包含标准化
        self.transform_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.alpha = 0.5  # DER蒸馏损失权重
        self.beta = 1.0   # DER++分类损失权重
        self.num_tasks = 5  # CIFAR-10分为5个任务
        self.num_classes = 10  # CIFAR-10总类别数
        self.epochs_per_task = 1
        self.batch_size = 32

args = Args()
