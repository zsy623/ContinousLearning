import random
import torch
import numpy as np
from torchvision import transforms  # 导入transforms

# 定义一个只包含标准化的transform
transform_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


class Buffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.examples = None
        self.labels = None
        self.logits = None  # 存储logits用于DER
        self.current_size = 0

    def add_data(self, examples, labels, logits=None):
        if self.examples is None:
            self.examples = torch.zeros(self.buffer_size, *examples.shape[1:], dtype=examples.dtype, device=self.device)
            self.labels = torch.zeros(self.buffer_size, dtype=labels.dtype, device=self.device)
            self.logits = torch.zeros(self.buffer_size, logits.shape[1], dtype=logits.dtype, device=self.device)

        for i in range(examples.shape[0]):
            if self.current_size < self.buffer_size:
                index = self.current_size
                self.current_size += 1
            else:
                index = random.randint(0, self.buffer_size - 1)

            self.examples[index] = examples[i]
            self.labels[index] = labels[i]
            if logits is not None:
                self.logits[index] = logits[i]

    def get_data(self, size: int):
        if size > min(self.examples.shape[0], self.current_size):
            size = min(self.examples.shape[0], self.current_size)

        indexes = np.random.choice(self.current_size, size=size, replace=False)

        # 直接返回tensor，不再应用transform
        # 因为数据在add_data之前已经被transform过了
        examples = self.examples[indexes]
        labels = self.labels[indexes]
        logits = self.logits[indexes] if self.logits is not None else None

        return examples, labels, logits