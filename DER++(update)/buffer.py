import random
import torch
import numpy as np
from torchvision import transforms

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
        self.total_seen = 0  # 记录已经处理过的样本总数

    def add_data(self, examples, labels, logits=None):
        if self.examples is None:
            self.examples = torch.zeros(self.buffer_size, *examples.shape[1:], dtype=examples.dtype, device=self.device)
            self.labels = torch.zeros(self.buffer_size, dtype=labels.dtype, device=self.device)
            self.logits = torch.zeros(self.buffer_size, logits.shape[1], dtype=logits.dtype,
                                      device=self.device) if logits is not None else None

        for i in range(examples.shape[0]):
            self.total_seen += 1

            # 应用蓄水池抽样算法
            if self.current_size < self.buffer_size:
                # 如果缓冲区还有空间，直接添加
                index = self.current_size
                self.current_size += 1
            else:
                # 否则，以 buffer_size/total_seen 的概率替换缓冲区中的样本
                if random.random() < self.buffer_size / self.total_seen:
                    # 随机选择要替换的位置
                    index = random.randint(0, self.buffer_size - 1)
                else:
                    # 跳过这个样本
                    continue

            self.examples[index] = examples[i]
            self.labels[index] = labels[i]
            if logits is not None and self.logits is not None:
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