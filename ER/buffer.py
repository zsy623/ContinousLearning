import torch
import numpy as np

# 经验回放缓冲区
class buffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.examples = None
        self.labels = None
        self.task_labels = None

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def is_empty(self) -> bool:
        return self.num_seen_examples == 0

    def add_data(self, examples, labels=None, task_labels=None):
        if self.examples is None:
            self.examples = torch.zeros((self.buffer_size, *examples.shape[1:]), dtype=examples.dtype,device=self.device)
            self.labels = torch.zeros(self.buffer_size, dtype=labels.dtype, device=self.device)
            self.task_labels = torch.zeros(self.buffer_size, dtype=torch.long, device=self.device)

        for i in range(examples.shape[0]):
            index = self.num_seen_examples % self.buffer_size
            self.examples[index] = examples[i]
            if labels is not None:
                self.labels[index] = labels[i]
            if task_labels is not None:
                self.task_labels[index] = task_labels[i]
            self.num_seen_examples += 1

    def get_data(self, size: int, transform=None):
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        indices = np.random.choice(min(self.num_seen_examples, self.buffer_size), size=size, replace=False)
        if transform is None:
            transform = lambda x: x

        return (
            torch.stack([transform(ee) for ee in self.examples[indices]]),
            self.labels[indices],
            self.task_labels[indices] if self.task_labels is not None else None
        )
