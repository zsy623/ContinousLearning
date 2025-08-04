import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import quadprog
import copy

def compute_offsets(task, nc_per_task):
    offset1 = task * nc_per_task
    offset2 = (task + 1) * nc_per_task
    return offset1, offset2


def get_split_mnist(task_id, n_tasks=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    classes = list(range(task_id * 2, (task_id + 1) * 2))

    train_indices = [i for i, label in enumerate(mnist_train.targets) if label in classes]
    train_data = mnist_train.data[train_indices]
    train_labels = mnist_train.targets[train_indices]

    test_indices = [i for i, label in enumerate(mnist_test.targets) if label in classes]
    test_data = mnist_test.data[test_indices]
    test_labels = mnist_test.targets[test_indices]

    label_map = {cls: i for i, cls in enumerate(classes)}
    train_labels = torch.tensor([label_map[label.item()] for label in train_labels])
    test_labels = torch.tensor([label_map[label.item()] for label in test_labels])

    train_dataset = TensorDataset(train_data.float().unsqueeze(1), train_labels)
    test_dataset = TensorDataset(test_data.float().unsqueeze(1), test_labels)

    return train_dataset, test_dataset
