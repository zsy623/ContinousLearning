import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from backbone import CNN
from args import args
from model import ER
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

# 持续学习数据集
class SplitMNIST(Dataset):
    def __init__(self, data, targets, task_id):
        # 数据预处理：归一化并添加通道维度
        self.data = data.float() / 255.0
        self.data = (self.data - 0.1307) / 0.3081  # MNIST标准化
        self.data = self.data.unsqueeze(1)  # 添加通道维度 [N, 1, 28, 28]
        self.targets = targets
        self.task_id = task_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        return img, label, idx

args = args()  # determine arguments
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # determine device
num_class = 10
loss = nn.CrossEntropyLoss()
backbone = CNN(num_class).to(device)

model = ER(backbone, loss, args, transform=None, device = device)

# 准备SplitMNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载原始数据（不应用transform）
full_train = MNIST(root='./data', train=True, download=True)
full_test = MNIST(root='./data', train=False, download=True)

# 创建任务划分
tasks = []
for task_id in range(args.n_tasks):
    start_class = task_id * 2
    end_class = (task_id + 1) * 2

    # 训练数据
    task_train_indices = [i for i, t in enumerate(full_train.targets)
                          if start_class <= t < end_class]
    task_train_data = full_train.data[task_train_indices]
    task_train_targets = full_train.targets[task_train_indices] - start_class

    # 测试数据
    task_test_indices = [i for i, t in enumerate(full_test.targets)
                         if start_class <= t < end_class]
    task_test_data = full_test.data[task_test_indices]
    task_test_targets = full_test.targets[task_test_indices] - start_class

    tasks.append({
        'train': (task_train_data, task_train_targets),
        'test': (task_test_data, task_test_targets)
    })

# 训练循环
for task_id in range(args.n_tasks):
    print(f"\n=== Starting Task {task_id + 1}/{args.n_tasks} ===")
    model.current_task = task_id

    # 准备当前任务数据
    train_data, train_targets = tasks[task_id]['train']
    train_dataset = SplitMNIST(train_data, train_targets, task_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 任务训练
    model.net.train()
    for epoch in range(args.n_epochs):
        total_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            not_aug_inputs = inputs.clone()

            loss = model.observe(inputs, labels, not_aug_inputs)
            total_loss += loss

        print(
            f"Task {task_id + 1} | Epoch {epoch + 1}/{args.n_epochs} | Loss: {total_loss / len(train_loader):.4f}")

    # 任务评估
    model.net.eval()
    all_acc = []
    for eval_task in range(task_id + 1):
        test_data, test_targets = tasks[eval_task]['test']
        test_dataset = SplitMNIST(test_data, test_targets, eval_task)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        all_acc.append(acc)
        print(f"Task {eval_task + 1} Accuracy: {acc * 100:.2f}%")

    print(f"Average Accuracy after Task {task_id + 1}: {np.mean(all_acc) * 100:.2f}%")
