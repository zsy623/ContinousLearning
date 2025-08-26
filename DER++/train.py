import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from model import DERplusplus
from backbone import CNN
from args import args


def get_cifar10_loaders():
    """下载并创建CIFAR-10数据加载器"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 完整训练集
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    # 测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return full_trainset, testset


def split_tasks(full_trainset, num_tasks=5):
    """将CIFAR-10划分为多个任务"""
    tasks = []
    classes_per_task = 10 // num_tasks

    for i in range(num_tasks):
        # 选择当前任务的类别
        class_start = i * classes_per_task
        class_end = (i + 1) * classes_per_task

        # 获取属于当前任务的样本索引
        indices = [idx for idx, (_, label) in enumerate(full_trainset)
                   if class_start <= label < class_end]

        # 创建当前任务的数据子集
        task_subset = torch.utils.data.Subset(full_trainset, indices)
        task_loader = torch.utils.data.DataLoader(
            task_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        tasks.append(task_loader)

    return tasks


def evaluate(model, testset, device, num_tasks_seen):
    """评估模型在所有已见任务上的性能，并分别打印"""
    model.eval()
    accuracies = {}
    classes_per_task = 10 // args.num_tasks

    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            class_start = task_id * classes_per_task
            class_end = (task_id + 1) * classes_per_task

            # 创建当前任务的测试集子集
            task_indices = [idx for idx, (_, label) in enumerate(testset)
                            if class_start <= label < class_end]
            task_subset = torch.utils.data.Subset(testset, task_indices)
            task_loader = torch.utils.data.DataLoader(
                task_subset, batch_size=100, shuffle=False, num_workers=2)

            correct = 0
            total = 0
            for images, labels in task_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model.backbone(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0
            accuracies[task_id + 1] = accuracy

    return accuracies


def main():
    # 设置设备
    device = args.device
    print(f"Using device: {device}")

    # 获取数据
    full_trainset, testset = get_cifar10_loaders()
    task_loaders = split_tasks(full_trainset, args.num_tasks)

    # 初始化模型
    backbone = CNN(args.num_classes).to(device)
    model = DERplusplus(backbone, args.loss, device, alpha=args.alpha, beta=args.beta).to(device)

    # 训练循环
    all_accuracies = []

    for task_id, task_loader in enumerate(task_loaders):
        print(f"\nTraining on Task {task_id + 1}")

        # 训练当前任务
        model.train()
        for epoch in range(args.epochs_per_task):
            running_loss = 0.0
            for i, data in enumerate(task_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # 观察数据（训练并添加到缓冲区）
                loss = model.observe(inputs, labels)
                running_loss += loss

                if i % 100 == 99:  # 每100个批次打印一次
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        # 评估当前任务后的性能
        task_accuracies = evaluate(model, testset, device, task_id + 1)
        all_accuracies.append(task_accuracies)

        print(f'--- Evaluation after Task {task_id + 1} ---')
        for t_id, acc in task_accuracies.items():
            print(f'  Accuracy on Task {t_id}: {acc:.2f}%')

    # 打印最终结果
    print("\n--- Final Results ---")
    for i, acc_dict in enumerate(all_accuracies):
        print(f'After Task {i + 1}:')
        for t_id, acc in acc_dict.items():
            print(f'  Accuracy on Task {t_id}: {acc:.2f}%')

    # 计算并打印平均准确率
    avg_accuracies = [np.mean(list(acc_dict.values())) for acc_dict in all_accuracies]
    print(f'\nAverage Accuracy after each task: {[f"{acc:.2f}%" for acc in avg_accuracies]}')
    print(f'Overall Average Accuracy: {np.mean(avg_accuracies):.2f}%')


if __name__ == '__main__':
    main()

# --- Final Results ---
# After Task 1:
#   Accuracy on Task 1: 93.55%
# After Task 2:
#   Accuracy on Task 1: 79.05%
#   Accuracy on Task 2: 76.60%
# After Task 3:
#   Accuracy on Task 1: 82.55%
#   Accuracy on Task 2: 35.35%
#   Accuracy on Task 3: 77.90%
# After Task 4:
#   Accuracy on Task 1: 75.25%
#   Accuracy on Task 2: 16.90%
#   Accuracy on Task 3: 59.70%
#   Accuracy on Task 4: 91.00%
# After Task 5:
#   Accuracy on Task 1: 0.10%
#   Accuracy on Task 2: 18.65%
#   Accuracy on Task 3: 66.90%
#   Accuracy on Task 4: 80.70%
#   Accuracy on Task 5: 93.65%
#
# Average Accuracy after each task: ['93.55%', '77.82%', '65.27%', '60.71%', '52.00%']
# Overall Average Accuracy: 69.87%

