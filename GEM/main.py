import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from MLP import MLP
from GEM import GEM
from EWC import EWC
from utils import get_split_mnist, compute_offsets


def train_and_eval(model_alg, n_tasks, n_epochs, device='cuda'):
    train_datasets = [get_split_mnist(i) for i in range(n_tasks)]
    accuracies = np.zeros((n_tasks, n_tasks))

    for t in range(n_tasks):
        print(f"--- Training on Task {t + 1}/{n_tasks} ---")
        train_dataset, _ = train_datasets[t]
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        for epoch in range(n_epochs):
            total_loss = 0
            for i, (x, y) in enumerate(dataloader):
                loss = model_alg.observe(x, t, y)
                total_loss += loss

            print(f"  Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        if isinstance(model_alg, EWC):
            model_alg.on_task_end(train_dataset)

        for eval_t in range(t + 1):
            _, test_dataset = train_datasets[eval_t]
            test_loader = DataLoader(test_dataset, batch_size=128)

            correct = 0
            total = 0
            model_alg.model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)

                    offset1, offset2 = compute_offsets(eval_t, 2)
                    output = model_alg.model(x)[:, offset1:offset2]

                    _, predicted = torch.max(output.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            acc = 100 * correct / total
            accuracies[t, eval_t] = acc
            print(f"  Accuracy on Task {eval_t + 1}: {acc:.2f}%")

    print("\n--- Final Accuracy Matrix (Rows: Trained Task, Columns: Tested Task) ---")
    print(accuracies)

    avg_acc = accuracies.diagonal().mean()
    print(f"\nAverage Accuracy: {avg_acc:.2f}%")

    forgetting = np.max(accuracies, axis=0) - accuracies[-1, :]
    print(f"\nAverage Forgetting: {np.mean(forgetting):.4f}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_tasks = 5
    n_epochs = 5
    n_memories = 100

    print("===================== Running GEM Algorithm =====================")
    gem_model = MLP(784, n_tasks * 2, 256, 2).to(device)
    gem_alg = GEM(gem_model, n_tasks, n_memories, device)
    train_and_eval(gem_alg, n_tasks, n_epochs, device)

    print("\n===================== Running EWC Algorithm =====================")
    ewc_model = MLP(784, n_tasks * 2, 256, 2).to(device)
    ewc_alg = EWC(ewc_model, n_tasks, device)
    train_and_eval(ewc_alg, n_tasks, n_epochs, device)
