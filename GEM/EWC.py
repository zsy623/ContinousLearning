import torch 
import torch.nn as nn  
import torch.optim as optim 
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader, TensorDataset 
import numpy as np 
import quadprog 
import copy 

class EWC(nn.Module):
    def __init__(self, model, importance, device='cuda', fisher_alpha=1000):
        super(EWC, self).__init__()
        self.model = model
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.model.parameters(), lr=0.001)
        self.fisher_alpha = fisher_alpha
        self.importance = importance

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.star_params = {}
        self.fisher = {}

    def observe(self, x, t, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        loss = self.ce(self.model(x), y)

        ewc_loss = 0
        if t > 0:
            for n, p in self.model.named_parameters():
                if n in self.fisher:
                    ewc_loss += (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()

        total_loss = loss + self.fisher_alpha * ewc_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return total_loss.item()

    def on_task_end(self, dataset):
        self.model.train()
        self.star_params = {n: p.clone().detach() for n, p in self.params.items()}

        fisher = {n: p.clone().detach().fill_(0) for n, p in self.params.items()}
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()

            output = self.model(x)
            loss = self.ce(output, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()

        for n, p in fisher.items():
            fisher[n] = p / len(dataloader)

        self.fisher = fisher
