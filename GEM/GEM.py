import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import quadprog
from utils import compute_offsets
import torch  # 导入PyTorch深度学习框架
from torchvision import datasets, transforms  # 导入计算机视觉相关数据集和变换
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载工具
import quadprog  # 导入二次规划求解器
import copy  # 导入对象复制工具

def store_grad(pp, grads, grad_dims, tid):
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]

    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin

    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class GEM(nn.Module):
    def __init__(self, model, n_tasks, n_memories, device='cuda', margin=0.5):
        super(GEM, self).__init__()
        self.model = model
        self.n_tasks = n_tasks
        self.n_memories = n_memories
        self.device = device
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.model.parameters(), lr=0.001)

        self.memory_data = torch.FloatTensor(n_tasks, n_memories, 1, 28, 28).to(device)
        self.memory_labs = torch.LongTensor(n_tasks, n_memories).to(device)

        self.grad_dims = [p.data.numel() for p in self.model.parameters()]
        self.grads = torch.zeros(sum(self.grad_dims), n_tasks).to(device)

        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.nc_per_task = 2

    def observe(self, x, t, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(x.data[:effbsz])
        self.memory_labs[t, self.mem_cnt: endcnt].copy_(y.data[:effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.model.zero_grad()
                past_task = self.observed_tasks[tt]
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task)

                ptloss = self.ce(
                    self.model(self.memory_data[past_task])[:, offset1:offset2],
                    self.memory_labs[past_task]
                )
                ptloss.backward()
                store_grad(self.model.parameters, self.grads, self.grad_dims, past_task)

        self.model.zero_grad()
        offset1, offset2 = compute_offsets(t, self.nc_per_task)
        loss = self.ce(self.model(x)[:, offset1:offset2], y)
        loss.backward()

        if len(self.observed_tasks) > 1:
            store_grad(self.model.parameters, self.grads, self.grad_dims, t)
            indx = torch.LongTensor(self.observed_tasks[:-1]).to(self.device)
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                overwrite_grad(self.model.parameters, self.grads[:, t], self.grad_dims)

        self.opt.step()
        return loss.item()
