import torch
import torch.nn as nn
import torch.optim as optim
from buffer import buffer

# 持续学习模型基类
class ContinualModel(nn.Module):
    def __init__(self, backbone: nn.Module, loss: nn.Module, args, transform, device):
        super().__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = optim.SGD(self.net.parameters(), lr=self.args.lr)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor) -> float:
        pass


# 经验回放模型实现
class ER(ContinualModel):
    def __init__(self, backbone, loss, args, transform, device):  # 添加 device 参数
        super().__init__(backbone, loss, args, transform, device)  # 传递 device
        self.buffer = buffer(self.args.buffer_size, self.device)
        self.current_task = 0

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        # 从缓冲区采样并计算损失
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        # 将当前数据添加到缓冲区
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
            task_labels=torch.ones(len(labels)) * self.current_task
        )

        return loss.item()