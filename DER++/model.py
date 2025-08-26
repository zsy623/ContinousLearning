import torch
import torch.nn as nn
from buffer import Buffer
from args import args
from backbone import CNN

class DERplusplus(nn.Module):
    def __init__(self, backbone, loss, device, alpha=0.5, beta=1.0):
        super().__init__()
        self.backbone = backbone
        self.loss = loss
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.buffer = Buffer(args.buffer_size, device)
        self.optim = torch.optim.Adam(self.parameters(), lr=args.lr)

    def observe(self, inputs, labels):
        self.optim.zero_grad()

        # 计算当前任务的损失，并获取 logits
        outputs = self.backbone(inputs)
        loss_current = self.loss(outputs, labels)

        # 总损失初始化为当前任务损失
        total_loss = loss_current

        # 如果有缓冲区数据，添加蒸馏损失和分类损失
        if self.buffer.current_size > 0:
            # 从缓冲区获取数据
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                min(args.get_size, self.buffer.current_size)
            )

            # 计算缓冲区数据的输出
            buf_outputs = self.backbone(buf_inputs)

            # DER蒸馏损失 (MSE between logits)
            if buf_logits is not None:
                # 蒸馏损失需要detach旧的logits，避免梯度回传
                loss_distill = nn.MSELoss()(buf_outputs, buf_logits.detach())
                total_loss += self.alpha * loss_distill

            # DER++分类损失
            loss_replay = self.loss(buf_outputs, buf_labels)
            total_loss += self.beta * loss_replay

        # 反向传播和优化
        total_loss.backward()
        self.optim.step()

        # 将当前数据和其对应的 logits 添加到缓冲区
        # 使用 detach() 来确保存储的 logits 不会携带梯度信息
        self.buffer.add_data(inputs.detach(), labels.detach(), outputs.detach())

        return total_loss.item()