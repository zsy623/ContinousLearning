import torch
import torch.nn as nn
import torchvision

class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        # 修改输入通道为 1 (原为 3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.ReLU = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_class)

    def conv(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.ReLU(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.ReLU(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x

    def classify(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.classify(x)
        return x


