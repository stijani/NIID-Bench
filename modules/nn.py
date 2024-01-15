from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)

        self.fc1 = nn.Linear(in_features=50 * 5 * 5, out_features=500)
        self.out = nn.Linear(in_features=500, out_features=10) # cifar10, mnist
        #self.out = nn.Linear(in_features=500, out_features=100) # cifar100

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        x = F.log_softmax(x, dim=1)

        return x


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=28 * 28 * 1, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=200)
        self.out = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        x = F.log_softmax(x, dim=1)

        return x

MODELS = {
    'lenet': LeNet(),
    'mlp': MLP(),
    'resnet18': resnet18(num_classes=100)
}