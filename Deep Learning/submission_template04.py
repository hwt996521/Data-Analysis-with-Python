import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Сверточные слои с BatchNorm и увеличенным количеством фильтров
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # MaxPool слои
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.3)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Первый сверточный блок
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Второй сверточный блок
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Третий сверточный блок
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Преобразование в одномерный вектор
        x = x.view(-1, 128 * 4 * 4)
        
        # Полносвязные слои с dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def create_model():
    return ConvNet()
