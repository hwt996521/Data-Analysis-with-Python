import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определение слоев сети согласно требованиям задачи №1
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5)  # 3 входных канала, 3 фильтра (5x5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # MaxPool с ядром 2x2
        self.conv2 = nn.Conv2d(3, 5, kernel_size=3)  # 3 входных канала, 5 фильтров (3x3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # MaxPool с ядром 2x2
        
        self.flatten = nn.Flatten()  # Операция flatten
        
        # Полносвязные слои
        self.fc1 = nn.Linear(5 * 6 * 6, 100)  # 5*6*6 входных нейронов, 100 выходных
        self.fc2 = nn.Linear(100, 10)  # 100 входных нейронов, 10 выходных

    def forward(self, x):
        # Размерность x ~ [batch_size, 3, 32, 32]
        
        # Первый сверточный блок
        x = self.conv1(x)
        x = F.relu(x)  # ReLU активация
        x = self.pool1(x)
        
        # Второй сверточный блок
        x = self.conv2(x)
        x = F.relu(x)  # ReLU активация
        x = self.pool2(x)
        
        # Преобразование в одномерный вектор
        x = self.flatten(x)
        
        # Полносвязные слои
        x = self.fc1(x)
        x = F.relu(x)  # ReLU активация
        x = self.fc2(x)
        
        return x

def create_model():
    return ConvNet()
