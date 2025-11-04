import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):      
        super().__init__()
        # Определение слоев сети
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5)  # 3 входных канала, 3 фильтра (5x5)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # Явно указываем кортеж для kernel_size
        self.conv2 = nn.Conv2d(3, 5, kernel_size=3)  # 3 входных канала, 5 фильтров (3x3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # Явно указываем кортеж для kernel_size
        
        self.flatten = nn.Flatten()  # Операция flatten
        
        # Полносвязные слои
        self.fc1 = nn.Linear(5 * 6 * 6, 100)  # 5*6*6 входных нейронов, 100 выходных
        self.fc2 = nn.Linear(100, 10)  # 100 входных нейронов, 10 выходных


    
    def forward(self, x):
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
    class AdvancedConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Сверточные слои с BatchNorm и увеличенным количеством фильтров
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32 фильтра 3x3
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 фильтра 3x3
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 фильтров 3x3
        self.bn3 = nn.BatchNorm2d(128)
        
        # MaxPool слои
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.3)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Увеличиваем количество нейронов
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # 10 классов на выходе

    def forward(self, x):
        # Размерность x ~ [batch_size, 3, 32, 32]
        
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
