import numpy as np
import torch
from torch import nn

def create_model():
    # Создаем модель с тремя линейными слоями: 784->256->16->10
    model = nn.Sequential(
        nn.Linear(784, 256),  # 784 входа, 256 нейронов
        nn.ReLU(),            # Функция активации ReLU
        nn.Linear(256, 16),   # 256 входа, 16 нейронов
        nn.ReLU(),            # Функция активации ReLU
        nn.Linear(16, 10)     # 16 входа, 10 нейронов (без активации)
    )
    return model

def count_parameters(model):
    # Подсчитываем общее количество параметров модели
    return sum(p.numel() for p in model.parameters())

# Пример использования
if __name__ == "__main__":
    # Создаем модель
    model = create_model()
    
    # Выводим информацию о модели
    print("Модель создана:")
    print(model)
    
    # Подсчитываем параметры
    num_params = count_parameters(model)
    print(f"\nКоличество параметров в модели: {num_params}")
    
    # Проверяем на тестовых данных
    test_input = torch.ones(1, 784)
    output = model(test_input)
    print(f"\nРазмер вывода для тестового входа (1, 784): {output.shape}")
