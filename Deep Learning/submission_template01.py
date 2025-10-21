import numpy as np
import torch
from torch import nn
version = torch.__version__
assert version is not None, 'Версия PyTorch не сохранилась в переменную version'
major_version, minor_version = re.findall("\d+\.\d+", version)[0].split('.')
assert float(major_version) >= 2 or (float(major_version) >= 1 and float(minor_version) >= 7), 'Нужно обновить PyTorch'

def create_model():
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 10, bias=True)
    )
    return model
model = create_model()
for param in model.parameters():
    nn.init.constant_(param, 1.)
assert torch.allclose(model(torch.ones((1, 784))), torch.ones((1, 10)) * 3215377.), 'Что-то не так со структурой модели'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
small_model = nn.Linear(128, 256)
assert count_parameters(small_model) == 128 * 256 + 256, 'Что-то не так, количество параметров неверное'
medium_model = nn.Sequential(*[nn.Linear(128, 32, bias=False), nn.ReLU(), nn.Linear(32, 10, bias=False)])
assert count_parameters(medium_model) == 128 * 32 + 32 * 10, 'Что-то не так, количество параметров неверное'
print("Seems fine!")

