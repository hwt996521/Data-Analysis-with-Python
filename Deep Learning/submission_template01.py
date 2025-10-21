import numpy as np
import torch
from torch import nn

def create_model():
    # 创建包含三个线性层的模型: 784->256->16->10
    model = nn.Sequential(
        nn.Linear(784, 256),  # 784 输入, 256 神经元
        nn.ReLU(),            # ReLU 激活函数
        nn.Linear(256, 16),   # 256 输入, 16 神经元  
        nn.ReLU(),            # ReLU 激活函数
        nn.Linear(16, 10)     # 16 输入, 10 神经元 (无激活函数)
    )
    return model

def count_parameters(model):
    # 计算模型参数总数
    return sum(p.numel() for p in model.parameters())

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = create_model()
    
    # 输出模型信息
    print("模型创建成功:")
    print(model)
    
    # 计算参数数量
    num_params = count_parameters(model)
    print(f"\n模型参数总数: {num_params}")
    
    # 测试数据验证
    test_input = torch.ones(1, 784)
    output = model(test_input)
    print(f"\n测试输入 (1, 784) 的输出维度: {output.shape}")
