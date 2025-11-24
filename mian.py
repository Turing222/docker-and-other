import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import model
from model import test_data



# 1. 重新实例化一个"空脑子"的模型
model_new = model.MyFashionModel()

# 2. 加载保存好的参数 (注入记忆)
# map_location='cpu' 确保即使你在GPU上训练的，在没有GPU的电脑上也能加载
model_new.load_state_dict(torch.load("model_weights.pth", map_location='cpu'))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# 3. 把模型搬到现在的设备上
model_new.to(device)
model_new.eval() # 【关键】切换到评估模式

print("模型加载成功！")

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
