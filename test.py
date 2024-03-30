import torch

# 加载保存的模型
model = torch.load("savemodels/restaurant_0.59826.pt")

# 遍历模型的每一层
for name, module in model.items():
    print(f"Network name: {name}")
    print(module.__class__.__name__)
    print(module)
    print()
