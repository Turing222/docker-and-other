import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# 设置批次大小 (一次训练64张图片)
batch_size = 64

# 1. 创建训练集加载器 (注意 shuffle=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# 2. 创建测试集加载器 (注意 shuffle=False)
# (假设你之前也定义了 test_data，如果没有，这行可以先忽略)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
print("DataLoader 创建完成！")

# 从 DataLoader 中通过 for 循环取出一批数据
for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break # 我们只看第一批，看完就跳出循环




# 1. 定义类，必须继承 nn.Module
class MyFashionModel(nn.Module):
    
    # 2. 初始化函数：这里是“备料”的地方
    def __init__(self):
        super().__init__()  # <---【重点】必须先调用父类的初始化
        
        # 定义我们会用到的各种“零件” (层)
        self.flatten = nn.Flatten()  # 把二维图像(28x28)拍扁成一维(784)
        
        # 一个包含多个层的容器
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # 全连接层：输入784 -> 输出512
            nn.ReLU(),             # 激活函数
            nn.Linear(512, 10)     # 输出层：512 -> 10 (对应10个分类)
        )

    # 3. 前向传播函数：这里是“组装/流水线”的地方
    def forward(self, x):
        # x 就是传送带送进来的数据
        x = self.flatten(x)          # 第一步：拍扁
        logits = self.linear_relu_stack(x) # 第二步：过层
        return logits                # 返回结果
# 1. 自动检测设备
# 如果有NVIDIA显卡则用cuda，如果是Mac M1/M2则用mps，否则用cpu    
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# 2. 实例化模型 (造出来)
model = MyFashionModel()

# 3. 搬运模型 (关键步骤！)
# 这一步会把模型里所有的权重矩阵(Weights)移动到显存里
model.to(device)

print(model) # 打印看看结构


# 1. 定义损失函数
# 对于分类问题(10类衣服)，最常用的是 CrossEntropyLoss (交叉熵损失)
loss_fn = nn.CrossEntropyLoss()

# 2. 定义优化器
# SGD (随机梯度下降) 是最经典的优化器。
# model.parameters() 告诉优化器："你要调整的是这个模型里的参数"
# lr=1e-3 (Learning Rate) 是学习率，决定了参数调整的步子大小
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # 切换到训练模式
    
    # 从 DataLoader 里一批一批地拿数据
    for batch, (X, y) in enumerate(dataloader):
        # 【重要】把数据也搬到显卡上，否则会报错！
        X, y = X.to(device), y.to(device)

        # --- 五步走核心 ---
        
        # 1. 计算预测值 (Forward)
        pred = model(X)

        # 2. 计算误差 (Loss)
        loss = loss_fn(pred, y)

        # 3. 梯度清零 (Zero Grad)
        # 每次更新前必须把上一次算的梯度清空，否则会累加
        optimizer.zero_grad()

        # 4. 反向传播 (Backward)
        # 计算每个参数对误差的贡献(梯度)
        loss.backward()

        # 5. 参数更新 (Step)
        # 根据梯度，调整参数
        optimizer.step()

        # ----------------
        
        # 每隔100个批次打印一次进度
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # 切换到评估模式 (很重要！)
    test_loss, correct = 0, 0

    # with torch.no_grad() 表示："接下来的计算不需要算梯度"
    # 这样可以省显存，加速计算
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # 统计猜对的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer) # 训练
        # 假设你有 test_dataloader，如果没有，这行先注释掉
        # test(test_dataloader, model, loss_fn)           # 考试
    print("Done!")
    # 保存模型参数到文件 'model_weights.pth'
    torch.save(model.state_dict(), "model_weights.pth")

    print("模型参数已保存到 model_weights.pth")
