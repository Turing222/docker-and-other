import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import model
from model import test_data
from fastapi import FastAPI, UploadFile, File
import uvicorn

from preimage import transform_image

# 1. 重新实例化一个"空脑子"的模型
model_new = model.MyFashionModel()

# 2. 加载保存好的参数 (注入记忆)
# map_location='cpu' 确保即使你在GPU上训练的，在没有GPU的电脑上也能加载

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

try:
    model_new.load_state_dict(torch.load("model_weights.pth", map_location=device))
    print("模型权重加载成功！")
except FileNotFoundError:
    print("⚠️ 警告：找不到 model_weights.pth，模型将使用随机参数（预测会不准）")
# 3. 把模型搬到现在的设备上
model_new.to(device)
model_new.eval() # 切换到预测模式


#print("模型加载成功！")
#image_path = 'src/1.jpg'
#with open(image_path, 'rb') as f:
#    input_tensor=transform_image(f.read())
#pred = model_new(input_tensor)


classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
#创建 FastAPI 应用
app = FastAPI()
print("FastAPI 应用创建成功！")
@app.get("/")
def home():
    return {"message": "欢迎使用 Fashion AI 识别服务！请访问 /docs 进行测试"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. 读取上传的文件内容
    image_bytes = await file.read()
    
    # 2. 预处理图片
    try:
        input_tensor = transform_image(image_bytes)
    except Exception as e:
        return {"error": f"图片处理失败: {str(e)}"}

    # 3. 模型推理
    with torch.no_grad():
        pred = model_new(input_tensor)
        predicted_index = pred.argmax(1).item()
        confidence = pred.softmax(1).max().item() # 计算置信度(概率)

    # 4. 返回结果
    return {
        "filename": file.filename,
        "prediction": classes[predicted_index], # 预测的类别名
        "confidence": f"{confidence*100:.2f}%"  # 这是一个百分比
    }