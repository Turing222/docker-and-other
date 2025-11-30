import cv2
import numpy as np
import torch

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def transform_image(image_bytes):
    """
    将用户上传的图片字节流，转换为模型能看懂的 Tensor
    """
    # A. 字节 -> OpenCV 图像格式
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # 1. 转为灰度图

    # B. 缩放 -> 28x28
    img = cv2.resize(img, (28, 28))

    # C. 颜色反转 (关键步骤！)
    # 真实照片通常是白底黑物，但 FashionMNIST 是黑底白物。
    # 如果不反转，模型会把背景当成衣服，预测全错。
    img = cv2.bitwise_not(img)

    # D. 归一化 (0-255 -> 0-1) 并转为 float32
    img = img.astype(np.float32) / 255.0

    # E. 转换为 PyTorch Tensor
    tensor = torch.from_numpy(img)

    # F. 增加维度 [28, 28] -> [1, 1, 28, 28] (Batch, Channel, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor.to(device)