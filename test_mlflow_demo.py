import requests
import json
import pandas as pd

# 1. 定义服务的 URL
# 注意：MLflow 标准的预测接口路径是 /invocations
url = "http://127.0.0.1:5000/invocations"

# 2. 构造测试数据
# 我们模拟两条数据，基于 Iris 数据集的 4 个特征
data = [
    [5.1, 3.5, 1.4, 0.2],  # 第一条样本
    [6.7, 3.0, 5.2, 2.3]   # 第二条样本
]

# 3. 构造请求 Payload (数据包)
# MLflow 2.0+ 推荐使用 'dataframe_split' 格式，包含列名和数据
# 这种格式最通用，能避免 Pandas 索引报错
payload = {
    "dataframe_split": {
        "columns": ["sepal length", "sepal width", "petal length", "petal width"],
        "data": data
    }
}

# 4. 设置请求头
headers = {"Content-Type": "application/json"}

# 5. 发送 POST 请求
try:
    print(f"正在向 {url} 发送请求...")
    response = requests.post(url, json=payload, headers=headers)
    
    # 6. 处理结果
    if response.status_code == 200:
        predictions = response.json()
        print("\n✅ 成功获取预测结果！")
        print("----------------------")
        print(f"输入数据: {data}")
        print(f"模型预测: {predictions['predictions']}") # 或者是 predictions
        print("----------------------")
    else:
        print(f"\n❌ 请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

except Exception as e:
    print(f"\n❌ 发生连接错误: {e}")
    print("请检查：1. 服务窗口是否还在运行？ 2. 端口号 5001 是否正确？")