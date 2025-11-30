# app.py
import mlflow.pyfunc
import mlflow
import pandas as pd
import time
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app, Counter, Histogram
import uvicorn

import os
#设置使用本地SQLite数据库
db_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns", "mlflow.db"))
#DB_URI = f"sqlite:///{db_path}"
#print(DB_URI)
#mlflow.set_tracking_uri(DB_URI)

# 1. 初始化 FastAPI 应用
app = FastAPI()

# 2. 定义 Prometheus 指标 (Metrics)
# Counter: 只增不减的计数器，用于统计请求总量
REQUEST_COUNT = Counter(
    'model_request_count',          # 指标名称
    'Total number of model requests', # 描述
    ['status']                      # 标签: 成功/失败
)

# Histogram: 直方图，用于统计耗时分布 (P99, P95, Avg Latency)
REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Model inference latency in seconds'
)

# 3. 创建 Prometheus 的 metrics 接口 (/metrics)
# Prometheus 会定期访问这个接口抓取数据
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 4. 加载模型
# ⚠️ 注意：为了 Demo 方便，这里我们写死模型路径
# 在 Docker 中，我们会把本地的 mlruns 挂载进去
# 假设我们加载 DemoModel 的最新生产版本或特定版本
#MODEL_URI = "models:/RandomForestModel/Latest" # 或者使用本地路径

#Run ID: 3967c8324f23425eb7abb942376d2c4c
#MODEL_URI = "/app/mlruns/419493442711422412/models/m-d3e89cdc620242159a62f4007e1c1b59/artifacts"
MODEL_URI = "file:///E:/study/docker_demo/monitor_demo/mlruns/419493442711422412/models/m-d3e89cdc620242159a62f4007e1c1b59/artifacts"

print(f"正在加载模型: {MODEL_URI} ...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("模型加载成功！")

# 5. 定义预测接口
@app.post("/invocations")
async def predict(request: Request):
    # 开始计时
    start_time = time.time()
    
    try:
        # 获取 JSON 数据
        json_data = await request.json()
        
        # 转换为 Pandas DataFrame (适配 MLflow 标准输入)
        # 这里假设输入格式是 split 格式，如之前 demo 所示
        if "dataframe_split" in json_data:
            data = pd.DataFrame(**json_data["dataframe_split"])
        else:
            # 简单处理其他格式
            data = pd.DataFrame(json_data)

        # 模型推理
        result = model.predict(data)
        
        # 记录成功指标
        REQUEST_COUNT.labels(status='success').inc()
        
        # 计算耗时并记录
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        return result.tolist()

    except Exception as e:
        # 记录失败指标
        REQUEST_COUNT.labels(status='error').inc()
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # 启动服务，监听 5000 端口
    uvicorn.run(app, host="0.0.0.0", port=5000)