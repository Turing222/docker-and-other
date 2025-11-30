# 使用官方 Python 瘦身版镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
# 包含: mlflow, fastapi, uvicorn, prometheus-client, pandas, scikit-learn
RUN pip install mlflow fastapi uvicorn prometheus-client pandas scikit-learn

# 复制代码
COPY app.py .
COPY mlruns /app/mlruns
# 启动命令
CMD ["python", "app.py"]