# 1. 选择基础镜像 (就像选择操作系统)
# slim 版本更小，适合生产环境
FROM python:3.10-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 复制依赖文件到容器中
COPY requirements.txt .

# 4. 在容器内安装依赖
# --no-cache-dir 可以减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制当前目录下的所有代码到容器工作目录
COPY . .

# 6. 声明容器运行时监听的端口 (仅作文档说明用，实际映射需在启动时指定)
EXPOSE 8000

# 7. 启动命令：运行 uvicorn 服务器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]