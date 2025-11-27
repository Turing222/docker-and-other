import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



import os

# 使用 os.path.abspath 确保路径是绝对的
# 确保所有操作都在同一个 mlruns 文件夹中
db_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns", "mlflow.db"))
DB_URI = f"sqlite:///{db_path}"

print("db_uri=",DB_URI)
mlflow.set_tracking_uri(DB_URI)


# 1. 设定 MLflow 实验 (Tracking)
# 如果实验不存在，它会被创建
experiment_name = "Simple_Linear_Regression_Demo"
mlflow.set_experiment(experiment_name) 

# 获取当前的运行信息（用于后续的模型注册）
# 如果您在 PyTorch 中使用，可以把这行代码放在训练循环开始前
with mlflow.start_run() as run: 
    # 获取 Run ID
    run_id = run.info.run_id
    
    print(f"MLflow Run ID: {run_id}")

    # --- 2. 数据准备 ---
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1) # 特征 (x)
    y = 4 + 3 * X + np.random.randn(100, 1) # 标签 (y = 4 + 3x + 噪声)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. 模型训练 ---
    # 定义模型参数
    fit_intercept = True
    
    # 训练模型
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    
    # 进行预测
    y_pred = model.predict(X_test)

    # --- 4. 评估指标计算 ---
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print(f"Coefficient: {model.coef_[0][0]:.4f}")

    # --- 5. 使用 MLflow Tracking 记录所有信息 ---

    # 记录参数 (Params)
    # 在 PyTorch 中，这可能是学习率、批次大小、优化器类型等
    mlflow.log_param("fit_intercept", fit_intercept)
    mlflow.log_param("model_type", "LinearRegression")

    # 记录指标 (Metrics)
    # 在 PyTorch 中，这通常是 epoch 损失、准确率等
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # 记录模型本身 (Artifacts - Model)
    # 这是最关键的一步，MLflow 会自动记录模型对象、依赖环境等信息
    # 对于 PyTorch 模型，您可以使用 mlflow.pytorch.log_model()
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        registered_model_name="SimpleLinearModel" # **可选**：同时注册为 MLflow Model Registry 中的模型
    )
    
    # 记录额外的 Artifacts (如数据样本、特征重要性图表等)
    # 我们创建一个简单的文本文件作为例子
    with open("results_summary.txt", "w") as f:
        f.write(f"MSE: {mse}\n")
        f.write(f"R2: {r2}\n")
    mlflow.log_artifact("results_summary.txt")

print("\n--- 训练和日志记录完成！请查看 MLflow UI ---")
print(f"Run ID: {run_id}")