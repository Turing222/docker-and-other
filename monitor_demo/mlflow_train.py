import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

import os

# 使用 os.path.abspath 确保路径是绝对的
# 确保所有操作都在同一个 mlruns 文件夹中
#db_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns", "mlflow.db"))
#DB_URI = f"sqlite:///{db_path}"
#mlflow.set_tracking_uri(DB_URI)
mlflow.set_experiment("mlflow_Demo")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#mlflow.set_tracking_uri("http://127.0.0.1:5001")
input_example = X[:2, :]

n_estimators = 100
max_depth = 3

with mlflow.start_run() as run:
    # 获取 Run ID
    run_id = run.info.run_id
    
    print(f"MLflow Run ID: {run_id}")
    # 记录参数
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    print("跟踪实验参数完毕")
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    ).fit(X_train, y_train)
    print("训练完毕")
    # 记录指标
    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)

    # 保存模型
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="RandomForestModel",
        input_example=input_example
        )
print("\n--- 训练和日志记录完成！请查看 MLflow UI ---")
print(f"Run ID: {run_id}")