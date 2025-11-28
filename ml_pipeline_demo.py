import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys


import os
#设置使用本地SQLite数据库
db_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns", "mlflow.db"))
DB_URI = f"sqlite:///{db_path}"
print(DB_URI)
mlflow.set_tracking_uri(DB_URI)


print("第一次自动化测试")
# 设置实验名称
mlflow.set_experiment("CI_CD_Automation_Demo")

def train_model():
    """训练新模型并返回准确率和Run ID"""
    print("🚀 开始训练新模型 (Challenger)...")
    
    # 1. 准备数据
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    # 2. 训练 (为了演示，我们随机调整参数以模拟模型变化)
    # 在实际场景中，这里通常读取配置文件
    with mlflow.start_run() as run:
        # 这里故意把 n_estimators 设大一点，争取获得好结果
        clf = RandomForestClassifier(n_estimators=50) 
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # 记录指标
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(clf, "model")
        
        print(f"✅ 新模型训练完成。Accuracy: {accuracy:.4f}")
        return run.info.run_id, accuracy

def get_production_accuracy(model_name):
    """获取当前 Production 模型的准确率"""
    client = mlflow.tracking.MlflowClient()
    
    try:
        # 寻找被标记为 "Production" 的模型版本
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            print("ℹ️ 当前没有 Production 模型。")
            return 0
        
        latest_production = versions[0]
        run_id = latest_production.run_id
        
        # 获取该版本的指标
        metric_history = client.get_metric_history(run_id, "accuracy")
        if metric_history:
            return metric_history[0].value
        return 0
        
    except Exception as e:
        # 如果模型还没注册过，会报错，视为没有 Production 模型
        print(f"ℹ️ 获取 Production 模型失败 (可能是第一次运行): {e}")
        return 0

def promote_model(model_name, run_id, new_accuracy, old_accuracy):
    """将新模型注册并升级为 Production"""
    client = mlflow.tracking.MlflowClient()
    
    # 1. 注册模型 (会在 Model Registry 创建新版本)
    print(f"📝 正在注册新模型版本...")
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        model_name
    )
    version = result.version
    
    # 2. 只有当新模型更优时，才标记为 Production
    # (如果是第一次运行，old_accuracy 为 0，也会直接升级)
    if new_accuracy >= old_accuracy:
        print(f"🏆 挑战成功! (New: {new_accuracy:.4f} >= Old: {old_accuracy:.4f})")
        print(f"🔄 正在将版本 {version} 转换为 Production...")
        
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True # 把旧的 Production 归档
        )
    else:
        print(f"❌ 挑战失败。 (New: {new_accuracy:.4f} < Old: {old_accuracy:.4f})")
        print("该模型已注册，但不会被推送到 Production。")

if __name__ == "__main__":
    MODEL_NAME = "DemoModel"
    
    # 1. 训练新模型
    new_run_id, new_acc = train_model()
    
    # 2. 获取旧模型指标
    old_acc = get_production_accuracy(MODEL_NAME)
    
    # 3. 比较并部署
    promote_model(MODEL_NAME, new_run_id, new_acc, old_acc)