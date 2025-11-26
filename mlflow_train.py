import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_estimators = 100
max_depth = 3

with mlflow.start_run():
    # 记录参数
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # 训练模型
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    ).fit(X_train, y_train)

    # 记录指标
    acc = clf.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)

    # 保存模型
    mlflow.sklearn.log_model(clf, "model")
