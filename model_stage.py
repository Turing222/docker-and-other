from mlflow import MlflowClient

# 实例化 Mlflow Client
client = MlflowClient()

# 获取刚刚注册的模型版本信息
model_name = "Simple_Linear_Regression_Demo"
# model_info.version 包含了新创建的模型版本号
model_info_version=client.get_latest_versions(model_name)
print(f"Model version {model_info_version} of {model_name} transitioned to Staging.")
# 假设您想将此最新版本标记为 Staging
# 使用 transition_model_version_stage 方法来更改 stage
client.transition_model_version_stage(
    name=model_name,
    version=model_info_version,
    stage="Staging"
)

print(f"Model version {model_info_version} of {model_name} transitioned to Staging.")

# 您也可以将现有的 Production 模型存档，并将新模型提升到 Production
# client.transition_model_version_stage(
#     name=model_name,
#     version=model_info.version,
#     stage="Production",
#     archive_existing_versions=True # 自动将当前 Production 版本的模型移动到 Archived 阶段
# )
