import mlflow

mlflow.set_experiment("lung-ai-model-tracking")

model_path = "models/lung_model_raw.tflite"

with mlflow.start_run(run_name="tflite_model_v1_with_metrics"):
    mlflow.log_param("model_type", "TFLite")
    mlflow.log_param("model_name", "lung_model_raw.tflite")
    mlflow.log_param("epochs", 50)

    mlflow.log_metric("accuracy", 0.8010)
    mlflow.log_metric("loss", 0.46)

    mlflow.log_artifact(model_path, artifact_path="model")

print("Model with metrics logged successfully in MLflow")