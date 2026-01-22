import mlflow.pyfunc
import torch
import pandas as pd
import os
from paysage_classification.inference import inference


# Описываем обертку
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Импорты внутри, чтобы избежать ошибок сериализации
        import torch
        from torchvision import transforms
        
        self.model = torch.load(context.artifacts["model_path"], map_location="cpu", weights_only=False)
        self.model.eval()

        self.ss = 150 # params['ss']
        self.labels_list = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
        
        self.transform_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([self.ss, self.ss]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, context, model_input):
        
        # Извлекаем путь к картинке
        if isinstance(model_input, pd.DataFrame):
            img_path = str(model_input.iloc[0, 0])
        else:
            img_path = str(model_input)

        try:
            inference(
                self.model,
                img_path,
                self.labels_list,
                self.transform_list,
                output="std" # Попробуй передать спец. флаг, чтобы функция вернула строку
            )
            result = 10
            # Если твоя функция только принтит, придется возвращать кастомную строку
            if result is None:
                return ["Check server console (inference function returned None)"]
            return [result]
        except Exception as e:
            return [f"Error during inference: {str(e)}"]

# Запуск процесса логирования
if __name__ == "__main__":

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="landscape_server",
            python_model=ModelWrapper(),
            artifacts={"model_path": "models/260121_100-256-88_1-0.0000.pt"},
            code_paths=["paysage_classification"] # Убедись, что папка рядом со скриптом
        )
        print(f"\nSuccess! RUN_ID: {run.info.run_id}")
        print(f"Команда для запуска сервера:")
        print(f"poetry run mlflow models serve -m runs:/{run.info.run_id}/landscape_server -p 5001 --no-conda")