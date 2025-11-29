from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics

from preprocess import SceneDataset
from samples import Sample


def model_test(
    model,
    samples_test: list[Sample],
    path_images: Path,
    transforms,
    batch_size: int = 256
) -> tuple[list[int], list[int]]:
               
    # Создаём экземпляры датасета и модели
    dataset_test = SceneDataset(
        samples_test,
        path_images,
        transforms,
    )

    model.test()

    # Формируем из датасета батчи
    batched_dataset_test = DataLoader(dataset_test, batch_size, shuffle=False)

    # images = [data for data in batched_dataset_test]
    classes_true = []
    classes_pred = []

    batched_dataset_progress = tqdm(
        batched_dataset_test, unit=f' Б по {batch_size}'
    )
        
    # Проходимся по батчам
    for images, labels in batched_dataset_progress:
        preds = model.predict(images)

        classes_true.extend(preds.tolist())
        classes_pred.extend(labels.tolist())

    return classes_true, classes_pred


def show_metrics(
    pred: list[int],
    true: list[int],
    labels: dict[int, str]
) -> None:
    pred_labelled = list(map(lambda cls: labels.get(cls) or str(cls), pred))
    true_labelled = list(map(lambda cls: labels.get(cls) or str(cls), true))

    cm = metrics.confusion_matrix(true_labelled, pred_labelled)
    cr = metrics.classification_report(true_labelled, pred_labelled)

    print('\nМатрица ошибок: \n', cm)
    print('\nТочность по классам: \n', cr)
