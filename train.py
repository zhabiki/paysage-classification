from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from preprocess import SceneDataset
from classifier import SceneClassifier
from samples import Sample


def model_train(
    samples_train: list[Sample],
    samples_eval: list[Sample],
    path_images: Path,
    labels: dict[int, str],
    transforms,
    epochs: int = 20,
    batch_size: int = 256
) -> SceneClassifier:
    
    # Создаём экземпляры датасетов и модели
    dataset_train = SceneDataset(
        samples_train,
        path_images,
        transforms
    )

    dataset_eval = SceneDataset(
        samples_eval,
        path_images,
        transforms
    )

    model = SceneClassifier(
        epochs,
        len(labels),
    )
    model.train()

    # Формируем из датасета батчи
    batched_dataset_train = DataLoader(dataset_train, batch_size, shuffle=True)
    batched_dataset_eval = DataLoader(dataset_eval, batch_size, shuffle=True)

    MAX_DEGRADATION_ALLOWED = 10
    best_acc = 0.0; best_epoch = 0; best_param = None
    for epoch in range(epochs):
        print(f'Эпоха {epoch + 1} / {epochs}...')
        epoch_acc = []

        batched_dataset_train_progress = tqdm(
            batched_dataset_train, unit=f' Б по {batch_size}'
        )
        
        # Проходимся по батчам --
        # вся логика обучения и классификации реализована внутри модели
        for images, labels in batched_dataset_train_progress:
            acc = model({'images': images, 'labels': labels})
            epoch_acc.append(acc)

            batched_dataset_train_progress.set_postfix(
                Точность=f'{int(acc * 100) / 100}'
            )

        # Переходим к эвалюации...
        model.eval()

        eval_acc = []
        for images, labels in batched_dataset_eval:
            acc = model({'images': images, 'labels': labels, 'eval': True})
            eval_acc.append(acc)
        print(f'Точность после эвалюации: {np.mean(eval_acc)} (лучшая: {best_acc})\n')

        # ...и обратно в трейн
        model.train()
        model.scheduler.step()

        # Сохраняем веса, если они дают самую лучшую точность
        if best_acc < np.mean(eval_acc):
            best_acc = np.mean(eval_acc)
            best_epoch = epoch
            best_param = deepcopy(model.state_dict())

        # Если больше N эпох точность падает, сворачиваем лавочку
        elif (epoch - best_epoch) > MAX_DEGRADATION_ALLOWED:
            print('Модель безнадёжно деградировала, ехать дальше смысла нет.')
            break

    if best_param is not None:
        model.load_state_dict(best_param)

    return model
