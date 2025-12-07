import click
import torch
from torchvision import transforms
import kagglehub
from pathlib import Path
from datetime import datetime

from samples import read_samples, read_splits, create_splits
from test import model_test, show_metrics
from train import model_train
from visualize import visualize
from inference import inference


@click.command()
@click.option('--mode', required=True, type=click.Choice([
    'train', 'test', 'visualize', 'inference' # Тренировка, тестирование, визуализация, инференс
]), help='Режим работы')
@click.option('--e', default=100, show_default=True, help='Кол-во эпох (обучение прервётся, если точность не растёт >10 эпох)')
@click.option('--bs', default=256, show_default=True, help='Размер батча')
@click.option('--ss', default=88, show_default=True, help='Размер стороны изображения')
@click.option('--path-labels', default='data/labels.txt', help='Путь до файла с названиями классов')
@click.option('--path-splits', default='data/', help='Путь до CSVшек с выборками train-eval-test')
@click.option('--path-checkpoint', default='models/', help='Путь до модели (при test) либо место сохранения оной (при train)')
@click.option('--path-infers', default='pred/', help='Путь до папки с картинками для предсказывания (при inference)')
@click.option('--dataset-name', default='nitishabharathi/scene-classification', help='Название датасета на KaggleHub')
@click.option('--dataset-splits', default='0.7-0.2-0.1', help='Как разделить датасет (три числа от 0 до 1 через дефис)')
def run(
    mode, e, bs, ss, # Epochs, BatchSize, SampleSize
    path_labels, path_splits, path_checkpoint, path_infers,
    dataset_name, dataset_splits
):
    # Для начала, скачем датасет! Если оный уже был скачан,
    # KaggleHub сам определит это и ничего перекачивать не будет.
    path_dataset = Path(
        kagglehub.dataset_download(dataset_name)
    )
    path_images = path_dataset / 'train-scene classification' / 'train'
    path_csv = path_dataset / 'train-scene classification' / 'train.csv'
    # print(path_dataset)

    # Считаем входные метки, чтобы не делать этого по многу раз
    labels_f = open(path_labels, 'r', encoding='utf-8')
    labels_list = list(map(lambda x: x.replace('\n', ''), labels_f.readlines()))
    labels = {label_no: label_name for label_no, label_name in enumerate(labels_list)}

    # Как мы хочем выполнить препроцессинг изображений
    transform_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([ss, ss]), # Ориг. разрешение: [150, 150]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f'Подготовка к {mode} завершена, обработка...')


    if mode == 'train':
        # Обучаем с нуля модель и сохраняем чек-поинты
        try:
            [train_samples, eval_samples] = read_splits(['train', 'eval'], path_splits)
        except:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(path_csv, dataset_splits, path_splits)
            [train_samples, eval_samples] = read_splits(['train', 'eval'], path_splits)

        model = model_train(
            train_samples,
            eval_samples,
            path_images,
            labels,
            transform_list,
            e,
            bs
        )

        model_date = datetime.now().strftime("%y%m%d")
        model_name = f'{path_checkpoint}{model_date}-{e}-{bs}-{ss}.pt'

        torch.save(model, model_name)
        print('Обучение завершено!')


    elif mode == 'test':
        # Загружаем модель и прогоняем на тестовых данных
        try:
            [test_samples] = read_splits(['test'], path_splits)
        except:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(path_csv, dataset_splits, path_splits)
            [test_samples] = read_splits(['test'], path_splits)

        model = torch.load(path_checkpoint, weights_only=False)

        true, pred = model_test(
            model,
            test_samples,
            path_images,
            transform_list,
            bs
        )
        show_metrics(true, pred, labels)
    

    elif mode == 'visualize':
        # В случае с визуализацией, нас сплиты не интересуют
        samples = read_samples(path_csv)

        visualize(
            samples,
            path_images,
            labels_list,
            transform_list
        )
    

    else:
        # В остальных случаях -- просто выполняем инференс
        image_pathes = [file for file in Path(path_infers).iterdir() if file.is_file()]

        model = torch.load(path_checkpoint, weights_only=False)

        for image_path in image_pathes:
            inference(
                model,
                image_path,
                labels_list,
                transform_list,
                output='mpl'
            )


if __name__ == '__main__':
    run()
