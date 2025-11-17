import argparse
import numpy as np
import torch
from dataset import SceneDataset, SceneTestset
from scene_classifier import SceneClassifier
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description='Аргументы для пайплайна')

    #Задаём параметры работы нашей модели
    parser.add_argument('--mode', type=str, default='train', help='Режим работы')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох обучения')
    #Если файл с весами находится не в одной папке с main.py, то скорее всего придётся указывать полный путь
    parser.add_argument('--checkpoint', type=str, default='scene_classifier.pt', help='Файл для сохранения и загрузки модели')
    parser.add_argument('--batch_size', type=int, default=256, help='Размер батча данных для обучения и теста')

    #Задаём пути, откуда мы будем брать данные
    parser.add_argument('--train_dataset_table', type=str, default='train-scene classification/train.csv', help='Путь до файла с таблицей названий изображений')
    parser.add_argument('--test_dataset_table', type=str, default='test_WyRytb0.csv', help='Путь до файла с таблицей названий изображений (тестовая выборка)')
    parser.add_argument('--label_file', type=str, default='labels.txt', help='Путь до файла со списком меток')
    parser.add_argument('--train_folder', type=str, default='train-scene classification/train', help='Путь до папки с тренировочными данными')

    args = parser.parse_args()
    
    if args.mode == 'train':
        transform_list = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([150,150]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
        ])

        print('transforms created')

        # Создаём экземпляры датасета и модели
        dataset = SceneDataset(args.train_dataset_table,
                            args.label_file,
                            args.train_folder,
                            transform_list)
        model = SceneClassifier(args.epochs)

        print("Preprocessed, slava tebe gospodi")

        # Формируем из датасета батчи
        batched_dataset = DataLoader(dataset=dataset, batch_size=256, shuffle=True)

        for epoch in range(args.epochs):
            print(f'Эпоха {epoch} началась')
            epoch_acc = []
            
            # Проходимся по батчам - вся логика обучения и классификации реализована внутри модели
            for images, labels in batched_dataset:
                output = model({'images': images, 'labels': labels})
                epoch_acc.append(output)

            print(np.mean(epoch_acc))

        torch.save(model, args.checkpoint)

    if args.mode == 'test':
        transform_list = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([150,150]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
        ])

        model = torch.load('/kaggle/input/scene-classifier-checkpoint/pytorch/default/1/scene_classifier.pt', weights_only=False)
        model.test()

        testset = SceneTestset(args.test_dataset_table,
                            args.label_file,
                            args.train_folder,
                            transform_list)

        batched_testset = DataLoader(dataset=testset, batch_size=args.batch_size)


        predictions = []
        for img in batched_testset:
            img = img.to(model.device)
            predictions.append(model(img))

        predictions = torch.cat(predictions, dim=0)
        print(predictions)

if __name__ == '__main__':
    print('main started')    
    main()