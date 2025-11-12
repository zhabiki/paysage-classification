import argparse
from dataset import SceneDataset
from scene_classifier import SceneClassifier
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description='Мир вашему дому')
    
    # Параметры модели
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--mode', type=str, default='train', help='Режим работы модели')

    # Работа с файлами
    parser.add_argument('--save', type=bool, default=True, help='Сохранить модель?')
    parser.add_argument('--saveFileName', type=str, default='checkpoint', help='Имя файла чекпоинта')
    parser.add_argument('--loadFile', type=str, help='Загрузить чекпоинт')

    args = parser.parse_args()

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((150,150)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # Создаём экземпляры датасета и модели
    dataset = SceneDataset(train_transform)
    model = SceneClassifier(args.epochs)

    # Формируем из датасета батчи
    batched_dataset = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    # Проходимся по батчам - вся логика обучения и классификации реализована внутри модели
    for images, labels in batched_dataset:
        print({'images': images, 'labels': labels})
        model({'images': images, 'labels': labels})


if __name__ == '__main__':
    import os
    print(len(os.listdir('./')))
    main()