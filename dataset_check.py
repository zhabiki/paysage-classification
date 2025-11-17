import argparse
from collections import Counter
from dataset import SceneDataset
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Аргументы для визуализации')

parser.add_argument('--train_dataset_table', type=str, default='train-scene classification/train.csv', help='Путь до файла с таблицей названий изображений')
parser.add_argument('--label_file', type=str, default='labels.txt', help='Путь до файла со списком меток')
parser.add_argument('--train_folder', type=str, default='train-scene classification/train', help='Путь до папки с тренировочными данными')
parser.add_argument('--visualize', type=bool, default=False, help='Отобразить экземпляры классов в двумерном пространстве')

args = parser.parse_args()

transform_list = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([150,150]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
])

# Создаём экземпляры датасета и модели
dataset = SceneDataset(args.train_dataset_table,
                    args.label_file,
                    args.train_folder,
                    transform_list)

label_list = list(map(lambda x: dataset.labels[x[1]], dataset[:]))

total = len(label_list)
counts = Counter(label_list)

print([f"Доля {label}: {cnt / total}" for label, cnt in counts.items()])

if args.visualize:
    X = [x[0].numpy().reshape(-1) for x in dataset[:]]
    y = [int(x[1]) for x in dataset[:]]

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))  # выбирать палитру под кол-во классов

    plt.figure(figsize=(8, 6))
    for idx, cls in enumerate(classes):
        plt.scatter(
            X_2d[np.array(y) == cls, 0],
            X_2d[np.array(y) == cls, 1],
            color=colors[idx],
            label=f"Класс {cls}",
            alpha=0.8,
            edgecolors='k',
            linewidths=0.5
        )
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("PCA визуализация с легендой по классам")
    plt.legend(title="Класс")
    plt.show()