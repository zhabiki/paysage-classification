from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from samples import Sample


class SceneDataset(Dataset):
    def __init__(
            self,
            samples: list[Sample],
            images_path: Path,
            transforms: Callable,
        ):
        self.transforms = transforms
        self.images_path = images_path

        # Запускаем предобработку изображений
        self.preprocess_samples(samples)

    def load_sample(self, sample):
        # Соединяем переданную папку с названием файла в датафрейме
        file_path = self.images_path / sample['image_name']

        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')

            # Масштабируем изображение для лучшей сходимости
            img_array = np.array(img_rgb) / 255.0

            # Переводим изображение в тензор
            img_tensor = self.transforms(img_array).float()

            return (img_tensor, int(sample['label']))
    
    def preprocess_samples(self, samples):
        # Создаём список сэмплов (картинка + метка);
        # Нам нужен список кортежей. Кортеж -- пара тензор-метка
        img_list = list(
            map(lambda sample: self.load_sample(sample), samples)
        )

        self.__dataset = img_list

    def __getitem__(self, index):
        return self.__dataset[index]
    
    def __len__(self):
        return len(self.__dataset)
