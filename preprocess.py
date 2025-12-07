from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from samples import Sample


def process_image(
    path_image: Path,
    transforms: Callable
):
    try:
        with Image.open(path_image) as img:
            img_rgb = img.convert('RGB')

            # Масштабируем изображение для лучшей сходимости
            img_array = np.array(img_rgb) / 255.0

            # Применяем трансформации, включая перевод в тензор
            img_tensor = transforms(img_array).float()

            return img_tensor

    except Exception:
        print(f'Не удалось загрузить картинку "{path_image}"!')
        return None


class SceneDataset(Dataset):
    def __init__(
            self,
            samples: list[Sample],
            path_images: Path,
            transforms: Callable
        ):
        self.transforms = transforms
        self.path_images = path_images

        # Запускаем предобработку изображений
        self.preprocess_samples(samples)

    def load_sample(self, sample):
        # Соединяем переданную папку с названием файла в датафрейме
        path_image = self.path_images / sample['image_name']

        tensor = process_image(path_image, self.transforms)

        return (tensor, int(sample['label']))
    
    def preprocess_samples(self, samples):
        # Создаём список сэмплов (картинка + метка);
        # Нам нужен список кортежей. Кортеж -- пара тензор-метка
        img_list = list(
            map(lambda sample: self.load_sample(sample), samples)
        )

        img_list = list(
            filter(lambda tensor: tensor is not None, img_list)
        )

        self.__dataset = img_list

    def __getitem__(self, index):
        return self.__dataset[index]
    
    def __len__(self):
        return len(self.__dataset)
