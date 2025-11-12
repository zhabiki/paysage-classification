import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SceneDataset(Dataset):
    def __init__(self, transforms=None):
        # Создаём словарь меток
        with open('labels.txt', 'r', encoding='utf-8') as f:
            label_list = list(map(lambda x: x.replace('\n', ''), f.readlines()))
            self.labels = {label: label_name for label, label_name in enumerate(label_list)}

        self.transforms = transforms

        # Предобрабатываем изображения
        self.preprocess_images()

    def load_an_image(self, row, path_to_img='train-scene classification/train'):
        file_path = f"{path_to_img}/{row['image_name']}"
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')

            img_array = np.array(img_rgb) / 255.0

            img_tensor = self.transforms(img_array).float()

            # img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

            return (img_tensor,int(row['label']))
    
    def preprocess_images(self):
        # Открываем датафрейм с изображениями
        path = "train-scene classification/train.csv"
        df = pd.read_csv(path)

        img_list = df.apply(
            lambda row: self.load_an_image(row),
            axis=1
        ).tolist()

        self.__dataset = img_list

    def __getitem__(self, index):
        return self.__dataset[index]
    
    def __len__(self):
        return len(self.__dataset)