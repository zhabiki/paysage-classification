import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SceneDataset(Dataset):
    def __init__(self, trainset_path, label_path, path_to_img, transforms=None):        
        # Создаём словарь меток
        with open(label_path, 'r', encoding='utf-8') as f:
            label_list = list(map(lambda x: x.replace('\n', ''), f.readlines()))
            self.labels = {label: label_name for label, label_name in enumerate(label_list)}

        print('labels created')

        self.transforms = transforms

        # Предобрабатываем изображения
        self.preprocess_images(trainset_path, path_to_img) # Процесс предобработки изображений

    def load_an_image(self, row, path_to_img):
        #Соединяем переданную папку с названием файла в датафрейме
        file_path = f"{path_to_img}/{row['image_name']}"

        #Открываем изображение
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB') #Перевод в RGB

            img_array = np.array(img_rgb) / 255.0 #Масштабируем изображение для лучшей сходимости

            img_tensor = self.transforms(img_array).float() #Переводим изображение в тензор

            return (img_tensor,int(row['label'])) #Возвращаем в виде тензора с меткой
    
    def preprocess_images(self, trainset_path, path_to_img):
        # Открываем датафрейм с изображениями (создаём тренировочную выборку)
        path = trainset_path
        df = pd.read_csv(path)

        # Создаём список изображений
        img_list = df.apply(
            lambda row: self.load_an_image(row,path_to_img),
            axis=1
        ).tolist() #Нам нужен список кортежей. Кортеж - пара тензор-метка

        self.__dataset = img_list #Тренировочный датасет готов

    def __getitem__(self, index):
        return self.__dataset[index]
    
    def __len__(self):
        return len(self.__dataset)
    

class SceneTestset(Dataset):
    def __init__(self, testset_path, label_path, path_to_img, transforms=None):        
        # Создаём словарь меток
        with open(label_path, 'r', encoding='utf-8') as f:
            label_list = list(map(lambda x: x.replace('\n', ''), f.readlines()))
            self.labels = {label: label_name for label, label_name in enumerate(label_list)}

        print('labels created')

        self.transforms = transforms

        # Предобрабатываем изображения
        self.preprocess_images(testset_path, path_to_img) # Процесс предобработки изображений

    def load_an_image(self, row, path_to_img):
        #Соединяем переданную папку с названием файла в датафрейме
        file_path = f"{path_to_img}/{row['image_name']}"

        #Открываем изображение
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB') #Перевод в RGB

            img_array = np.array(img_rgb) / 255.0 #Масштабируем изображение

            img_tensor = self.transforms(img_array).float() #Переводим изображение в тензор

            return img_tensor #Возвращаем в виде тензора
    
    def preprocess_images(self, testset_path, path_to_img):
        # Открываем датафрейм с изображениями (создаём тренировочную выборку)
        path = testset_path
        df = pd.read_csv(path)

        # Создаём список изображений
        img_list = df.apply(
            lambda row: self.load_an_image(row,path_to_img),
            axis=1
        ).tolist() #Нам нужен список

        self.__dataset = img_list #Тестовая выборка готова

    def __getitem__(self, index):
        return self.__dataset[index]
    
    def __len__(self):
        return len(self.__dataset)