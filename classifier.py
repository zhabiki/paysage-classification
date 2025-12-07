import torchvision as tv
import torch
from tqdm import tqdm


class SceneClassifier(torch.nn.Module):
    def __init__(
        self,
        total_epochs: int,
        n_classes: int
    ):
        super().__init__()
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html
        # https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py#L233

        # Загружаем предобученную на ImageNet модель
        self.model = tv.models.efficientnet_v2_s(
            weights='IMAGENET1K_V1',
            dropout=0.3
        )

        # Отрезаем старую головёшку, пришиваем новую...
        self.model.classifier[-1] = torch.nn.Linear(
            in_features=1280,
            out_features=n_classes
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs # Период косинуса (обычно total_epochs)
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.mode = 'train'

    def train(self, is_eval = False):
        self.mode = 'train' if not is_eval else 'eval'
        self.model.train()
    
    def eval(self):
        self.mode = 'eval'
        self.model.eval()
    
    def test(self):
        self.mode = 'test'
        # self.model.test() # Такого режима нет, брух
        self.model.eval()

    def forward(self, batch):
        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)

        if self.mode == 'train':
            self.optimizer.zero_grad() # Обнуляем прошлые градиенты

            output = self.model(images)
            labels = labels.long()
            loss = self.criterion(output, labels)
            loss.backward()

            self.optimizer.step()

            accuracy = (torch.argmax(output, dim=1) == labels).sum().item() / len(labels)
            return accuracy

        elif self.mode in ['eval', 'test']:
            # НЕ СЧИТАЕМ ГРАДИЕНТЫ В РЕЖИМАХ ЭВАЛЮАЦИИ И ТЕСТА
            with torch.no_grad():
                try:
                    logits = self.model(images)

                    # Выполняем нормализацию значений и выбираем наибольшее
                    probas = torch.nn.functional.softmax(logits, dim=1)
                    predicted_labels = torch.argmax(probas, dim=1)

                    accuracy = (predicted_labels == labels).sum().item() / len(labels)
                    return accuracy

                except BaseException:
                    img = batch
                    output = self.model(img)

                    predicted_labels = torch.argmax(output, dim=1)
                    return predicted_labels

    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            logits = self.model(images)

            # Выполняем нормализацию значений и выбираем наибольшее
            probas = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probas, dim=1)

        return (preds.cpu(), probas.cpu())
