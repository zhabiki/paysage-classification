import torchvision as tv
import torch

class SceneClassifier(torch.nn.Module):
    def __init__(self, total_epochs):
        super().__init__()

        self.model = tv.models.resnet50().float()
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=6, bias=True)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs # период косинуса (обычно total_epochs)
        )

        self.model = self.model.to(self.device)
        self.mode = 'train'

    def set_mode(self, mode):
        assert mode in ['train', 'test']
    
    def toggle_mode(self):
        if self.mode == 'train':
            self.mode = 'test'
        else:
            self.mode = 'train'

    def train(self):
        self.mode = 'train'
    
    def test(self):
        self.mode = 'test'

    def forward(self, batch):
        if self.mode == 'train':
            self.optimizer.zero_grad()

            batch['images'] = batch['images'].to(self.device)
            batch['labels'] = batch['labels'].to(self.device)

            output = self.model(batch['images'])
            true_labels = batch['labels']

            loss = self.criterion(output.float(), torch.tensor(true_labels).long())
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            #return f'Текущая точность: {(torch.argmax(output, dim=1) == true_labels).sum().item() / len(true_labels)}'
            return (torch.argmax(output, dim=1) == true_labels).sum().item() / len(true_labels)
        elif self.mode == 'test':
            try:
                img = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(img)
                probs = self.softmax(logits)
                predicted_labels = torch.argmax(probs, dim=1)

                accuracy = (predicted_labels == labels).sum().item() / len(labels)
                return accuracy
            except BaseException:
                img = batch

                output = self.model(img)
                predicted_labels = torch.argmax(output, dim=1)

                return predicted_labels