import torchvision as tv
import torch

class SceneClassifier(torch.nn.Module):
    def __init__(self, total_epochs):
        super().__init__()

        self.model = tv.models.resnet50().float()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs # период косинуса (обычно total_epochs)
        )
        self.softmax = torch.nn.Softmax()

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
        self.optimizer.zero_grad()

        logits = self.model(batch['images'])
        probs = self.softmax(logits)

        predicted_labels = torch.argmax(probs, dim=1)
        true_labels = batch['labels']

        loss = self.loss(predicted_labels.float(), torch.tensor(true_labels).float())
        loss.backward()

        self.optimizer.step()