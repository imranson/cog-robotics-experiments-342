import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

BATCH_SIZE = 512
EPOCHS = 100
LR = 8e-3
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print('using', DEVICE)

training_data = datasets.CIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

print(next(iter(train_dataloader))[1])

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32*32*32//4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    def forward(self, input):
        logits = self.model(input)
        return logits

model = CustomCNN().to(DEVICE)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

train_loss_history = []
test_loss_history = []
test_auc_history = []
test_f1_history = []

print("Epoch train_loss test_loss test_auc test_f1")
for epoch in range(1, EPOCHS+1):
    train_loss = 0
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    test_loss = 0
    labels = []
    softmax_preds = []
    preds = []
    model.eval()
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)
        test_loss += loss.item()
        labels.extend(y.cpu().numpy())
        softmax_preds.extend(nn.Softmax(-1)(pred).cpu().detach().tolist())
        preds.extend(pred.argmax(-1).cpu().detach().tolist())
    auc = roc_auc_score(labels, softmax_preds, multi_class='ovr')
    f1 = f1_score(labels, preds, average='macro')
    print(epoch, train_loss / len(training_data), test_loss / len(test_data), auc, f1)
    train_loss_history.append(train_loss / len(training_data))
    test_loss_history.append(test_loss / len(test_data))
    test_auc_history.append(auc)
    test_f1_history.append(f1)

plt.plot(test_auc_history)
plt.plot(test_f1_history)
plt.ylabel('some numbers')
plt.show()

plt.plot(train_loss_history)
plt.plot(test_loss_history)
plt.ylabel('some numbers')
plt.show()
