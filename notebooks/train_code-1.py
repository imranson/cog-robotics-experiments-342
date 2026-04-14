import json
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_ID = f"cifar100_ep{EPOCHS}_bs{BATCH_SIZE}_lr{LR}"
print('using', DEVICE)

training_data = datasets.CIFAR100(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR100(
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
            nn.Flatten(),
            nn.Linear(512, 10),
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

# history csv
history_df = pd.DataFrame({
    "epoch": list(range(1, EPOCHS + 1)),
    "train_loss": train_loss_history,
    "test_loss": test_loss_history,
    "auc": test_auc_history,
    "f1": test_f1_history,
})
history_df.to_csv(os.path.join(OUTPUT_DIR, f"{RUN_ID}_history.csv"), index=False)

# save best score json
best_idx = history_df["auc"].idxmax()
best_epoch = int(history_df.loc[best_idx, "epoch"])
best_auc = history_df["auc"].max()
best_f1 = float(history_df.loc[best_idx, "f1"])
best_train_loss = float(history_df.loc[best_idx, "train_loss"])
best_test_loss = float(history_df.loc[best_idx, "test_loss"])
with open(os.path.join(OUTPUT_DIR, f"{RUN_ID}_best.json"), "w") as f:
    json.dump({"epoch": best_epoch, "train_loss": best_train_loss, "test_loss": best_test_loss, "auc": best_auc, "f1": best_f1}, f, indent=2)

# save figs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
epochs = history_df["epoch"]
ax1.plot(epochs, history_df["train_loss"], label="Train Loss")
ax1.plot(epochs, history_df["test_loss"], label="Test Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss")
ax1.legend()
ax2.plot(epochs, history_df["auc"], label="AUC")
ax2.plot(epochs, history_df["f1"], label="F1")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.set_title("Metrics")
ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
ax2.annotate(f"Best AUC: {best_auc:.4f} (ep {best_epoch})",
             xy=(best_epoch, best_auc), xytext=(5, -15),
             textcoords="offset points", fontsize=9,
             arrowprops=dict(arrowstyle="->", color="gray"))
ax2.legend()
fig.suptitle(RUN_ID)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"{RUN_ID}_history.png"), dpi=150)
print(f"Saved outputs to {OUTPUT_DIR}/ with prefix {RUN_ID}")
