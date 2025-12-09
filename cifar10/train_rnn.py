import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from generate_datasets import sample_proportions


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((24, 24)),
    transforms.Grayscale(),
    transforms.Normalize(0.5, 1)
])

# Load the training dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=len(dataset))

X_torch, y_torch = next(iter(loader))
X = X_torch.reshape(X_torch.shape[0], -1)
y = y_torch


class ProportionDataset(Dataset):
    def __init__(self, X, y, seq_len=9):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        proportions = torch.rand(10)
        proportions = proportions / (proportions.sum() + 1e-8)
        seq, _ = self.sample_with_prop(self.X, self.y, proportions, self.seq_len)

        return seq, proportions

    def sample_with_prop(self, X, y, props, n):
        classes = torch.unique(y)
        w = torch.zeros_like(y, dtype=float)
        for cls, p in zip(classes, props):
            w[y == cls] = p
        idx = torch.multinomial(w, n, replacement=True)
        return X[idx], y[idx]


# Hyperparameters
epochs = 5
batch_size = 128
learning_rate = 1e-4
num_images = 30

num_layers = 2
hidden_size = 585

# Train test split
dataset = ProportionDataset(X, y, seq_len=num_images)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class ProportionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)  # Predict average noise

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(1)

input_size = X.shape[1]
model = ProportionRNN(input_size, hidden_size, num_layers)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for seqs, targets in train_loader:
        outputs = model(seqs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * seqs.size(0)

    epoch_loss /= train_size
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")


# Test loss
model.eval()
test_loss = 0
with torch.no_grad():
    for seqs, targets in test_loader:
        outputs = model(seqs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * seqs.size(0)

test_loss /= test_size
print(f"Test Loss: {test_loss:.6f}")

