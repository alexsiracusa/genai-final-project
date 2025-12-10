import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from util import add_noise
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


X = torch.tensor(np.load('./data/faces_vae.npy')).flatten(start_dim=1) / 255

class NoisyFaceDataset(Dataset):
    def __init__(self, X, seq_len=9):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        noise = torch.rand(1)
        seq = self.X[idx:idx + self.seq_len].clone()
        seq = add_noise(seq, noise)

        return seq, noise

# Hyperparameters
epochs = 5
batch_size = 128
learning_rate = 1e-4
seq_len = 9

num_layers = 2
hidden_size = 585

# Train test split
dataset = NoisyFaceDataset(X, seq_len=seq_len)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class NoiseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Predict average noise

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(1)


input_size = X.shape[1]
model = NoiseRNN(input_size, hidden_size, num_layers)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for seqs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(seqs)

        loss = criterion(outputs.view(-1, 1), targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * seqs.size(0)

    epoch_loss /= train_size
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")


# Test loss
model.eval()
model.eval()
test_loss = 0
y = np.array([], dtype=float)
y_hat = np.array([], dtype=float)

with torch.no_grad():
    for seqs, targets in test_loader:
        outputs = model(seqs)
        loss = criterion(outputs.view(-1, 1), targets)
        test_loss += loss.item() * seqs.size(0)

        y = np.concatenate((y, targets.detach().numpy().flatten()))
        y_hat = np.concatenate((y_hat, torch.sigmoid(outputs).detach().numpy().flatten()))

test_loss /= test_size
print(f"Test Loss: {test_loss:.6f}")

correlation_coefficient, p_value = pearsonr(y, y_hat)
print(f"Correlation coefficient: {correlation_coefficient:.6f}")
print(f"p-value: {p_value}")

plt.scatter(y, y_hat)
plt.show()

