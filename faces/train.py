import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


# Hyperparameters
epochs = 5
batch_size = 128
learning_rate = 1e-4


# Load dataset
dataset = torch.load("./data/faces_dataset.pt")
tensor_dataset = TensorDataset(dataset["Ws"], dataset["Means"], dataset["noises"].view(-1, 1))

train_size = int(0.8 * len(tensor_dataset))
test_size  = len(tensor_dataset) - train_size
train_dataset, test_dataset = random_split(tensor_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


# Define model
model = nn.Sequential(
    nn.Linear(5184, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for Ws, Means, noises in train_loader:
        input = torch.cat((Ws.flatten(start_dim=1), Means), dim=1).type(torch.float32)
        out = model(input)

        optimizer.zero_grad()
        loss = criterion(out, noises.view(-1, 1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * Ws.size(0)

    epoch_loss /= train_size
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")


# Evaluate model
model.eval()
test_loss = 0
y = np.array([], dtype=float)
y_hat = np.array([], dtype=float)

with torch.no_grad():
    for Ws, Means, noises in test_loader:
        input = torch.cat((Ws.flatten(start_dim=1), Means), dim=1).type(torch.float32)
        out = model(input)
        loss = criterion(out, noises)
        test_loss += loss.item() * Ws.size(0)

        y = np.concatenate((y, noises.detach().numpy().flatten()))
        y_hat = np.concatenate((y_hat, torch.sigmoid(out).detach().numpy().flatten()))

test_loss /= test_size
print(f"Test Loss: {test_loss:.6f}")

correlation_coefficient, p_value = pearsonr(y, y_hat)
print(f"Correlation coefficient: {correlation_coefficient:.6f}")
print(f"p-value: {p_value}")

plt.scatter(y, y_hat)
plt.show()

