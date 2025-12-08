import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


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
    nn.Linear(5184, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for Ws, Means, noises in train_loader:
        input = torch.cat((Ws.flatten(start_dim=1), Means), dim=1).type(torch.float32)
        out = model(input)

        optimizer.zero_grad()
        loss = criterion(out, noises.view(-1, 1))
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")


# Evaluate model
model.eval()
total_loss = 0.0
count = 0

with torch.no_grad():
    for Ws, Means, noises in test_loader:
        input = torch.cat((Ws.flatten(start_dim=1), Means), dim=1).type(torch.float32)
        out = model(input)
        loss = criterion(out, noises)

        total_loss += loss.item() * Ws.size(0)
        count += Ws.size(0)

print(f"Test Loss: {total_loss / count}")

