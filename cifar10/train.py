import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# Hyperparameters
epochs = 20
batch_size = 128
learning_rate = 1e-4


# Load dataset
dataset = torch.load("./data/cifar_dataset.pt")
tensor_dataset = TensorDataset(dataset["Ws"], dataset["Means"], dataset["proportions"].view(-1, 10))

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
    nn.Linear(128, 10),
)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for Ws, Means, proportions in train_loader:
        input = torch.cat((Ws.flatten(start_dim=1), Means), dim=1).type(torch.float32)
        out = model(input)

        # normalize proportions
        row_sums = proportions.sum(dim=1, keepdim=True)
        proportions = proportions / (row_sums + 1e-8)

        optimizer.zero_grad()
        loss = criterion(out, proportions)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * Ws.size(0)

    epoch_loss /= train_size
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")


# Evaluate model
model.eval()
test_loss = 0
with torch.no_grad():
    for Ws, Means, proportions in test_loader:
        input = torch.cat((Ws.flatten(start_dim=1), Means), dim=1).type(torch.float32)
        out = model(input)
        # out = torch.full(out.shape, 0.1)

        # normalize proportions
        row_sums = proportions.sum(dim=1, keepdim=True)
        proportions = proportions / (row_sums + 1e-8)

        loss = criterion(out, proportions)
        test_loss += loss.item() * Ws.size(0)

test_loss /= test_size
print(f"Test Loss: {test_loss:.6f}")

