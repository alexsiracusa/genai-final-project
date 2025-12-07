import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(5184, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
)

learning_rate = 1e-4

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# input = torch.cat((Ws.flatten(start_dim=1), Means), dim=1).type(torch.float32)

