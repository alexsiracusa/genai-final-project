from util import ppca
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os

import numpy as np


def sample_proportions(X, y, proportions, total_samples=5000):
    classes = np.unique(y)

    proportions = np.array(proportions, dtype=float)
    proportions = proportions / proportions.sum()  # normalize
    samples_per_class = (proportions * total_samples).astype(int)

    X_out_list = []
    y_out_list = []

    for cls, n_samples in zip(classes, samples_per_class):
        class_indices = np.where(y == cls)[0]

        if n_samples > len(class_indices):
            raise ValueError(f"Not enough samples for class {cls}")

        selected = np.random.choice(class_indices, n_samples, replace=False)
        X_out_list.append(X[selected])
        y_out_list.append(y[selected])

    # Concatenate final dataset
    X_out = np.concatenate(X_out_list, axis=0)
    y_out = np.concatenate(y_out_list, axis=0)

    return X_out, y_out


def generate_faces_dataset():
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
    X = X_torch.numpy().reshape(X_torch.shape[0], -1)
    y = y_torch.numpy()

    save_dir = "./data/cifar"
    os.makedirs(save_dir, exist_ok=True)

    num_batches = 200
    batch_size = 100

    # Generate batch shards
    for batch in range(num_batches):
        shard_path = f"{save_dir}/shard_{batch:05d}.pt"

        # Skip if this shard already finished
        if os.path.exists(shard_path):
            print(f"Skipping batch {batch} (already computed)")
            continue

        proportions = [torch.rand(10) for _ in range(batch_size)]

        datasets = [sample_proportions(X, y, p) for p in proportions]
        Ws, Means, _ = zip(*[ppca(dataset, d=8) for dataset, _ in datasets])
        Ws, Means = torch.tensor(np.stack(Ws)), torch.tensor(np.stack(Means))

        torch.save({
            "Ws": Ws,
            "Means": Means,
            "proportions": proportions,
        }, shard_path)

        print(batch)

    # Combine shards
    shards = sorted(os.listdir(save_dir))

    Ws_all = []
    Means_all = []
    proportions_all = []

    for shard in shards:
        data = torch.load(f"{save_dir}/{shard}")
        Ws_all.append(data["Ws"])
        Means_all.append(data["Means"])
        proportions_all.append(torch.stack(data["proportions"]))

    Ws_all = torch.cat(Ws_all, dim=0)
    Means_all = torch.cat(Means_all, dim=0)
    noises_all = torch.cat(proportions_all, dim=0)

    torch.save(
        {
            "Ws": Ws_all,
            "Means": Means_all,
            "proportions": noises_all,
        },
        "./data/cifar_dataset.pt",
    )





if __name__ == "__main__":
    generate_faces_dataset()




