import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import math


def ppca(X, d=16):
    n, m = X.shape

    mean = X.mean(axis=0)                                   # (m)

    S = (1 / n) * np.dot((X - mean).T, (X - mean))          # (m ,m)

    eigenvalues, eigenvectors = np.linalg.eigh(S)
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]                          # (m)
    eigenvectors = eigenvectors[:, idx].T                   # (m, m)     - rows are the eigenvectors

    var = 1 / (m - d) * np.sum(eigenvalues[d:m])            # (1)
    U_q = eigenvectors[:d].T                                # (m, d)
    L = np.diag(eigenvalues[:d])                            # (d, d)

    W = np.dot(U_q, np.sqrt(L - var * np.eye(d)))           # (m, d)
    M = np.dot(W.T, W) + var * np.eye(d)                    # (d, d)

    # Z = np.linalg.solve(M, np.eye(d)) @ W.T @ (X - mean).T  # (d, n)
    # X_hat = (W @ Z).T + mean                                # (n, m)

    return W, mean, M


def add_noise(X, percent=0.0):
    return (1 - percent) * X + percent * torch.randn_like(X)


def display_images(images, clip=True):
    # images (num_images, rows, cols)
    size = math.ceil(math.sqrt(images.shape[0]))
    fig, axes = plt.subplots(size, size, figsize=(size, size), facecolor="black")

    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i >= images.shape[0]:
            continue
        ax.imshow(images[i], cmap='gray', interpolation='nearest',
            vmin=0 if clip else None,
            vmax=1 if clip else None,
        )

    plt.tight_layout()
    plt.show()
    fig.clf()
    plt.clf()


if __name__ == '__main__':
    X = torch.tensor(np.load('./faces/data/faces_vae.npy')).flatten(start_dim=1) / 255
    X = add_noise(X, percent=0.7).numpy()
    d = 8

    W, mean, M = ppca(X, d)

    print(f'W: {W.shape}')
    print(f'μ: {mean.shape}')
    print(f'M: {M.shape}')

    # Generate images to ensure PPCA is working
    n_images = 9

    Z_generated = np.random.standard_normal(size=(d, n_images))
    X_generated = (W @ Z_generated).T + mean
    images = X_generated.reshape((n_images, 24, 24))

    display_images(images)

    ppca_images = np.insert(W.T, int(d/2), mean, axis=0).reshape(-1, 24, 24)
    display_images(ppca_images, clip=False)

    display_images(X[:9].reshape(9, 24, 24))


    # Code to load cifar10 dataset instead
    # import torchvision
    # import torchvision.transforms as transforms
    # from torch.utils.data import DataLoader
    # from cifar10.generate_datasets import sample_proportions
    #
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((24, 24)),
    #     transforms.Grayscale(),
    #     transforms.Normalize(0.5, 1)
    # ])
    #
    # # Load the training dataset
    # dataset = torchvision.datasets.CIFAR10(root='./cifar10/data', train=True, download=True, transform=transform)
    # loader = DataLoader(dataset, batch_size=len(dataset))
    #
    # X_torch, y_torch = next(iter(loader))
    # X = X_torch.numpy().reshape(X_torch.shape[0], -1)
    # y = y_torch.numpy()






