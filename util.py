import numpy as np
import torch
import math
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    X = torch.tensor(np.load('./data/faces_vae.npy')).flatten(start_dim=1) / 255
    X = add_noise(X, percent=0.5).numpy()
    d = 8

    W, mean, M = ppca(X, d)

    print(f'W: {W.shape}')
    print(f'μ: {mean.shape}')
    print(f'M: {M.shape}')

    # Generate images to ensure PPCA is working
    n_images = 25

    Z_generated = np.random.standard_normal(size=(d, n_images))
    X_generated = (W @ Z_generated).T + mean
    images = X_generated.reshape((n_images, 24, 24))

    size = int(math.sqrt(n_images))
    fig, axes = plt.subplots(size, size, figsize=(size, size), facecolor="black")

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    fig.clf()
    plt.clf()






