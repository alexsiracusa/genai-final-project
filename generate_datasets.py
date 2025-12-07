from util import ppca, add_noise
import torch
import numpy as np
import os


def generate_faces_dataset():
    X = torch.tensor(np.load('./data/faces_vae.npy')).flatten(start_dim=1) / 255

    save_dir = "./data/faces"
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

        noises = torch.rand(batch_size)
        datasets = [add_noise(X, noise) for noise in noises]
        Ws, Means, _ = zip(*[ppca(dataset, d=8) for dataset in datasets])
        Ws, Means = torch.tensor(np.stack(Ws)), torch.tensor(np.stack(Means))

        torch.save({
            "Ws": Ws,
            "Means": Means,
            "noises": noises,
        }, shard_path)

        print(batch)

    # Combine shards
    shards = sorted(os.listdir(save_dir))

    Ws_all = []
    Means_all = []
    noises_all = []

    for shard in shards:
        data = torch.load(f"{save_dir}/{shard}")
        Ws_all.append(data["Ws"])
        Means_all.append(data["Means"])
        noises_all.append(data["noises"])

    Ws_all = torch.cat(Ws_all, dim=0)
    Means_all = torch.cat(Means_all, dim=0)
    noises_all = torch.cat(noises_all, dim=0)

    torch.save(
        {
            "Ws": Ws_all,
            "Means": Means_all,
            "noises": noises_all,
        },
        "./data/faces_dataset.pt",
    )





if __name__ == "__main__":
    generate_faces_dataset()




