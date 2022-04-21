import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import torch


class FashionDataset(Dataset):
    meta_data = {"img_vector_size": 784, "img_dim": (28, 28),
                 "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                             "Ankle boot"]}

    def __init__(self, csv_pth, normalized=False, n_rows=None):
        df = pd.read_csv(csv_pth, header=None, skiprows=1, nrows=n_rows)
        self.size = len(df)
        self.labels = np.asarray(df.iloc[:, 0])
        self.img_vectors = np.asarray(df.iloc[:, 1:], dtype=float)

        if normalized:
            self.normalize_data()

    def normalize_data(self):
        means = self.img_vectors.mean(axis=1)
        stds = self.img_vectors.std(axis=1)
        for i in range(self.size):
            self.img_vectors[i] -= means[i]
            self.img_vectors[i] /= stds[i]

    def __getitem__(self, idx):
        return torch.tensor(self.img_vectors[idx], dtype=torch.float), torch.tensor(self.labels[idx])

    def __len__(self):
        return self.size
