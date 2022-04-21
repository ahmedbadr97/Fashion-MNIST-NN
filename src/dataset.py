import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset


class FashionDataset(Dataset):
    def __init__(self, csv_pth, normalized=False, n_rows=None):
        df = pd.read_csv(csv_pth, header=None, skiprows=1)
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
        return self.img_vectors[idx], self.labels[idx]

    def __len__(self):
        return self.size