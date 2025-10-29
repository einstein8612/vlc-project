import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class InMemoryCNNLSTMDataset(Dataset):
    def __init__(self, root_dir, device='cpu'):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self.device = device

        # map folder names to numeric labels
        id_folders = sorted(os.listdir(root_dir))
        self.id2label = {id_name: i for i, id_name in enumerate(id_folders)}

        # loop through all id folders
        for id_name in tqdm(id_folders, desc="Loading data"):
            id_path = os.path.join(root_dir, id_name)
            if not os.path.isdir(id_path):
                continue

            label = self.id2label[id_name]

            # each .npy file = one sample
            npy_files = sorted([
                os.path.join(id_path, f)
                for f in os.listdir(id_path)
                if f.endswith(".npy")
            ])

            for f in npy_files:
                data = np.load(f)  # shape: (T, 4)
                self.samples.append(torch.tensor(data, dtype=torch.float32, device=self.device))
                self.labels.append(torch.tensor(label, dtype=torch.long, device=self.device))

        print(f"✅ Loaded {len(self.samples)} samples into memory "
              f"from {len(self.id2label)} IDs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
