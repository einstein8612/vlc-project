import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import CNNLSTM
from dataset import InMemoryCNNLSTMDataset

from arcface import ArcFaceLoss

DEVICE = 'cuda'
DATA = 'preprocessed_data_no_zoom_in'

rand_g = torch.Generator().manual_seed(42)

# Load the full dataset into memory
dataset = InMemoryCNNLSTMDataset(root_dir=DATA, device=DEVICE)

# 80/20 train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=rand_g)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Dataset split into {len(train_dataset)} training and {len(test_dataset)} testing samples.")

model = CNNLSTM(num_classes=8, classify=False, device=DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = ArcFaceLoss(num_classes=8, embedding_dim=64, margin=0.5, scale=30.0, device=DEVICE)

bar = tqdm(range(50), desc="Training epochs")
for epoch in bar:
    model.train()
    for X, y in train_loader:
        preds = model(X)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    bar.set_description(f"Epoch {epoch+1} loss: {loss.item():.4f}")

model.save("cnn_lstm_arcface_no_zoom_in.pth")
