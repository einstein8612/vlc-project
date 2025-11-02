from torch import nn, optim
import torch

from timegan import TimeGAN
from dataset import InMemoryCNNLSTMDataset
from torch.utils.data import DataLoader

DEVICE = 'cuda'

dataset = InMemoryCNNLSTMDataset(root_dir="preprocessed_data", device=DEVICE)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

timegan = TimeGAN(feature_dim=4, device=DEVICE)
timegan.fit(dataloader, epochs=500)

timegan.save("timegan_gesture_model.pth")

synthetic_data = timegan.generate_synthetic(num_samples=100, seq_len=100)
print(synthetic_data.shape)
torch.save(synthetic_data, "synthetic_gesture_data.pt")
