import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt

from model import CNNLSTM
from dataset import InMemoryCNNLSTMDataset

DEVICE = 'cuda'
DATA = 'preprocessed_data'

rand_g = torch.Generator().manual_seed(42)

# Load the full dataset into memory
dataset = InMemoryCNNLSTMDataset(root_dir=DATA, device=DEVICE)
labels = dataset.get_labels()
print(labels)
print(f"✅ Loaded dataset with {len(labels)} gesture IDs.")

# 80/20 train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

_, test_dataset = random_split(dataset, [train_size, test_size], generator=rand_g)

# DataLoaders
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CNNLSTM(num_classes=10, classify=False, device=DEVICE)
model.load("cnn_lstm_arcface_no_zoom_in.pth")

# One-shot evaluation
totals = torch.zeros(len(labels), device=DEVICE)
corrects = torch.zeros(len(labels), device=DEVICE)
model.eval()

# Gather embeddings and labels for the test set
embeddings_done = torch.zeros(len(labels), device=DEVICE)
embeddings = torch.zeros((len(labels), 64), device=DEVICE)  # assuming embedding dim is 64
for X_test, y_test in test_loader:
    with torch.no_grad():
        test_embeddings = model(X_test)
        for i in range(len(y_test)):
            label = y_test[i].item()
            if embeddings_done[label] < 1:  # only take one sample per class
                embeddings[label] = test_embeddings[i]
                embeddings_done[label] += 1
            
        if torch.sum(embeddings_done) >= 9:
            break
embeddings = nn.functional.normalize(embeddings, dim=1)

# Now evaluate
for X_test, y_test in test_loader:
    with torch.no_grad():
        test_embeddings = model(X_test)
        test_embeddings = nn.functional.normalize(test_embeddings, dim=1)

        similarities = torch.matmul(test_embeddings, embeddings.t())
        _, preds = similarities.max(dim=1)

        totals[y_test] += 1
        corrects[y_test] += (preds == y_test).long()

accuracy = corrects.sum() / totals.sum()
print(f"One-shot evaluation accuracy: {accuracy*100:.2f}%")

plt.bar(range(10), (corrects / totals).cpu().numpy())
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.title("One-shot Classification Accuracy per Class")
plt.ylim(0, 1)
plt.show()