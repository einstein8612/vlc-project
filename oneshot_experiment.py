import argparse
import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from arcface import ArcFaceLoss

from model import CNNLSTM
from dataset import InMemoryCNNLSTMDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_loss(losses, save_path):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, save_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format=fmt)
    plt.title(title)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(save_path)
    plt.close()
    return cm.tolist()  # return as list for JSON saving

def evaluate(model, dataloader, labels):
    model.eval()
    all_preds, all_labels = [], []

    # Gather embeddings and labels for the test set
    embeddings_done = torch.zeros(len(labels), device=DEVICE)
    embeddings = torch.zeros((len(labels), 64), device=DEVICE)  # assuming embedding dim is 64
    for X_test, y_test in dataloader:
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

    print("✅ Gathered one embedding per class for one-shot evaluation.")

    # Now evaluate
    for X_test, y_test in dataloader:
        with torch.no_grad():
            test_embeddings = model(X_test)
            test_embeddings = nn.functional.normalize(test_embeddings, dim=1)

            similarities = torch.matmul(test_embeddings, embeddings.t())
            _, preds = similarities.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return all_labels, all_preds, accuracy

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_ONESHOT")
    save_dir = os.path.join("saved_runs", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    rand_g = torch.Generator().manual_seed(42)

    # Load dataset
    dataset = InMemoryCNNLSTMDataset(
        root_dir=args.data_dir,
        ignored_gesture_ids=args.ignore_gestures,
        device=DEVICE
    )
    labels = dataset.get_labels()
    print(f"✅ Loaded dataset with {len(labels)} gesture IDs after ignoring {args.ignore_gestures}.")

    # Train/test split
    if args.test_only:
        train_dataset, test_dataset = None, dataset
    else:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=rand_g)
        print(f"✅ Dataset split into {len(train_dataset)} training and {len(test_dataset)} testing samples.")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = CNNLSTM(num_classes=len(labels), classify=False, device=DEVICE)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
        print(f"✅ Loaded model from {args.load_model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = ArcFaceLoss(num_classes=len(labels), embedding_dim=64, margin=0.5, scale=30.0, device=DEVICE)

    losses = []
    if not args.test_only:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        bar = tqdm(range(args.epochs), desc="Training epochs")
        for epoch in bar:
            model.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                preds = model(X)
                loss = criterion(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            bar.set_description(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save model
        model_path = os.path.join(save_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved model to {model_path}")

        # Save loss graph
        plot_loss_path = os.path.join(save_dir, "loss_graph.png")
        plot_loss(losses, plot_loss_path)
        print(f"✅ Saved loss graph to {plot_loss_path}")

    # Evaluate on test set
    y_true, y_pred, accuracy = evaluate(model, test_loader, labels)
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    cm_list = plot_confusion_matrix(y_true, y_pred, labels, cm_path, normalize=False)
    print(f"✅ Saved confusion matrix to {cm_path}")

    # Save results
    results = {
        "dataset": args.data_dir,
        "ignored_gestures": args.ignore_gestures,
        "accuracy": accuracy,
        "losses": losses,
        "confusion_matrix": cm_list,
        "labels": labels,
        "test_distribution": torch.bincount(torch.tensor(y_true)).tolist()
    }
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved results to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-LSTM gesture model.")
    parser.add_argument("--data_dir", type=str, default="dataset/preprocessed_data", help="Directory of dataset")
    parser.add_argument("--ignore_gestures", type=str, nargs="*", default=['triple_tap'], help="Gesture IDs to ignore")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--load_model", type=str, help="Path to a saved model to load")
    parser.add_argument("--test_only", action="store_true", help="Skip training and only evaluate")
    args = parser.parse_args()

    main(args)
