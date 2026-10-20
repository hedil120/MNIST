"""Entraînement d'un CNN sur le dataset MNIST avec PyTorch (CPU).

Usage :
    python train.py [--epochs N] [--batch-size N] [--lr F] [--out MODEL_PATH]

Sorties :
    - modèle entraîné sauvegardé (state_dict PyTorch)
    - précision sur le jeu de test à la fin de chaque époque
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# Modèle : petit CNN (Conv -> Conv -> FC -> FC)
# --------------------------------------------------------------------------- #
class MNISTNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


# --------------------------------------------------------------------------- #
# Entraînement / évaluation
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        correct += pred.eq(target).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne un CNN sur MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="mnist_cnn.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Dataset : normalisation standard de MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    print(f"Train : {len(train_set)} images | Test : {len(test_set)} images")

    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        elapsed = time.time() - start
        print(
            f"Époque {epoch}/{args.epochs} — "
            f"perte train : {train_loss:.4f} | "
            f"précision test : {test_acc:.2%} | "
            f"{elapsed:.1f}s"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_acc": best_acc,
            "epochs": args.epochs,
            "args": vars(args),
        },
        args.out,
    )
    print(f"\nMeilleure précision test : {best_acc:.2%}")
    print(f"Modèle sauvegardé → {args.out}")


if __name__ == "__main__":
    main()
