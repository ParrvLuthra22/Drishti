import time
from pathlib import Path

import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

console = Console()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "mnist"
MODEL_PATH = ROOT / "models" / "mnist_cnn.pt"

BATCH_SIZE = 64
EPOCHS = 3
LR = 0.001
LOG_EVERY = 100


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            correct += (model(inputs).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[bold]Device:[/bold] {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % LOG_EVERY == 0:
                console.print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"[dim]batch {batch_idx}/{len(train_loader)}[/dim] "
                    f"loss: [cyan]{loss.item():.4f}[/cyan]"
                )
        console.print(
            f"[bold green]Epoch {epoch} average loss: "
            f"{running_loss / len(train_loader):.4f}[/bold green]"
        )
    elapsed = time.perf_counter() - start

    accuracy = evaluate(model, test_loader, device)
    console.print(f"\n[bold]Test accuracy:[/bold] [yellow]{accuracy * 100:.2f}%[/yellow]")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    console.print(f"Saved model to {MODEL_PATH.relative_to(ROOT)}")
    console.print(f"[bold]Total training time:[/bold] {elapsed:.1f}s")


if __name__ == "__main__":
    main()
