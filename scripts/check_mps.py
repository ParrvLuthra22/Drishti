import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time

import torch
from rich.console import Console

console = Console()

SIZE = 2000


def flag(label: str, value: bool) -> None:
    color = "green" if value else "red"
    console.print(f"[{color}]{label}:[/{color}] {value}")


def time_matmul(device: str) -> float:
    """Return matmul time in milliseconds, after a warmup run."""
    x = torch.rand(SIZE, SIZE, dtype=torch.float32, device=device)
    sync = torch.mps.synchronize if device == "mps" else lambda: None

    torch.matmul(x, x)  # warmup (kernel compile / allocation)
    sync()

    start = time.perf_counter()
    torch.matmul(x, x)
    sync()
    return (time.perf_counter() - start) * 1000


def main() -> int:
    console.print(f"[bold]PyTorch version:[/bold] {torch.__version__}")

    available = torch.backends.mps.is_available()
    built = torch.backends.mps.is_built()
    flag("MPS available", available)
    flag("MPS built", built)

    if not available:
        console.print("[bold red]Error: MPS is not available on this machine.[/bold red]")
        return 1

    cpu_ms = time_matmul("cpu")
    mps_ms = time_matmul("mps")
    console.print(f"\nMatmul {SIZE}x{SIZE} float32")
    console.print(f"  CPU: [cyan]{cpu_ms:.2f} ms[/cyan]")
    console.print(f"  MPS: [cyan]{mps_ms:.2f} ms[/cyan]")
    console.print(f"  Speedup: [bold yellow]{cpu_ms / mps_ms:.2f}x[/bold yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
