import argparse

from rich.console import Console

from drishti.anomaly.autoencoder import AnomalyTrainer, NormalSceneDataset

console = Console()


def parse_source(value: str) -> int | str:
    """Webcam index if numeric, otherwise a file path or stream URL."""
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the scene autoencoder on normal footage")
    parser.add_argument("--source", default="1", help="webcam index (default 1), file path, or RTSP URL")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--frames", type=int, default=2000, help="number of frames to collect")
    args = parser.parse_args()

    console.print(
        f"Collecting {args.frames} frames from source {args.source!r}. "
        "Keep the scene looking [bold]normal[/bold] until this finishes."
    )
    source = parse_source(args.source)
    dataset = NormalSceneDataset(source, num_frames=args.frames)

    # After training, the trainer re-opens the same source for held-out frames to set the threshold.
    trainer = AnomalyTrainer(epochs=args.epochs)
    trainer.train(dataset, calibration_source=source)
    console.print(f"\n[bold green]Final threshold: {trainer.threshold:.6f}[/bold green]")


if __name__ == "__main__":
    main()
