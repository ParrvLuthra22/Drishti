import os
import random
import time

from rich.console import Console

from drishti.actions.dataset import RWF2000Dataset

console = Console()

ROOT = os.environ.get("RWF2000_ROOT", "~/Downloads/RWF-2000")

# ImageNet-normalised pixels in [0, 1] map to [(0-mean)/std, (1-mean)/std] per channel:
# R: [-2.118, 2.249], G: [-2.036, 2.429], B: [-1.804, 2.640].
EXPECTED_MIN, EXPECTED_MAX = -2.12, 2.65


def main() -> int:
    train = RWF2000Dataset(ROOT, split="train")
    val = RWF2000Dataset(ROOT, split="val")
    console.print(f"train class counts: {train.class_counts()}")
    console.print(f"val class counts:   {val.class_counts()}")

    video, label = train[0]
    console.print("\n[bold]train[0][/bold]")
    console.print(f"  shape: {tuple(video.shape)}")
    console.print(f"  dtype: {video.dtype}")
    console.print(f"  min / max: {video.min().item():.3f} / {video.max().item():.3f}")
    console.print(f"  label: {label} ({'Fight' if label == 1 else 'NonFight'})")

    indices = random.Random(0).sample(range(len(train)), 5)
    timings = []
    for i in indices:
        start = time.perf_counter()
        train[i]
        timings.append((time.perf_counter() - start) * 1000)
    console.print(f"\n__getitem__ on 5 random clips: {[f'{t:.0f}' for t in timings]} ms")
    console.print(f"  average: [bold]{sum(timings) / len(timings):.1f} ms/clip[/bold]")

    in_range = EXPECTED_MIN <= video.min().item() and video.max().item() <= EXPECTED_MAX
    loaded = bool(video.abs().sum() > 0)
    console.print(
        f"\nvalues within ImageNet-normalised range [{EXPECTED_MIN}, {EXPECTED_MAX}]: "
        f"{'[green]yes[/green]' if in_range else '[red]NO[/red]'}"
    )
    console.print(f"clip decoded (not the zero fallback): {'[green]yes[/green]' if loaded else '[red]NO[/red]'}")
    return 0 if in_range and loaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
