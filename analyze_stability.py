import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


def evaluate(path):

    rgb = np.asarray(
        Image.open(path).convert("RGB")
    ).astype(np.float32) / 255.0

    gray = (
        rgb[:, :, 0] * 0.299
        + rgb[:, :, 1] * 0.587
        + rgb[:, :, 2] * 0.114
    )

    # Laplacian variance
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )

    sharpness = float(np.var(lap))

    # Shannon entropy
    hist, _ = np.histogram(
        gray, bins=256, range=(0, 1)
    )

    p = hist[hist > 0] / hist.sum()

    entropy = float(
        -np.sum(p * np.log2(p))
    )

    # Spatial color variation
    color_variance = float(
        np.var(rgb, axis=(0, 1)).mean()
    )

    # Edge density
    dy, dx = np.gradient(gray)

    gradient = np.sqrt(dx**2 + dy**2)

    edge_density = float(
        np.mean(gradient > 0.05)
    )

    return [
        sharpness,
        entropy,
        color_variance,
        edge_density
    ]


names = [
    "Sharpness",
    "Entropy",
    "Color variance",
    "Edge density"
]

experiments = sorted(
    Path("benchmarks").glob("stability_*")
)

if not experiments:
    raise SystemExit("No stability experiments found.")

fig, axes = plt.subplots(
    2, 2, figsize=(13, 9)
)

for folder in experiments:

    paths = sorted(
        (folder / "frames").glob("*.png")
    )

    if not paths:
        continue

    data = np.asarray([
        evaluate(p) for p in paths
    ])

    # Normalize against the first generated frames.
    reference = np.median(data[:5], axis=0)

    normalized = data / np.maximum(
        reference, 1e-8
    )

    collapse = None
    bad_count = 0

    for i, row in enumerate(normalized):

        # Heuristic collapse criterion:
        # at least two metrics below 30%
        # for five consecutive frames.

        if np.sum(row < 0.30) >= 2:
            bad_count += 1
        else:
            bad_count = 0

        if bad_count >= 5:
            collapse = i - 3
            break

    print("\nExperiment:", folder.name)
    print("Total frames:", len(data))
    print("Collapse frame:", collapse)

    with open(
        folder / "stability.csv",
        "w", newline=""
    ) as f:

        writer = csv.writer(f)
        writer.writerow(["frame"] + names)

        for i, row in enumerate(data):
            writer.writerow([i + 1] + row.tolist())

    for j, ax in enumerate(axes.flat):

        ax.plot(
            np.arange(1, len(data) + 1),
            normalized[:, j],
            label=folder.name
        )

        ax.axhline(
            0.30,
            linestyle="--",
            alpha=0.3
        )

        ax.set_title(names[j])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Relative to initial")
        ax.legend()
        ax.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    "benchmarks/stability_comparison.png",
    dpi=150
)

print("\nSaved stability comparison.")
