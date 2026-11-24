import json
import csv
from pathlib import Path

results = []

for path in sorted(
    Path("benchmarks").glob("*/metrics.json")
):
    with open(path) as f:
        results.append(json.load(f))

if not results:
    raise SystemExit("No benchmark results found.")

baseline = next(
    (
        r for r in results
        if r["experiment"] == "baseline"
    ),
    None
)

columns = [
    "experiment",
    "precision",
    "backend",
    "context",
    "ddim_steps",
    "fps",
    "mean_latency_ms",
    "p95_latency_ms",
    "peak_vram_allocated_gib",
    "peak_vram_reserved_gib",
    "process_peak_ram_gib",
    "mean_generated_sharpness",
    "speedup",
    "vram_reduction_pct"
]

for r in results:

    if baseline:

        r["speedup"] = (
            r["fps"] / baseline["fps"]
        )

        r["vram_reduction_pct"] = (
            1 - r["peak_vram_allocated_gib"]
            / baseline["peak_vram_allocated_gib"]
        ) * 100

    else:
        r["speedup"] = None
        r["vram_reduction_pct"] = None

with open(
    "benchmarks/summary.csv",
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns,
        extrasaction="ignore"
    )

    writer.writeheader()

    for r in results:
        writer.writerow(r)

print(
    f"{'Experiment':<16}"
    f"{'FPS':>9}"
    f"{'Latency(ms)':>15}"
    f"{'VRAM(GiB)':>12}"
    f"{'RAM(GiB)':>12}"
    f"{'Sharpness':>12}"
    f"{'Speedup':>10}"
)

print("-" * 86)

for r in results:

    speedup = (
        f"{r['speedup']:.2f}x"
        if r["speedup"] is not None
        else "N/A"
    )

    print(
        f"{r['experiment']:<16}"
        f"{r['fps']:>9.3f}"
        f"{r['mean_latency_ms']:>15.2f}"
        f"{r['peak_vram_allocated_gib']:>12.2f}"
        f"{r['process_peak_ram_gib']:>12.2f}"
        f"{r['mean_generated_sharpness']:>12.2f}"
        f"{speedup:>10}"
    )

print("\nSaved: benchmarks/summary.csv")
