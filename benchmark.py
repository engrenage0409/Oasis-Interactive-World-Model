import argparse
import json
import time
import resource
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video

from torch.nn.attention import (
    sdpa_kernel,
    SDPBackend
)

from interactive_bench import InteractiveOasis
from utils import load_actions


BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}


def backend_context(name):
    if name == "auto":
        return nullcontext()

    return sdpa_kernel(BACKENDS[name])


def get_available_ram():
    # Available memory inside WSL, not Windows host RAM.
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024**2

    return None


def get_peak_ram():
    # Linux process peak resident memory.
    return resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024**2


def get_sharpness(frame):
    # Variance of discrete grayscale Laplacian.
    gray = (
        frame[:, :, 0].astype(np.float32) * 0.299
        + frame[:, :, 1].astype(np.float32) * 0.587
        + frame[:, :, 2].astype(np.float32) * 0.114
    )

    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )

    return float(np.var(laplacian))


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):

    output_dir = Path("benchmarks") / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Experiment =====")
    print(vars(args))

    reset_seed(args.seed)

    # Load model only once per experiment.
    game = InteractiveOasis(args)

    actions = load_actions(args.actions_path)

    # load_actions prepends an initial zero action.
    # The first generated frame uses action index 1.
    actions = actions[:, 1:]

    if actions.shape[1] < args.frames:
        raise ValueError(
            "Not enough actions in the action file."
        )

    initial_frame = game.reset()

    # -----------------------------
    # 1. Warmup
    # -----------------------------
    print("\nWarming up...")

    with backend_context(args.backend):
        for i in range(args.warmup):
            action = actions[
                :, i % actions.shape[1]:
                i % actions.shape[1] + 1
            ]

            game.step(action)

    torch.cuda.synchronize()

    # Reset after warmup so all experiments
    # begin from the same initial state.
    reset_seed(args.seed)
    initial_frame = game.reset()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    frames = []
    sharpness_scores = []
    ram_samples = []

    initial_available_ram = get_available_ram()

    # -----------------------------
    # 2. Measured inference
    # -----------------------------
    print("\nRunning benchmark...")

    with backend_context(args.backend):

        for i in range(args.frames):

            action = actions[:, i:i + 1]

            torch.cuda.synchronize()
            start = time.perf_counter()

            frame = game.step(action)

            torch.cuda.synchronize()

            latency = time.perf_counter() - start

            latencies.append(latency)
            frames.append(frame.copy())
            sharpness_scores.append(
                get_sharpness(frame)
            )

            available_ram = get_available_ram()

            if available_ram is not None:
                ram_samples.append(available_ram)

            peak_vram = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            print(
                f"Frame {i + 1:03d}/{args.frames} | "
                f"Latency {latency:.3f}s | "
                f"FPS {1 / latency:.3f} | "
                f"VRAM {peak_vram:.2f} GiB | "
                f"Available RAM {available_ram:.2f} GiB"
            )

    latencies = np.asarray(latencies)

    # Exclude image/video saving and profiler overhead
    # from inference measurements.
    results = {
        "experiment": args.name,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": args.seed,
        "precision": args.precision,
        "backend": args.backend,
        "context": game.context,
        "ddim_steps": args.ddim_steps,
        "frames": args.frames,
        "warmup_frames": args.warmup,

        "fps": float(
            len(latencies) / latencies.sum()
        ),

        "mean_latency_ms": float(
            latencies.mean() * 1000
        ),

        "p50_latency_ms": float(
            np.percentile(latencies, 50) * 1000
        ),

        "p95_latency_ms": float(
            np.percentile(latencies, 95) * 1000
        ),

        "total_inference_seconds": float(
            latencies.sum()
        ),

        "peak_vram_allocated_gib": float(
            torch.cuda.max_memory_allocated()
            / 1024**3
        ),

        "peak_vram_reserved_gib": float(
            torch.cuda.max_memory_reserved()
            / 1024**3
        ),

        "process_peak_ram_gib": get_peak_ram(),

        "initial_wsl_available_ram_gib":
            initial_available_ram,

        "min_wsl_available_ram_gib":
            min(ram_samples) if ram_samples else None,

        "initial_sharpness":
            get_sharpness(initial_frame),

        "mean_generated_sharpness": float(
            np.mean(sharpness_scores)
        ),

        "final_frame_sharpness":
            sharpness_scores[-1],

        "frame_latency_ms":
            (latencies * 1000).tolist(),

        "frame_sharpness":
            sharpness_scores,
    }

    # -----------------------------
    # 3. Save experimental results
    # -----------------------------
    with open(
        output_dir / "metrics.json",
        "w"
    ) as f:
        json.dump(results, f, indent=4)

    Image.fromarray(initial_frame).save(
        output_dir / "initial.png"
    )

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(
            frame_dir / f"frame_{i + 1:04d}.png"
        )

    video = torch.from_numpy(
        np.stack([initial_frame] + frames)
    )

    try:
        write_video(
            str(output_dir / "generated.mp4"),
            video,
            fps=20
        )
    except Exception as e:
        print("Video saving failed:", e)
        print("PNG frames are still available.")

    # -----------------------------
    # 4. Optional SDPA profiling
    # -----------------------------
    # Profiling is separate from measured latency.
    if args.profile:

        from torch.profiler import (
            profile,
            ProfilerActivity
        )

        print("\nProfiling Attention kernels...")

        with backend_context(args.backend):

            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                ],
                record_shapes=False,
                profile_memory=False
            ) as prof:

                game.step(actions[:, :1])
                torch.cuda.synchronize()

        attention_ops = []

        for event in prof.key_averages():

            if any(
                key in event.key.lower()
                for key in [
                    "attention",
                    "flash",
                    "fmha",
                    "scaled_dot_product"
                ]
            ):
                attention_ops.append(event.key)

        with open(
            output_dir / "attention_ops.json",
            "w"
        ) as f:
            json.dump(attention_ops, f, indent=4)

        print("\nDetected Attention operators:")

        for op in attention_ops:
            print(op)

    print("\n===== Final Results =====")

    for key, value in results.items():
        if not isinstance(value, list):
            print(f"{key}: {value}")

    print("\nSaved to:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", default="baseline"
    )

    parser.add_argument(
        "--oasis-ckpt",
        default="weights/oasis500m.safetensors"
    )

    parser.add_argument(
        "--vae-ckpt",
        default="weights/vit-l-20.safetensors"
    )

    parser.add_argument(
        "--prompt-path",
        default="sample_data/sample_image_0.png"
    )

    parser.add_argument(
        "--actions-path",
        default=(
            "sample_data/"
            "sample_actions_0.one_hot_actions.pt"
        )
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32"
    )

    parser.add_argument(
        "--backend",
        choices=[
            "auto", "flash",
            "efficient", "cudnn", "math"
        ],
        default="auto"
    )

    parser.add_argument(
        "--context", type=int, default=8
    )

    parser.add_argument(
        "--ddim-steps", type=int, default=10
    )

    parser.add_argument(
        "--frames", type=int, default=24
    )

    parser.add_argument(
        "--warmup", type=int, default=3
    )

    parser.add_argument(
        "--seed", type=int, default=0
    )

    parser.add_argument(
        "--profile", action="store_true"
    )

    main(parser.parse_args())
