import argparse

from torch.nn.attention import (
    sdpa_kernel, SDPBackend
)

import interactive_bench
from adaptive_world import AdaptiveWorld


parser = argparse.ArgumentParser()

parser.add_argument(
    "--max-context",
    type=int,
    choices=[16, 32],
    default=16
)

parser.add_argument(
    "--sensitivity",
    type=float,
    default=0.002
)

args = parser.parse_args()

args.oasis_ckpt = (
    "weights/oasis500m.safetensors"
)

args.vae_ckpt = (
    "weights/vit-l-20.safetensors"
)

args.prompt_path = (
    "sample_data/sample_image_0.png"
)

args.context = 16
args.ddim_steps = 10
args.precision = "fp16"

interactive_bench.InteractiveOasis = (
    AdaptiveWorld
)

try:

    with sdpa_kernel(
        SDPBackend.EFFICIENT_ATTENTION
    ):
        interactive_bench.main(args)

finally:

    model = AdaptiveWorld.last_instance

    if model is not None:
        model.save_report(
            "benchmarks/adaptive_interactive.json"
        )
