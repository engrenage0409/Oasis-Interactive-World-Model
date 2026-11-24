import argparse

from torch.nn.attention import (
    sdpa_kernel,
    SDPBackend
)

from interactive_bench import main


MODES = {
    "quality": (8, 10),
    "balanced": (8, 6),
    "fast": (4, 6),
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="fast"
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
        "--sensitivity",
        type=float,
        default=0.015
    )

    args = parser.parse_args()

    args.context, args.ddim_steps = MODES[
        args.mode
    ]

    args.precision = "fp16"

    print("===== Optimized Interactive Oasis =====")
    print("Mode:", args.mode)
    print("Precision: FP16")
    print("Attention: Memory Efficient")
    print("Context:", args.context)
    print("DDIM Steps:", args.ddim_steps)

    with sdpa_kernel(
        SDPBackend.EFFICIENT_ATTENTION
    ):
        main(args)
