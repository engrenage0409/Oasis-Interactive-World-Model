import argparse
import time
import numpy as np
import pygame
import torch

from torch import autocast
from einops import rearrange
from safetensors.torch import load_model

from dit import DiT_models
from vae import VAE_models
from utils import (
    load_prompt,
    sigmoid_beta_schedule,
    ACTION_KEYS,
)


DEVICE = "cuda:0"
SCALING_FACTOR = 0.07843137255

ACTION_INDEX = {
    key: i for i, key in enumerate(ACTION_KEYS)
}


class InteractiveOasis:

    def __init__(self, args):

        self.args = args

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        print("Loading Oasis 500M...")

        self.model = DiT_models["DiT-S/2"]()
        load_model(self.model, args.oasis_ckpt)
        self.model = self.model.to(device=DEVICE, dtype=torch.float16 if args.precision == 'fp16' else torch.float32).eval()

        print("Loading ViT VAE...")

        self.vae = VAE_models[
            "vit-l-20-shallow-encoder"
        ]()

        load_model(self.vae, args.vae_ckpt)
        self.vae = self.vae.to(device=DEVICE, dtype=torch.float16 if args.precision == 'fp16' else torch.float32).eval()

        self.context = min(
            max(1, args.context),
            self.model.max_frames
        )

        self.ddim_steps = args.ddim_steps
        self.stabilization_level = 15

        self.noise_range = torch.linspace(
            -1, 999, self.ddim_steps + 1
        ).long().tolist()

        betas = sigmoid_beta_schedule(1000)
        betas = betas.float().to(DEVICE)

        alphas = 1.0 - betas

        self.alphas_cumprod = rearrange(
            torch.cumprod(alphas, dim=0),
            "t -> t 1 1 1"
        )

        self.reset()

        print("Model loaded.")
        print("Model max frames:", self.model.max_frames)
        print("Context window:", self.context)
        print("DDIM steps:", self.ddim_steps)
        print("Action dimensions:", len(ACTION_KEYS))

    @torch.inference_mode()
    def reset(self):

        prompt = load_prompt(
            self.args.prompt_path,
            n_prompt_frames=1
        ).to(DEVICE)

        _, _, _, h, w = prompt.shape

        pixels = rearrange(
            prompt,
            "b t c h w -> (b t) c h w"
        )

        with autocast("cuda", dtype=torch.float16):
            latent = self.vae.encode(
                pixels * 2 - 1
            ).mean * SCALING_FACTOR

        self.latents = rearrange(
            latent,
            "(b t) (h w) c -> b t c h w",
            t=1,
            h=h // self.vae.patch_size,
            w=w // self.vae.patch_size
        ).float()

        self.actions = torch.zeros(
            1, 1, len(ACTION_KEYS),
            device=DEVICE
        )

        self.frame_count = 0

        return (
            prompt[0, 0]
            .permute(1, 2, 0)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def step(self, action):

        action = action.to(DEVICE)

        # Add a noisy latent for the next frame.
        noise = torch.randn(
            (1, 1, *self.latents.shape[-3:]),
            device=DEVICE
        ).clamp(-20, 20)

        self.latents = torch.cat(
            [self.latents, noise],
            dim=1
        )[:, -self.context:]

        self.actions = torch.cat(
            [self.actions, action],
            dim=1
        )[:, -self.context:]

        length = self.latents.shape[1]

        # DDIM sampling.
        for noise_idx in reversed(
            range(1, self.ddim_steps + 1)
        ):

            t_ctx = torch.full(
                (1, length - 1),
                self.stabilization_level - 1,
                dtype=torch.long,
                device=DEVICE
            )

            t = torch.full(
                (1, 1),
                self.noise_range[noise_idx],
                dtype=torch.long,
                device=DEVICE
            )

            t_next = torch.full(
                (1, 1),
                self.noise_range[noise_idx - 1],
                dtype=torch.long,
                device=DEVICE
            )

            t_next = torch.where(
                t_next < 0, t, t_next
            )

            t = torch.cat([t_ctx, t], dim=1)

            t_next = torch.cat(
                [t_ctx, t_next], dim=1
            )

            x_curr = self.latents

            with autocast(
                "cuda", dtype=torch.float16
            ):
                v = self.model(
                    x_curr,
                    t,
                    self.actions
                )

            alpha = self.alphas_cumprod[t]

            x_start = (
                alpha.sqrt() * x_curr
                - (1 - alpha).sqrt() * v
            )

            x_noise = (
                (1 / alpha).sqrt() * x_curr
                - x_start
            ) / (
                1 / alpha - 1
            ).sqrt()

            alpha_next = (
                self.alphas_cumprod[t_next]
            )

            alpha_next[:, :-1] = 1

            if noise_idx == 1:
                alpha_next[:, -1:] = 1

            x_pred = (
                alpha_next.sqrt() * x_start
                + (1 - alpha_next).sqrt() * x_noise
            )

            self.latents[:, -1:] = x_pred[:, -1:]

        # Decode only the latest generated frame.
        latest = rearrange(
            self.latents[:, -1:],
            "b t c h w -> (b t) (h w) c"
        )

        with autocast(
            "cuda", dtype=torch.float16
        ):
            frame = (
                self.vae.decode(
                    latest / SCALING_FACTOR
                ) + 1
            ) / 2

        frame = (
            frame[0]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )

        self.frame_count += 1

        return frame


def read_action(sensitivity, buffered_keys=None):
    buffered_keys = buffered_keys or set()

    keys = pygame.key.get_pressed()
    buttons = pygame.mouse.get_pressed()
    dx, dy = pygame.mouse.get_rel()

    action = torch.zeros(
        1, 1, len(ACTION_KEYS),
        dtype=torch.float32
    )

    key_mapping = {
        "forward": pygame.K_w,
        "back": pygame.K_s,
        "left": pygame.K_a,
        "right": pygame.K_d,
        "jump": pygame.K_SPACE,
        "sneak": pygame.K_LSHIFT,
        "sprint": pygame.K_LCTRL,
        "inventory": pygame.K_e,
        "drop": pygame.K_q,
        "swapHands": pygame.K_f,
    }

    for name, key in key_mapping.items():
        action[0, 0, ACTION_INDEX[name]] = (
            float(keys[key] or key in buffered_keys)
        )

    for i in range(1, 10):
        name = f"hotbar.{i}"

        action[0, 0, ACTION_INDEX[name]] = (
            float(keys[pygame.K_0 + i])
        )

    action[0, 0, ACTION_INDEX["attack"]] = float(
        buttons[0]
    )

    action[0, 0, ACTION_INDEX["use"]] = float(
        buttons[2]
    )

    action[0, 0, ACTION_INDEX["pickItem"]] = float(
        buttons[1]
    )

    # VPT-style camera order: pitch, yaw.
    # Values are normalized to [-1, 1].
    action[0, 0, ACTION_INDEX["cameraX"]] = float(
        np.clip(dy * sensitivity, -1, 1)
    )

    action[0, 0, ACTION_INDEX["cameraY"]] = float(
        np.clip(dx * sensitivity, -1, 1)
    )

    return action


def display_frame(screen, frame):

    surface = pygame.surfarray.make_surface(
        np.transpose(frame, (1, 0, 2))
    )

    surface = pygame.transform.scale(
        surface, screen.get_size()
    )

    screen.blit(surface, (0, 0))


def main(args):

    game = InteractiveOasis(args)

    pygame.init()

    screen = pygame.display.set_mode(
        (960, 540)
    )

    pygame.display.set_caption(
        "Interactive Oasis"
    )

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    frame = game.reset()

    display_frame(screen, frame)
    pygame.display.flip()

    pygame.mouse.get_rel()

    torch.cuda.reset_peak_memory_stats()

    running = True
    paused = False
    buffered_keys = set()
    fps = 0.0
    elapsed = 0.0

    print("WASD: Move")
    print("Mouse: Look")
    print("Space: Jump")
    print("P: Pause")
    print("R: Reset")
    print("ESC: Exit")

    try:

        while running:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if not paused:
                        buffered_keys.add(event.key)

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        frame = game.reset()

                    elif event.key == pygame.K_p:
                        paused = not paused

                        pygame.event.set_grab(
                            not paused
                        )

                        pygame.mouse.set_visible(
                            paused
                        )

                        pygame.mouse.get_rel()

            if not running:
                break

            if not paused:

                action = read_action(
                    args.sensitivity, buffered_keys
                )
                buffered_keys.clear()

                start = time.perf_counter()

                frame = game.step(action)

                elapsed = (
                    time.perf_counter() - start
                )

                fps = 1.0 / max(elapsed, 1e-6)

            display_frame(screen, frame)

            memory = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            status = (
                f"Frame: {game.frame_count} | "
                f"FPS: {fps:.2f} | "
                f"Latency: {elapsed:.2f}s | "
                f"VRAM: {memory:.2f} GiB"
            )

            if paused:
                status = "PAUSED | " + status

            text = font.render(
                status, True, (255, 255, 255)
            )

            pygame.draw.rect(
                screen,
                (0, 0, 0),
                (0, 0, screen.get_width(), 30)
            )

            screen.blit(text, (10, 6))

            pygame.display.flip()

            clock.tick(20)

    finally:
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        pygame.quit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--context",
        type=int,
        default=8
    )

    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=4
    )

    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.015
    )

    main(parser.parse_args())
