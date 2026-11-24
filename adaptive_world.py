import time
import json
from pathlib import Path

import numpy as np
import torch

from interactive_bench import InteractiveOasis
from utils import ACTION_KEYS


ACTION_INDEX = {
    name: i for i, name in enumerate(ACTION_KEYS)
}


# ==========================================
# Module 1: World Stability Monitor
# ==========================================

class StabilityMonitor:

    def __init__(self):
        self.reference_frames = []
        self.reference = None

    def metrics(self, frame):

        rgb = frame.astype(np.float32) / 255.0

        gray = (
            rgb[:, :, 0] * 0.299
            + rgb[:, :, 1] * 0.587
            + rgb[:, :, 2] * 0.114
        )

        lap = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4 * gray[1:-1, 1:-1]
        )

        sharpness = np.var(lap)

        hist, _ = np.histogram(
            gray, bins=256, range=(0, 1)
        )

        p = hist[hist > 0] / hist.sum()

        entropy = -np.sum(p * np.log2(p))

        color_variance = np.var(
            rgb, axis=(0, 1)
        ).mean()

        dy, dx = np.gradient(gray)

        edge_density = np.mean(
            np.sqrt(dx**2 + dy**2) > 0.05
        )

        return np.asarray([
            sharpness,
            entropy,
            color_variance,
            edge_density
        ], dtype=np.float32)

    def evaluate(self, metrics):

        if self.reference is None:
            return 1.0, False

        ratios = metrics / np.maximum(
            self.reference, 1e-8
        )

        score = float(
            np.clip(ratios, 0, 1).mean()
        )

        degraded = bool(
            np.sum(ratios < 0.30) >= 2
        )

        return score, degraded

    def observe(self, metrics):

        if self.reference is None:

            self.reference_frames.append(metrics)

            if len(self.reference_frames) >= 5:
                self.reference = np.median(
                    self.reference_frames, axis=0
                )


# ==========================================
# Module 2: Motion-Aware Scheduler
# ==========================================

class MotionScheduler:

    def __init__(self, max_context=16):

        self.max_context = max_context
        self.motion_hold = 0

    def moving(self, action):

        a = action.detach().cpu().reshape(-1)

        keys = [
            "forward", "back",
            "left", "right",
            "jump", "sprint", "sneak",
            "attack", "use"
        ]

        movement = any(
            a[ACTION_INDEX[k]].item() > 0.5
            for k in keys
        )

        camera = (
            abs(a[ACTION_INDEX["cameraX"]].item())
            > 0.005
            or
            abs(a[ACTION_INDEX["cameraY"]].item())
            > 0.005
        )

        return movement or camera

    def select(self, action, degraded=False):

        if self.moving(action):
            self.motion_hold = 8

        elif self.motion_hold > 0:
            self.motion_hold -= 1

        if degraded:
            return (
                min(32, self.max_context),
                15,
                "RECOVERY"
            )

        if self.motion_hold > 0:
            return (
                min(16, self.max_context),
                10,
                "MOVING"
            )

        return (
            min(4, self.max_context),
            6,
            "IDLE"
        )


# ==========================================
# Module 3: Adaptive World Model
# ==========================================

class AdaptiveWorld(InteractiveOasis):

    last_instance = None

    def __init__(self, args):

        self.requested_max_context = getattr(
            args, "max_context", 16
        )

        super().__init__(args)

        self.max_context = min(
            self.requested_max_context,
            self.model.max_frames
        )

        AdaptiveWorld.last_instance = self

    def reset(self):

        frame = super().reset()

        self.bank_limit = min(
            32, self.model.max_frames
        )

        self.latent_bank = self.latents.clone()
        self.action_bank = self.actions.clone()

        self.monitor = StabilityMonitor()

        self.scheduler = MotionScheduler(
            min(
                self.requested_max_context,
                self.model.max_frames
            )
        )

        self.bad_streak = 0
        self.recovery_count = 0
        self.recovery_success = 0
        self.records = []

        return frame

    # Generate a candidate from the committed history.
    def generate_candidate(
        self, action, context, steps
    ):

        self.context = context
        self.ddim_steps = steps

        self.noise_range = torch.linspace(
            -1, 999, steps + 1
        ).long().tolist()

        # Rebuild active context from the history bank.
        # The parent step appends the next noisy latent.
        self.latents = self.latent_bank[
            :, -max(1, context - 1):
        ]

        self.actions = self.action_bank[
            :, -max(1, context - 1):
        ]

        frame = super().step(action)

        latest_latent = (
            self.latents[:, -1:].clone()
        )

        metrics = self.monitor.metrics(frame)

        score, degraded = (
            self.monitor.evaluate(metrics)
        )

        return {
            "frame": frame,
            "latent": latest_latent,
            "metrics": metrics,
            "score": score,
            "degraded": degraded,
        }

    @torch.inference_mode()
    def step(self, action):

        start = time.perf_counter()

        committed_frame_count = self.frame_count

        context, steps, mode = (
            self.scheduler.select(
                action,
                degraded=self.bad_streak >= 2
            )
        )

        candidate = self.generate_candidate(
            action, context, steps
        )

        initial_score = candidate["score"]

        retried = False
        recovered = False

        # ==================================
        # Module 4: Candidate Recovery
        # ==================================

        if candidate["degraded"]:

            self.recovery_count += 1
            retried = True

            # Roll back the uncommitted candidate.
            self.frame_count = committed_frame_count

            retry_context = min(
                32, self.scheduler.max_context
            )

            retry_steps = 15

            retry = self.generate_candidate(
                action,
                retry_context,
                retry_steps
            )

            # Prefer a non-degraded candidate;
            # otherwise select the higher score.
            if (
                not retry["degraded"]
                or retry["score"] > candidate["score"]
            ):
                candidate = retry
                context = retry_context
                steps = retry_steps

            if not candidate["degraded"]:
                self.recovery_success += 1
                recovered = True

            mode = "RECOVERY"

        # ==================================
        # Module 5: Commit World State
        # ==================================

        self.latent_bank = torch.cat(
            [
                self.latent_bank,
                candidate["latent"]
            ],
            dim=1
        )[:, -self.bank_limit:].contiguous()

        self.action_bank = torch.cat(
            [
                self.action_bank,
                action.to(self.action_bank.device)
            ],
            dim=1
        )[:, -self.bank_limit:].contiguous()

        self.latents = self.latent_bank
        self.actions = self.action_bank

        self.frame_count = (
            committed_frame_count + 1
        )

        self.monitor.observe(
            candidate["metrics"]
        )

        if candidate["degraded"]:
            self.bad_streak += 1
        else:
            self.bad_streak = 0

        elapsed = (
            time.perf_counter() - start
        )

        record = {
            "frame": self.frame_count,
            "mode": mode,
            "context": context,
            "ddim_steps": steps,
            "latency_ms": elapsed * 1000,
            "initial_score": initial_score,
            "score": candidate["score"],
            "degraded": candidate["degraded"],
            "retried": retried,
            "recovered": recovered,
            "sharpness": float(
                candidate["metrics"][0]
            ),
            "entropy": float(
                candidate["metrics"][1]
            ),
            "color_variance": float(
                candidate["metrics"][2]
            ),
            "edge_density": float(
                candidate["metrics"][3]
            )
        }

        self.records.append(record)

        print(
            f"Frame {self.frame_count:04d} | "
            f"{mode:8s} | "
            f"CTX={context:02d} | "
            f"DDIM={steps:02d} | "
            f"Score={candidate['score']:.3f} | "
            f"FPS={1/max(elapsed, 1e-6):.2f} | "
            f"Recovery={self.recovery_success}/"
            f"{self.recovery_count}"
        )

        return candidate["frame"]

    def save_report(self, path):

        path = Path(path)
        path.parent.mkdir(
            parents=True, exist_ok=True
        )

        records = self.records

        total_ms = sum(
            r["latency_ms"] for r in records
        )

        report = {
            "frames": len(records),

            "fps": (
                len(records) * 1000 / total_ms
                if total_ms else 0
            ),

            "mean_context": (
                float(np.mean([
                    r["context"] for r in records
                ])) if records else 0
            ),

            "mean_ddim_steps": (
                float(np.mean([
                    r["ddim_steps"] for r in records
                ])) if records else 0
            ),

            "degraded_frames": sum(
                r["degraded"] for r in records
            ),

            "recovery_attempts":
                self.recovery_count,

            "recovery_success":
                self.recovery_success,

            "recovery_rate": (
                self.recovery_success
                / self.recovery_count
                if self.recovery_count else None
            ),

            "records": records
        }

        path.write_text(
            json.dumps(report, indent=2)
        )

        print("\n===== Adaptive World Report =====")

        for key, value in report.items():

            if key != "records":
                print(f"{key}: {value}")

        print("Saved:", path)
