import torch
from pathlib import Path
from utils import ACTION_KEYS

output = Path("test_actions")
output.mkdir(exist_ok=True)

NUM_FRAMES = 48

experiments = {
    "idle": [],
    "forward": ["forward"],
    "backward": ["back"],
    "left": ["left"],
    "right": ["right"],
    "sprint": ["forward", "sprint"],
}

for name, keys in experiments.items():

    actions = torch.zeros(
        NUM_FRAMES,
        len(ACTION_KEYS)
    )

    for key in keys:
        index = ACTION_KEYS.index(key)
        actions[:, index] = 1.0

    path = output / f"{name}.one_hot_actions.pt"

    torch.save(actions, path)

    print(f"{name}: {actions.shape}")
    print(f"Active actions: {keys}")
    print(f"Saved: {path}\n")
