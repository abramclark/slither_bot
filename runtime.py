from time import time

import numpy as np
import torch
import torch.nn as nn

from model import (
    SAVE_PATH,
    ActorCritic,
    bot_script,
    get_flat,
)


def load_compatible_state_dict(model, state_dict):
    current = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        if key not in current:
            skipped.append(key)
            continue
        if current[key].shape != value.shape:
            skipped.append(f"{key} (ckpt {tuple(value.shape)} != model {tuple(current[key].shape)})")
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return missing, unexpected, skipped


class Runtime:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic().to(self.device)
        try:
            ckpt = torch.load(SAVE_PATH, map_location=self.device)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            missing, unexpected, skipped = load_compatible_state_dict(self.model, state)
            ep = ckpt.get("ep", 0) if isinstance(ckpt, dict) else 0
            steps = ckpt.get("total_steps", 0) if isinstance(ckpt, dict) else 0
            print(f"Resumed from {SAVE_PATH} (ep={ep} steps={steps})")
            if missing:
                print(f"Missing checkpoint keys: {missing}")
            if unexpected:
                print(f"Unexpected checkpoint keys: {unexpected}")
            if skipped:
                print(f"Skipped incompatible checkpoint keys: {skipped}")
            if any("shared.0" in k for k in skipped):
                nn.init.zeros_(self.model.dir_residual.weight)
                nn.init.zeros_(self.model.dir_residual.bias)
                nn.init.zeros_(self.model.boost_head.weight)
                nn.init.zeros_(self.model.boost_head.bias)
                print("Architecture changed: zeroed dir_residual and boost_head")
        except FileNotFoundError:
            print("Starting fresh")

        self.model.eval()


class Session:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def handle_message(self, state_d):
        if not state_d:
            print("DEAD\n")
            return [0, 0]

        x = get_flat(state_d)
        _, _, is_avoiding = bot_script(state_d)
        x_aug = np.append(x, float(is_avoiding)).astype(np.float32)
        x_t = torch.from_numpy(x_aug).unsqueeze(0).to(self.runtime.device)
        with torch.no_grad():
            game_dir, boost, _, _, _ = self.runtime.model.act(x_t)
        return [game_dir, boost, time() * 1000]


_runtime = None


def get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime
