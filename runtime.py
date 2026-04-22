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
from survival_model import SURVIVAL_SAVE_PATH, SurvivalNet
from value_model import VALUE_SAVE_PATH, ValueNet

SURVIVAL_FOOD_THRESHOLD = 2.0  # log(frames+1); expm1(2.0) ≈ 6.4 frames


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


class SurvivalRuntime:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SurvivalNet().to(self.device)
        try:
            ckpt = torch.load(SURVIVAL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            ep = ckpt.get("ep", 0)
            print(f"Survival: resumed from {SURVIVAL_SAVE_PATH} (ep={ep})")
        except FileNotFoundError:
            print(f"Survival: no checkpoint at {SURVIVAL_SAVE_PATH}, starting fresh")
        self.model.eval()


class SurvivalSession:
    def __init__(self, runtime: SurvivalRuntime):
        self.runtime = runtime

    def handle_message(self, state_d):
        if not state_d:
            print("DEAD (survival)\n")
            return [0, 0]

        x = get_flat(state_d).astype(np.float32)
        game_dir, boost, val = self.runtime.model.act(x)

        if val > SURVIVAL_FOOD_THRESHOLD:
            food_dir, food_boost, _ = bot_script(state_d)
            print(f"[survival] FOOD  dir={food_dir:.3f}  boost={food_boost}  val={val:.3f}")
            return [food_dir, food_boost, time() * 1000]

        print(f"[survival] EVADE dir={game_dir:.3f}  boost={boost}  val={val:.3f}")
        return [game_dir, boost, time() * 1000]


class ValueRuntime:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ValueNet().to(self.device)
        try:
            ckpt = torch.load(VALUE_SAVE_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model"])
            ep = ckpt.get("ep", 0)
            print(f"Value: resumed from {VALUE_SAVE_PATH} (ep={ep})")
        except FileNotFoundError:
            print(f"Value: no checkpoint at {VALUE_SAVE_PATH}, starting fresh")
        self.model.eval()


class ValueSession:
    def __init__(self, runtime: ValueRuntime):
        self.runtime = runtime

    def handle_message(self, state_d):
        if not state_d:
            print("DEAD (value)\n")
            return [0, 0]

        x = get_flat(state_d).astype(np.float32)
        game_dir, boost, val = self.runtime.model.act(x)
        print(f"[value] dir={game_dir:.3f}  boost={boost}  val={val:.3f}")
        return [game_dir, boost, time() * 1000]


_runtime = None
_survival_runtime = None
_value_runtime = None


def get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


def get_survival_runtime():
    global _survival_runtime
    if _survival_runtime is None:
        _survival_runtime = SurvivalRuntime()
    return _survival_runtime


def get_value_runtime():
    global _value_runtime
    if _value_runtime is None:
        _value_runtime = ValueRuntime()
    return _value_runtime
