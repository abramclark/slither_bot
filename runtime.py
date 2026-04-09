from time import time

import numpy as np
import torch

from model import (
    AVOID_DIST,
    BATCH_SIZE,
    LR,
    MINI_BATCH,
    SAVE_PATH,
    ActorCritic,
    PPOTrainer,
    SupervisedTrainer,
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
        self.training_mode = "supervised"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic().to(self.device)
        self.resume_ep = 0
        self.resume_steps = 0
        try:
            ckpt = torch.load(SAVE_PATH, map_location=self.device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                missing, unexpected, skipped = load_compatible_state_dict(self.model, ckpt["model"])
                self.resume_ep = ckpt.get("ep", 0)
                self.resume_steps = ckpt.get("total_steps", 0)
            else:
                missing, unexpected, skipped = load_compatible_state_dict(self.model, ckpt)
            print(f"Resumed from {SAVE_PATH} (ep={self.resume_ep} steps={self.resume_steps})")
            if missing:
                print(f"Missing checkpoint keys: {missing}")
            if unexpected:
                print(f"Unexpected checkpoint keys: {unexpected}")
            if skipped:
                print(f"Skipped incompatible checkpoint keys: {skipped}")
        except FileNotFoundError:
            print("Starting fresh")

        self.model.train()
        self.trainer = PPOTrainer(self.model, self.device)
        self.trainer.ep = self.resume_ep
        self.trainer.total_steps = self.resume_steps
        self.sl_trainer = SupervisedTrainer(self.model, self.device)
        self.sl_trainer.ep = self.resume_ep
        self.sl_trainer.total_steps = self.resume_steps

    def set_training_mode(self, mode):
        if mode not in ("ppo", "supervised"):
            raise ValueError(f"mode: {mode} must be 'ppo' or 'supervised'")
        self.training_mode = mode
        print(f"[config] TRAINING_MODE -> {mode}")

    def reset_optimizer(self):
        self.sl_trainer.reset_optimizer()

    def get_config(self):
        return {
            "AVOID_DIST": AVOID_DIST,
            "TRAINING_MODE": self.training_mode,
            "LR": LR,
            "BATCH_SIZE": BATCH_SIZE,
            "MINI_BATCH": MINI_BATCH,
        }

    def finish_episode(self, episode_steps):
        if self.training_mode == "ppo":
            with self.trainer.lock:
                n = min(10, episode_steps, len(self.trainer.rewards))
                for i in range(-n, 0):
                    self.trainer.rewards[i] = -10
                if n > 0:
                    self.trainer.dones[-1] = 1.0
                    self.trainer.ep += 1
        else:
            self.sl_trainer.finish_episode()


class Session:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.prev_size = None
        self.episode_steps = 0

    def handle_message(self, state_d):
        if not state_d:
            print("DEAD\n")
            self.runtime.finish_episode(self.episode_steps)
            self.prev_size = None
            self.episode_steps = 0
            return [0, 0]

        if self.runtime.training_mode == "ppo":
            reward = (state_d[0] - self.prev_size) * 10 if self.prev_size is not None else 0.0
            self.prev_size = state_d[0]
            x = get_flat(state_d)
            _, _, is_avoiding = bot_script(state_d)
            x_aug = np.append(x, float(is_avoiding)).astype(np.float32)
            x_t = torch.from_numpy(x_aug).unsqueeze(0).to(self.runtime.device)
            with torch.no_grad():
                direction, boost, log_prob, value = self.runtime.model.act(x_t)
            self.runtime.trainer.push(x_aug, direction, boost, log_prob, value, reward, False)
            self.episode_steps += 1
            return [direction, boost]

        x = get_flat(state_d)
        target_dir, target_boost, is_avoiding = bot_script(state_d)
        x_aug = np.append(x, float(is_avoiding)).astype(np.float32)
        self.runtime.sl_trainer.step(x_aug, target_dir)
        self.episode_steps += 1
        return [target_dir, target_boost, time() * 1000]


_runtime = None


def get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime
