#!/usr/bin/env python3
"""
Offline convergence test for avoidance imitation learning.

Reads rl_experience.jsonl, applies bot_script to get avoidance targets,
trains with shuffled data, and reports whether the model can converge.

Usage: python test_offline.py
"""
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    AVOID_ANGLE_INDEX,
    AVOID_DIST_INDEX,
    FOOD_END_INDEX,
    FOOD_START_INDEX,
    HEADING_INDEX,
    IS_AVOIDING_INDEX,
    bot_script,
    get_flat,
    supervised_focus_features,
)
LR         = 1e-3
EPOCHS     = 20
BATCH      = 64

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading data...")
states, targets, avoid_angles_raw = [], [], []
avoid_count = 0

with open("experience.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if not isinstance(d, list) or len(d) != 4:
            continue
        target_dir, _target_boost, is_avoiding = bot_script(d)
        try:
            x = get_flat(d).astype(np.float32)
            x_aug = np.append(x, float(is_avoiding)).astype(np.float32)
        except Exception:
            continue
        states.append(x_aug)
        targets.append(target_dir)
        avoid_count += int(is_avoiding)

        # also record raw avoid_angle for diagnostic
        snakes = d[3]
        min_dist, avoid_angle = float('inf'), 0.0
        for s in snakes:
            body = s[4:]
            for angle, dist in zip(body[::2], body[1::2]):
                if dist < min_dist:
                    min_dist, avoid_angle = dist, angle
        avoid_angles_raw.append(avoid_angle)

n = len(states)
print(f"Supervised frames: {n}")
print(f"Avoidance frames: {avoid_count}")
if n < BATCH:
    print("Not enough supervised data.")
    raise SystemExit(1)

states  = np.stack(states)
targets = np.array(targets, dtype=np.float32)
avoid_angles_raw = np.array(avoid_angles_raw, dtype=np.float32)

# Diagnostic: verify avoid_angle is present in features at expected index.
feat35 = states[:, AVOID_ANGLE_INDEX]
corr = np.corrcoef(avoid_angles_raw, feat35)[0, 1]
print(f"\nDiagnostic: corr(avoid_angle_raw, feature[{AVOID_ANGLE_INDEX}]) = {corr:.4f}  (want ~1.0)")
print(f"  avoid_angle_raw range: [{avoid_angles_raw.min():.3f}, {avoid_angles_raw.max():.3f}]")
print(f"  feature[35]    range: [{feat35.min():.3f}, {feat35.max():.3f}]")
print(f"  target_dir     range: [{targets.min():.3f}, {targets.max():.3f}]")
print(f"  food feature count used by SL focus: {FOOD_END_INDEX - FOOD_START_INDEX}")

# ---------------------------------------------------------------------------
# Model (same supervised focus architecture as model.py)
# ---------------------------------------------------------------------------

class FocusedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dir_focus = nn.Sequential(
            nn.Linear(34, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return torch.tanh(self.dir_focus(supervised_focus_features(x))).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = FocusedNet().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=LR)

sx = torch.from_numpy(states).to(device)
sy = torch.from_numpy(targets).to(device)

def circular_loss(pred, target):
    err = (pred - target) % 2
    err = torch.where(err > 1, err - 2, err)
    return (err ** 2).mean()

def train_loop(label, loss_fn):
    model2 = FocusedNet().to(device)
    opt2   = torch.optim.Adam(model2.parameters(), lr=LR)
    print(f"\n--- {label} ---")
    print(f"{'Epoch':>5}  {'loss':>8}  {'grad_norm':>10}  {'pred_std':>9}")
    for epoch in range(EPOCHS):
        model2.train()
        idx = torch.randperm(n, device=device)
        epoch_loss, grad_norm_sum, pred_std_sum, batches = 0.0, 0.0, 0.0, 0
        for start in range(0, n - BATCH + 1, BATCH):
            mb = idx[start:start + BATCH]
            pred = model2(sx[mb])
            loss = loss_fn(pred, sy[mb])
            opt2.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model2.parameters(), 0.5)
            opt2.step()
            epoch_loss    += loss.item()
            grad_norm_sum += gn.item()
            pred_std_sum  += pred.detach().std().item()
            batches += 1
        print(f"{epoch+1:>5}  {epoch_loss/batches:>8.4f}  {grad_norm_sum/batches:>10.4f}  {pred_std_sum/batches:>9.4f}")

train_loop("Focused SL head (food + heading + snake + mode flag)", circular_loss)

# Minimal test: can a 1-layer net learn from the same focused supervised features?
print("\n--- Minimal: focused supervised feature set -> target_dir ---")
sx2 = supervised_focus_features(sx)
model3 = nn.Sequential(nn.Linear(34, 32), nn.Tanh(), nn.Linear(32, 1), nn.Tanh())
model3 = model3.to(device)
opt3 = torch.optim.Adam(model3.parameters(), lr=LR)
print(f"{'Epoch':>5}  {'loss':>8}  {'pred_std':>9}")
for epoch in range(EPOCHS):
    model3.train()
    idx = torch.randperm(n, device=device)
    epoch_loss, pred_std_sum, batches = 0.0, 0.0, 0
    for start in range(0, n - BATCH + 1, BATCH):
        mb = idx[start:start + BATCH]
        pred = model3(sx2[mb]).squeeze(-1)
        loss = circular_loss(pred, sy[mb])
        opt3.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model3.parameters(), 0.5)
        opt3.step()
        epoch_loss += loss.item(); pred_std_sum += pred.detach().std().item(); batches += 1
    print(f"{epoch+1:>5}  {epoch_loss/batches:>8.4f}  {pred_std_sum/batches:>9.4f}")

print(f"\nThe live supervised path now intentionally learns from heading, food features, snake avoid features, and feature[{IS_AVOIDING_INDEX}] while keeping the full input available for RL.")
