#!/usr/bin/env python3
"""
Train SurvivalNet to predict log(frames_until_death + 1).

Usage:
    python survival_train.py --labels survival_labels.npy
    python survival_train.py --labels survival_labels.npy --start 30000 --epochs 50
"""
import argparse
import json
from itertools import islice

import numpy as np
import torch
import torch.nn as nn

from model import get_flat
from survival_model import SURVIVAL_SAVE_PATH, SurvivalNet


def load_data(experience_path, labels_path, start, count, max_frames=None):
    labels = np.load(labels_path)

    states     = []
    targets    = []
    frame_idxs = []
    seen       = 0
    used       = 0

    with open(experience_path) as f:
        lines = islice(f, start, None if count is None else start + count)
        for line_offset, raw in enumerate(lines):
            line_idx = start + line_offset
            raw = raw.strip()
            if not raw:
                continue
            seen += 1

            if line_idx >= len(labels) or labels[line_idx] < 1:
                continue
            if max_frames is not None and labels[line_idx] > max_frames:
                continue

            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, list) or len(d) != 4:
                continue

            try:
                x = get_flat(d).astype(np.float32)
            except Exception:
                continue

            states.append(x)
            targets.append(np.log1p(labels[line_idx]).astype(np.float32))
            frame_idxs.append(line_idx)
            used += 1

    if not states:
        raise SystemExit("No labeled frames found in the selected range.")

    print(f"Scanned: {seen}  Labeled: {used}  "
          f"Frame range: {frame_idxs[0]}–{frame_idxs[-1]}")
    t = np.array(targets)
    print(f"Target stats (log scale):  mean={t.mean():.3f}  std={t.std():.3f}  "
          f"min={t.min():.3f}  max={t.max():.3f}")
    return np.stack(states), np.array(targets, dtype=np.float32), np.array(frame_idxs, dtype=np.int64)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SurvivalNet().to(device)

    resume_ep = 0
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        resume_ep = ckpt.get("ep", 0)
        print(f"Resumed from {args.model_path} (ep={resume_ep})")
    except FileNotFoundError:
        print(f"Starting fresh")

    states, targets, frame_idxs = load_data(args.experience, args.labels, args.start, args.count, args.max_frames)
    sx = torch.from_numpy(states).to(device)
    sy = torch.from_numpy(targets).to(device)
    fi = torch.from_numpy(frame_idxs).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    n          = len(states)
    batch_size = min(args.batch_size, n)

    print(f"Device: {device}  n={n}  batch={batch_size}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'mae_frames':>11}  {'skipped_range':>20}")
    for epoch in range(args.epochs):
        model.train()
        idx       = torch.randperm(n, device=device)
        loss_sum  = 0.0
        batches   = 0
        for start in range(0, n - batch_size + 1, batch_size):
            mb   = idx[start:start + batch_size]
            pred = model(sx[mb])
            loss = nn.functional.mse_loss(pred, sy[mb])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item()
            batches  += 1

        # MAE in original frame space (expm1 inverts log1p)
        model.eval()
        with torch.no_grad():
            all_pred   = model(sx)
            mae_frames = (torch.expm1(all_pred) - torch.expm1(sy)).abs().mean().item()

        skipped = fi[idx[batches * batch_size:]]
        skip_range = f"{skipped.min().item()}–{skipped.max().item()}" if len(skipped) else "none"
        print(f"{resume_ep + epoch + 1:>5}  "
              f"{loss_sum / batches:>8.4f}  "
              f"{mae_frames:>11.1f}  "
              f"{skip_range:>20}", flush=True)

    torch.save({"model": model.state_dict(), "ep": resume_ep + args.epochs}, args.model_path)
    print(f"Saved {args.model_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experience",  default="experience.jsonl")
    p.add_argument("--labels",      default="survival_labels.npy")
    p.add_argument("--start",       type=int,   default=0)
    p.add_argument("--count",       type=int,   default=None)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--max-frames",  type=int,   default=20)
    p.add_argument("--model-path",  default=SURVIVAL_SAVE_PATH)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
