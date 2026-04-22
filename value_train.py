#!/usr/bin/env python3
"""
Train ValueNet to predict V(t) = size_gain - 0.053 * horizon - death_indicator.

Generate labels first:
    python label_value.py < experience-t0.jsonl > value_labels.npy

Then train:
    python value_train.py
    python value_train.py --epochs 50 --start 30000
"""
import argparse
import json
from itertools import islice

import numpy as np
import torch
import torch.nn as nn

from model import get_flat
from value_model import VALUE_SAVE_PATH, ValueNet


def load_data(experience_path, labels_path, start, count):
    labels = np.load(labels_path)

    states     = []
    targets    = []
    frame_idxs = []
    seen       = 0

    with open(experience_path) as f:
        lines = islice(f, start, None if count is None else start + count)
        for line_offset, raw in enumerate(lines):
            line_idx = start + line_offset
            raw = raw.strip()
            if not raw:
                continue
            seen += 1

            if line_idx >= len(labels) or np.isnan(labels[line_idx]):
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
            targets.append(float(labels[line_idx]))
            frame_idxs.append(line_idx)

    if not states:
        raise SystemExit("No labeled frames found in the selected range.")

    print(f"Scanned: {seen}  Labeled: {len(states)}  "
          f"Frame range: {frame_idxs[0]}–{frame_idxs[-1]}")
    t = np.array(targets)
    n_death = int((t < -1.0).sum())
    print(f"Target stats:  mean={t.mean():.3f}  std={t.std():.3f}  "
          f"min={t.min():.3f}  max={t.max():.3f}  "
          f"death_frac={n_death / len(t):.2f}")
    return np.stack(states), np.array(targets, dtype=np.float32), np.array(frame_idxs, dtype=np.int64)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ValueNet().to(device)

    resume_ep = 0
    try:
        ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        resume_ep = ckpt.get("ep", 0)
        print(f"Resumed from {args.model_path} (ep={resume_ep})")
    except FileNotFoundError:
        print("Starting fresh")

    states, targets, _ = load_data(args.experience, args.labels, args.start, args.count)
    sx = torch.from_numpy(states).to(device)
    sy = torch.from_numpy(targets).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    n          = len(states)
    batch_size = min(args.batch_size, n)

    print(f"Device: {device}  n={n}  batch={batch_size}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'mae':>8}")
    for epoch in range(args.epochs):
        model.train()
        idx      = torch.randperm(n, device=device)
        loss_sum = 0.0
        batches  = 0
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

        model.eval()
        with torch.no_grad():
            mae = (model(sx) - sy).abs().mean().item()

        print(f"{resume_ep + epoch + 1:>5}  {loss_sum / batches:>8.4f}  {mae:>8.4f}", flush=True)

    torch.save({"model": model.state_dict(), "ep": resume_ep + args.epochs}, args.model_path)
    print(f"Saved {args.model_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experience",  default="experience-t0.jsonl")
    p.add_argument("--labels",      default="value_labels.npy")
    p.add_argument("--start",       type=int,   default=0)
    p.add_argument("--count",       type=int,   default=None)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--model-path",  default=VALUE_SAVE_PATH)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
