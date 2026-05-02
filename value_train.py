#!/usr/bin/env python3
"""
Train ValueNet to predict V(t) = size_gain - AVERAGE_VALUE * horizon - death_indicator.

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

from model import get_flat, IN_DIM, OWN_SEGMENTS_INDEX, FOOD_START_INDEX, FOOD_END_INDEX
from value_model import ValueNet, FINE_INDICES
from value2_model import Value2Net
import value_model  as _vm1
import value2_model as _vm2

BODY_FREEZE_INDICES = list(range(OWN_SEGMENTS_INDEX + 2, IN_DIM))
FOOD_FREEZE_INDICES = list(range(FOOD_START_INDEX, FOOD_END_INDEX))

MODEL_REGISTRY = {
    'value':  (_vm1, ValueNet),
    'value2': (_vm2, Value2Net),
}


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
    mod, model_cls = MODEL_REGISTRY[args.model]
    model = model_cls()

    resume_ep = 0
    try:
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        resume_ep = ckpt.get("ep", 0)
        print(f"Resumed from {args.model_path} (ep={resume_ep})")
    except FileNotFoundError:
        print("Starting fresh")

    states, targets, _ = load_data(args.experience, args.labels, args.start, args.count)
    sx = torch.from_numpy(states)
    sy = torch.from_numpy(targets)

    if args.freeze_mid:
        for i in mod.MID_LAYERS:
            for p in model.head[i].parameters(): p.requires_grad = False
            print(f"Frozen: head[{i}]")

    if args.freeze_fine:
        def _zero_fine_grads(grad):
            grad = grad.clone()
            grad[:, FINE_INDICES] = 0
            return grad
        model.head[0].weight.register_hook(_zero_fine_grads)
        print(f"Frozen: head[0] fine columns ({len(FINE_INDICES)} of {model.head[0].weight.shape[1]})")

    if args.freeze_body:
        def _zero_body_grads(grad):
            grad = grad.clone()
            grad[:, BODY_FREEZE_INDICES] = 0
            return grad
        model.head[0].weight.register_hook(_zero_body_grads)
        print(f"Frozen: head[0] body columns ({len(BODY_FREEZE_INDICES)} of {model.head[0].weight.shape[1]})")

    if args.freeze_food:
        def _zero_food_grads(grad):
            grad = grad.clone()
            grad[:, FOOD_FREEZE_INDICES] = 0
            return grad
        model.head[0].weight.register_hook(_zero_food_grads)
        print(f"Frozen: head[0] food columns ({len(FOOD_FREEZE_INDICES)} of {model.head[0].weight.shape[1]})")

    optimizer  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.decay
    )
    n          = len(states)
    batch_size = min(args.batch_size, n)

    noise_mask = torch.ones(sx.shape[1], dtype=torch.bool)
    noise_mask[mod.ACTION_INDICES] = False

    print(f"n={n}  batch={batch_size}  noise_factor={args.noise_factor}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'mae':>8}")
    for epoch in range(args.epochs):
        model.train()
        idx      = torch.randperm(n)
        loss_sum = 0.0
        batches  = 0
        for start in range(0, n - batch_size + 1, batch_size):
            mb    = idx[start:start + batch_size]
            batch = sx[mb]
            if args.noise_factor:
                noise = torch.zeros_like(batch)
                noise[:, noise_mask] = torch.randn(len(mb), noise_mask.sum().item()) * args.noise_factor
                batch = batch + noise
            pred = model(batch)
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
    p.add_argument("--experience",  default="old-experience.jsonl")
    p.add_argument("--labels",      default="value_labels.npy")
    p.add_argument("--start",       type=int,   default=0)
    p.add_argument("--count",       type=int,   default=None)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--decay",       type=float, default=1e-4)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--freeze-mid",  action="store_true")
    p.add_argument("--freeze-fine", action="store_true")
    p.add_argument("--freeze-body", action="store_true")
    p.add_argument("--freeze-food",   action="store_true")
    p.add_argument("--noise-factor",  type=float, default=0.0)
    p.add_argument("--model",       choices=['value', 'value2'], default='value')
    p.add_argument("--model-path",  default=None)
    args = p.parse_args()
    if args.model_path is None:
        args.model_path = MODEL_REGISTRY[args.model][1].save_path
    return args


if __name__ == "__main__":
    train(parse_args())
