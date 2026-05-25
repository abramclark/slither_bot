#!/usr/bin/env python3
"""
Train ValueNet.

Modes:
  rl     MSE against discounted return labels (default)
  clone  Contrastive: taken direction = +|label|, SAMPLE_COUNT others = -|label|

Generate labels first:
    python label_value.py < experience.jsonl > value_labels.npy

Then train:
    python value_train.py
    python value_train.py --mode clone --epochs 50
"""
import argparse
import json
from itertools import islice

import numpy as np
import torch
import torch.nn as nn

from environment import get_flat, IN_DIM, OWN_SEGMENTS_INDEX, FOOD_START_INDEX, FOOD_END_INDEX, DIRX_INDEX, HEADINGX_INDEX
from value_model import ValueNet, FINE_INDICES, SAMPLE_COUNT
import value_model as _vm1

BODY_FREEZE_INDICES = list(range(OWN_SEGMENTS_INDEX + 2, IN_DIM))
FOOD_FREEZE_INDICES = list(range(FOOD_START_INDEX, FOOD_END_INDEX))

SAMPLE_OFFSETS = torch.arange(SAMPLE_COUNT) * (2 * np.pi / SAMPLE_COUNT)  # relative offsets


def load_data(experience_path, labels_path, start, count):
    labels = np.load(labels_path)
    states, targets, frame_idxs = [], [], []
    seen = 0

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
            d = json.loads(raw)
            if d == []:
                continue
            state, action, improv = d
            states.append(get_flat(state).astype(np.float32))
            targets.append(float(labels[line_idx]))
            frame_idxs.append(line_idx)

    if not states:
        raise SystemExit("No labeled frames found in the selected range.")

    t = np.array(targets)
    print(f"Scanned: {seen}  Labeled: {len(states)}  "
          f"Frame range: {frame_idxs[0]}–{frame_idxs[-1]}")
    print(f"Target stats:  mean={t.mean():.3f}  std={t.std():.3f}  "
          f"min={t.min():.3f}  max={t.max():.3f}  "
          f"death_frac={(t < -1.0).mean():.2f}")
    return np.stack(states), t.astype(np.float32), np.array(frame_idxs, np.int64)


def load_data_clone(experience_path, labels_path, start, count):
    labels = np.load(labels_path)
    states, taken_dirs, label_vals, is_improv, frame_idxs = [], [], [], [], []
    seen = 0

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
            d = json.loads(raw)
            if d == []:
                continue
            state, action, improv = d
            executed = improv if improv else action
            angle = executed[0]
            states.append(get_flat(state).astype(np.float32))
            taken_dirs.append([np.cos(angle), np.sin(angle)])
            label_vals.append(float(labels[line_idx]))
            is_improv.append(bool(improv))
            frame_idxs.append(line_idx)

    if not states:
        raise SystemExit("No labeled frames found in the selected range.")

    t = np.array(label_vals)
    print(f"Scanned: {seen}  Labeled: {len(states)}  "
          f"Frame range: {frame_idxs[0]}–{frame_idxs[-1]}")
    return np.stack(states), np.array(taken_dirs, np.float32), t.astype(np.float32), np.array(is_improv)


def setup_freezes(model, args):
    if args.freeze_mid:
        for i in _vm1.MID_LAYERS:
            for p in model.head[i].parameters(): p.requires_grad = False
            print(f"Frozen: head[{i}]")

    if args.freeze_fine:
        def _zero_fine(grad):
            grad = grad.clone(); grad[:, FINE_INDICES] = 0; return grad
        model.head[0].weight.register_hook(_zero_fine)
        print(f"Frozen: head[0] fine columns ({len(FINE_INDICES)} of {model.head[0].weight.shape[1]})")

    if args.freeze_food:
        def _zero_food(grad):
            grad = grad.clone(); grad[:, FOOD_FREEZE_INDICES] = 0; return grad
        model.head[0].weight.register_hook(_zero_food)
        print(f"Frozen: head[0] food columns ({len(FOOD_FREEZE_INDICES)} of {model.head[0].weight.shape[1]})")


def train(args):
    model = ValueNet()
    resume_ep = model.load()
    setup_freezes(model, args)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.decay,
    )

    if args.mode == 'clone':
        train_clone(model, optimizer, args, resume_ep)
    else:
        train_rl(model, optimizer, args, resume_ep)

    torch.save({"model": model.state_dict(), "ep": resume_ep + args.epochs}, args.model_path)
    print(f"Saved {args.model_path}")


def train_rl(model, optimizer, args, resume_ep):
    states, targets, _ = load_data(args.experience, args.labels, args.start, args.count)
    sx = torch.from_numpy(states)
    sy = torch.from_numpy(targets)
    n = len(states)
    batch_size = min(args.batch_size, n)

    noise_mask = torch.ones(sx.shape[1], dtype=torch.bool)
    noise_mask[_vm1.ACTION_INDICES] = False

    print(f"n={n}  batch={batch_size}  noise_factor={args.noise_factor}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'mae':>8}")
    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(n)
        loss_sum = 0.0
        batches = 0
        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
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
            batches += 1

        model.eval()
        with torch.no_grad():
            mae = (model(sx) - sy).abs().mean().item()
        print(f"{resume_ep + epoch + 1:>5}  {loss_sum / batches:>8.4f}  {mae:>8.4f}", flush=True)


def train_clone(model, optimizer, args, resume_ep):
    states, taken_dirs, label_vals, is_improv = load_data_clone(args.experience, args.labels, args.start, args.count)
    sx = torch.from_numpy(states)
    td = torch.from_numpy(taken_dirs)    # (N, 2)
    ly = torch.from_numpy(label_vals)    # (N,)
    ii = torch.from_numpy(is_improv)     # (N,) bool

    global_mean = ly.mean().item()
    scores = ly - global_mean            # positive = above-average outcome

    n_improv       = int(ii.sum())
    n_improv_above = int((ii & (scores >= 0)).sum())

    keep = (scores >= 0) if args.drop_negative else (~ii | (scores >= 0))
    sx, td, scores = sx[keep], td[keep], scores[keep]
    n = len(sx)
    batch_size = min(args.batch_size, n)

    print(f"global_mean={global_mean:.3f}  n={n}  batch={batch_size}")
    print(f"Improv: {n_improv}  improv above mean: {n_improv_above}  dropped: {int((~keep).sum())}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'mae':>8}")
    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(n)
        loss_sum = 0.0
        batches = 0
        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
            B = len(mb)
            score_mb = scores[mb]        # (B,)

            # Heading-relative sample angles: (B, SAMPLE_COUNT)
            headings = torch.atan2(sx[mb, HEADINGX_INDEX + 1], sx[mb, HEADINGX_INDEX])
            sample_angles = headings.unsqueeze(1) + SAMPLE_OFFSETS.unsqueeze(0)  # (B, SAMPLE_COUNT)

            # Remove the bin closest to the taken direction
            taken_angle = torch.atan2(td[mb, 1], td[mb, 0])  # (B,)
            diff = ((sample_angles - taken_angle.unsqueeze(1) + np.pi) % (2 * np.pi) - np.pi).abs()
            nearest = diff.argmin(dim=1)  # (B,) bin to exclude
            mask = torch.ones(B, SAMPLE_COUNT, dtype=torch.bool)
            mask[torch.arange(B), nearest] = False  # (B, SAMPLE_COUNT)

            # (B*(SAMPLE_COUNT-1), IN_DIM) negative examples, target = -|score|
            neg = sx[mb].unsqueeze(1).expand(B, SAMPLE_COUNT, -1)[mask].clone()
            neg[:, DIRX_INDEX]     = sample_angles[mask].cos()
            neg[:, DIRX_INDEX + 1] = sample_angles[mask].sin()
            neg_targets = -score_mb.abs().unsqueeze(1).expand(B, SAMPLE_COUNT)[mask]

            # 1 positive example per state: taken direction, target = score
            pos = sx[mb].clone()
            pos[:, DIRX_INDEX]     = td[mb, 0]
            pos[:, DIRX_INDEX + 1] = td[mb, 1]

            batch   = torch.cat([neg, pos], dim=0)
            targets = torch.cat([neg_targets, score_mb], dim=0)

            pred = model(batch)
            loss = nn.functional.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item()
            batches += 1

        model.eval()
        with torch.no_grad():
            sx_pos = sx.clone()
            sx_pos[:, DIRX_INDEX]     = td[:, 0]
            sx_pos[:, DIRX_INDEX + 1] = td[:, 1]
            mae = (model(sx_pos) - scores).abs().mean().item()
        print(f"{resume_ep + epoch + 1:>5}  {loss_sum / batches:>8.4f}  {mae:>8.4f}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",         choices=['rl', 'clone'], default='rl')
    p.add_argument("--experience",   default="experience.jsonl")
    p.add_argument("--labels",       default="value_labels.npy")
    p.add_argument("--start",        type=int,   default=0)
    p.add_argument("--count",        type=int,   default=None)
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--decay",        type=float, default=1e-4)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--freeze-mid",     action="store_true")
    p.add_argument("--freeze-fine",    action="store_true")
    p.add_argument("--freeze-food",    action="store_true")
    p.add_argument("--noise-factor",   type=float, default=0.0)
    p.add_argument("--drop-negative",  action="store_true", help="drop all below-mean examples (clone mode)")
    p.add_argument("--model-path",   default=ValueNet.save_path)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
