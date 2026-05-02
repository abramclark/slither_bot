#!/usr/bin/env python3
"""
Train Model (model.py).

Modes:
  supervised  Imitate bot_script (default)
  rl          REINFORCE policy gradient using return labels

Usage:
    python train.py
    python train.py --experience experience.jsonl --epochs 50
    python train.py --mode rl --labels value_labels.npy
"""
import argparse
import json
from itertools import islice

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_flat, bot_script, Model, SAVE_PATH, BOOST_INDEX, FINE_INDICES, OWN_SEGMENTS_INDEX, IN_DIM, K_ANGLE_BINS

ANGLE_BINS = torch.linspace(0, 2 * math.pi, K_ANGLE_BINS + 1)[:-1]  # (K_ANGLE_BINS,)

def angle_to_bin(cos_v, sin_v):
    a = torch.atan2(sin_v, cos_v) % (2 * math.pi)
    d = (a.unsqueeze(1) - ANGLE_BINS.unsqueeze(0) + math.pi) % (2 * math.pi) - math.pi
    return d.abs().argmin(dim=1)

INPUT_OFFSET        = 2  # model.forward strips x[..., 2:] (turn direction)
BODY_FREEZE_INDICES = list(range(OWN_SEGMENTS_INDEX + 2 - INPUT_OFFSET, IN_DIM - INPUT_OFFSET))


def load_supervised(experience_path, start, count):
    states  = []
    targets = []

    with open(experience_path) as f:
        lines = islice(f, start, None if count is None else start + count)
        for raw in lines:
            raw = raw.strip()
            d = json.loads(raw)
            if not isinstance(d, list) or len(d) != 4:
                continue
            x = get_flat(d).astype(np.float32)
            angle, boost, _ = bot_script(d)

            states.append(x)
            targets.append([np.cos(angle), np.sin(angle), float(boost)])

    if not states:
        raise SystemExit("No frames found.")

    t = np.array(targets, dtype=np.float32)
    print(f"Loaded {len(states)} frames  boost_frac={t[:, 2].mean():.2f}")
    return np.stack(states), t


def load_rl(experience_path, labels_path, start, count):
    labels     = np.load(labels_path)
    states     = []
    returns    = []
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
            x = get_flat(d).astype(np.float32)
            states.append(x)
            returns.append(float(labels[line_idx]))
            frame_idxs.append(line_idx)

    if not states:
        raise SystemExit("No labeled frames found.")

    r = np.array(returns, dtype=np.float32)
    print(f"Scanned {seen}  labeled {len(states)}  "
          f"return mean={r.mean():.3f} std={r.std():.3f}")
    return np.stack(states), r


def setup_freezes(model, args):
    if args.freeze_fine:
        def _zero_fine(grad):
            g = grad.clone(); g[:, FINE_INDICES] = 0; return g
        model.head[0].weight.register_hook(_zero_fine)
        print(f"Frozen: fine columns ({len(FINE_INDICES)} of {model.head[0].weight.shape[1]})")
    if args.freeze_body:
        def _zero_body(grad):
            g = grad.clone(); g[:, BODY_FREEZE_INDICES] = 0; return g
        model.head[0].weight.register_hook(_zero_body)
        print(f"Frozen: body columns ({len(BODY_FREEZE_INDICES)} of {model.head[0].weight.shape[1]})")


def train(args):
    model = Model(dropout=args.dropout_factor)

    resume_ep = 0
    try:
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        resume_ep = ckpt.get("ep", 0)
        print(f"Resumed from {args.model_path} (ep={resume_ep})")
    except FileNotFoundError:
        print("Starting fresh")

    setup_freezes(model, args)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    batch_size = args.batch_size

    if args.mode == 'rl':
        train_rl(model, optimizer, args, resume_ep, batch_size)
    else:
        train_supervised(model, optimizer, args, resume_ep, batch_size)

    torch.save({"model": model.state_dict(), "ep": resume_ep + args.epochs}, args.model_path)
    print(f"Saved {args.model_path}")


def train_supervised(model, optimizer, args, resume_ep, batch_size):
    states, targets = load_supervised(args.experience, args.start, args.count)
    sx = torch.from_numpy(states)
    sy = torch.from_numpy(targets)
    n  = len(states)
    batch_size = min(batch_size, n)
    bce = nn.BCEWithLogitsLoss()

    target_idx = angle_to_bin(sy[:, 0], sy[:, 1])  # (N,) bin indices

    print(f"n={n}  batch={batch_size}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'dir_mae':>8}  {'boost_acc':>9}")
    for epoch in range(args.epochs):
        model.train()
        idx      = torch.randperm(n)
        loss_sum = 0.0
        batches  = 0
        for start in range(0, n - batch_size + 1, batch_size):
            mb       = idx[start:start + batch_size]
            pred     = model(sx[mb])
            loss     = (F.cross_entropy(pred[:, :K_ANGLE_BINS], target_idx[mb])
                        + bce(pred[:, K_ANGLE_BINS], sy[mb, 2]))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item()
            batches  += 1

        model.eval()
        with torch.no_grad():
            pred      = model(sx)
            probs     = F.softmax(pred[:, :K_ANGLE_BINS], dim=1)
            pred_cos  = (probs * ANGLE_BINS.cos()).sum(dim=1)
            pred_sin  = (probs * ANGLE_BINS.sin()).sum(dim=1)
            dir_mae   = ((pred_cos - sy[:, 0])**2 + (pred_sin - sy[:, 1])**2).sqrt().mean().item()
            boost_acc = ((pred[:, K_ANGLE_BINS] > 0) == sy[:, 2].bool()).float().mean().item()

        print(f"{resume_ep + epoch + 1:>5}  {loss_sum / batches:>8.4f}  "
              f"{dir_mae:>8.4f}  {boost_acc:>9.3f}", flush=True)


def train_rl(model, optimizer, args, resume_ep, batch_size):
    if not args.labels:
        raise SystemExit("--labels required for rl mode")
    states, returns = load_rl(args.experience, args.labels, args.start, args.count)
    sx = torch.from_numpy(states)
    sy = torch.from_numpy(returns)
    n  = len(states)
    batch_size = min(batch_size, n)

    global_mean = sy.mean()
    global_std  = sy.std() + 1e-8

    print(f"n={n}  batch={batch_size}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'log_prob':>9}  {'conf':>6}")
    for epoch in range(args.epochs):
        model.train()
        idx          = torch.randperm(n)
        loss_sum     = 0.0
        lp_sum       = 0.0
        batches      = 0
        for start in range(0, n - batch_size + 1, batch_size):
            mb          = idx[start:start + batch_size]
            pred        = model(sx[mb])                         # (B, K_ANGLE_BINS+1)

            taken_idx   = angle_to_bin(sx[mb, 0], sx[mb, 1])   # (B,) nearest bin
            taken_boost = (sx[mb, BOOST_INDEX] >= 1.0).float()

            log_prob_dir   = F.log_softmax(pred[:, :K_ANGLE_BINS], dim=1).gather(1, taken_idx.unsqueeze(1)).squeeze(1).clamp(min=-math.log(K_ANGLE_BINS))
            log_prob_boost = -F.binary_cross_entropy_with_logits(pred[:, K_ANGLE_BINS].clamp(-1, 1), taken_boost, reduction='none')
            log_prob = log_prob_dir + log_prob_boost

            adv = (sy[mb] - global_mean) / global_std

            # entropy on full batch before masking
            probs   = F.softmax(pred[:, :K_ANGLE_BINS], dim=1)
            entropy = -(probs * F.log_softmax(pred[:, :K_ANGLE_BINS], dim=1)).sum(dim=1).mean()

            if not args.both_signs:
                mask     = adv > 0
                adv      = adv[mask]
                log_prob = log_prob[mask]
                if adv.numel() == 0:
                    continue

            loss = -(log_prob * adv).mean() - args.entropy_coeff * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item()
            lp_sum   += log_prob.mean().item()
            batches  += 1

        model.eval()
        with torch.no_grad():
            p    = F.softmax(model(sx)[:, :K_ANGLE_BINS], dim=1)
            conf = p.std(dim=1).mean().item()

        print(f"{resume_ep + epoch + 1:>5}  {loss_sum / batches:>8.4f}  "
              f"{lp_sum / batches:>9.4f}  {conf:>6.3f}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=['supervised', 'rl'], default='supervised')
    p.add_argument("--experience",  default="experience.jsonl")
    p.add_argument("--labels",      default=None, help="return labels .npy (required for rl mode)")
    p.add_argument("--start",       type=int,   default=0)
    p.add_argument("--count",       type=int,   default=None)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--decay",       type=float, default=0)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--dropout-factor", type=float, default=0)
    p.add_argument("--both-signs",    action="store_true", help="use both positive and negative advantage (requires entropy-coeff to prevent runaway)")
    p.add_argument("--freeze-fine", action="store_true")
    p.add_argument("--freeze-body", action="store_true")
    p.add_argument("--model-path",  default=SAVE_PATH)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
