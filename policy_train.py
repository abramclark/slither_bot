#!/usr/bin/env python3
"""
Train PolicyNet (model.py).

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

from model import get_flat, bot_script, PolicyNet, BOOST_INDEX, FINE_INDICES, OWN_SEGMENTS_INDEX, IN_DIM, K_ANGLE_BINS

ANGLE_BINS = torch.linspace(0, 2 * math.pi, K_ANGLE_BINS + 1)[:-1]  # (K_ANGLE_BINS,)

def angle_to_bin(cos_v, sin_v):
    a = torch.atan2(sin_v, cos_v) % (2 * math.pi)
    d = (a.unsqueeze(1) - ANGLE_BINS.unsqueeze(0) + math.pi) % (2 * math.pi) - math.pi
    return d.abs().argmin(dim=1)

INPUT_OFFSET        = 2  # model.forward strips x[..., 2:] (turn direction)
BODY_FREEZE_INDICES = list(range(OWN_SEGMENTS_INDEX + 2 - INPUT_OFFSET, IN_DIM - INPUT_OFFSET))


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
    model = PolicyNet(dropout=args.dropout_factor)
    resume_ep = model.load(args.model_path)
    setup_freezes(model, args)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    batch_size = args.batch_size

    states, targets, returns = (load_rl if args.mode == 'rl' else load_supervised)(
        args.experience, args.start, args.count, args.labels
    )
    if args.positive is not None and returns is not None:
        keep = returns >= args.positive
        states, targets, returns = states[keep], targets[keep], returns[keep]
        print(f"Filtered to {keep.sum()} examples with return >= {args.positive}")

    sx = torch.from_numpy(states)
    sy = torch.from_numpy(targets)
    n  = len(states)
    batch_size = min(batch_size, n)

    target_idx = angle_to_bin(sy[:, 0], sy[:, 1])  # (N,) bin indices

    print(f"n={n}  batch={batch_size}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'dir_mae':>8}  {'boost_acc':>9}")
    for epoch in range(args.epochs):
        model.train()
        idx      = torch.randperm(n)
        loss_sum = 0.0
        batches  = 0
        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
            pred = model(sx[mb])
            loss = (F.cross_entropy(pred[:, :K_ANGLE_BINS], target_idx[mb]) +
                    F.binary_cross_entropy_with_logits(pred[:, K_ANGLE_BINS], sy[mb, 2]))
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

    torch.save({"model": model.state_dict(), "ep": resume_ep + args.epochs}, args.model_path)
    print(f"Saved {args.model_path}")


def load_supervised(experience_path, start, count, labels_path):
    states  = []
    targets = []

    with open(experience_path) as f:
        lines = islice(f, start, None if count is None else start + count)
        for raw in lines:
            raw = raw.strip()
            d = json.loads(raw)
            if d == []:
                continue
            state, action, improv, *_ = d
            x = get_flat(state).astype(np.float32)
            angle, boost, _ = bot_script(state)
            states.append(x)
            targets.append([np.cos(angle), np.sin(angle), float(boost)])

    if not states:
        raise SystemExit("No frames found.")

    t = np.array(targets, dtype=np.float32)
    print(f"Loaded {len(states)} frames  boost_frac={t[:, 2].mean():.2f}")
    return np.stack(states), t, None


def load_rl(experience_path, start, count, labels_path):
    labels = np.load(labels_path)
    states = []
    targets = []
    returns = []
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
            state, action, improv, *_ = d
            states.append(get_flat(state))
            if improv:
                action = improv
            targets.append([np.cos(action[0]), np.sin(action[0]), action[1]])
            returns.append(float(labels[line_idx]))

    if not states:
        raise SystemExit("No labeled frames found.")

    r = np.array(returns, dtype=np.float32)
    print(f"Scanned {seen}  labeled {len(states)}  "
          f"return mean={r.mean(): .3f} std={r.std():.3f}")
    return np.array(states), np.array(targets), r


def load_ppo(experience_path, start, count, labels_path):
    labels = np.load(labels_path)
    states, taken_dirs, taken_boosts, old_logits, returns = [], [], [], [], []
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
            state, action, improv, *_ = d
            if improv:
                continue
            if not isinstance(action[2], list) or len(action[2]) != K_ANGLE_BINS + 1:
                continue
            states.append(get_flat(state))
            taken_dirs.append([np.cos(action[0]), np.sin(action[0])])
            taken_boosts.append(float(action[1]))
            old_logits.append(action[2])
            returns.append(float(labels[line_idx]))

    if not states:
        raise SystemExit("No labeled frames found.")

    r = np.array(returns, dtype=np.float32)
    print(f"Scanned {seen}  labeled {len(states)}  "
          f"return mean={r.mean():.3f} std={r.std():.3f}")
    return (np.array(states, dtype=np.float32), np.array(taken_dirs, dtype=np.float32),
            np.array(taken_boosts, dtype=np.float32), np.array(old_logits, dtype=np.float32), r)


def train_ppo(args):
    model = PolicyNet(dropout=args.dropout_factor)
    resume_ep = model.load(args.model_path)
    setup_freezes(model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    states, taken_dirs, taken_boosts, old_logits_np, returns = load_ppo(
        args.experience, args.start, args.count, args.labels
    )
    if args.positive is not None:
        keep = returns >= args.positive
        states, taken_dirs, taken_boosts, old_logits_np, returns = (
            states[keep], taken_dirs[keep], taken_boosts[keep], old_logits_np[keep], returns[keep]
        )
        print(f"Filtered to {keep.sum()} examples with return >= {args.positive}")

    sx = torch.from_numpy(states)
    td = torch.from_numpy(taken_dirs)
    tb = torch.from_numpy(taken_boosts)
    old_logits = torch.from_numpy(old_logits_np)

    adv = torch.from_numpy(returns)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(states)
    batch_size = min(args.batch_size, n)
    taken_idx = angle_to_bin(td[:, 0], td[:, 1])

    with torch.no_grad():
        old_log_dir = F.log_softmax(old_logits[:, :K_ANGLE_BINS], dim=1).gather(1, taken_idx.unsqueeze(1)).squeeze(1)
        old_log_boost = -F.binary_cross_entropy_with_logits(old_logits[:, K_ANGLE_BINS], tb, reduction='none')
        old_log_prob = (old_log_dir + old_log_boost).detach()

    print(f"n={n}  batch={batch_size}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'pg_loss':>8}  {'clip_frac':>9}")
    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(n)
        loss_sum = pg_sum = clip_sum = 0.0
        batches = 0
        for i in range(0, n - batch_size + 1, batch_size):
            mb = idx[i:i + batch_size]
            pred = model(sx[mb])

            log_dir = F.log_softmax(pred[:, :K_ANGLE_BINS], dim=1).gather(1, taken_idx[mb].unsqueeze(1)).squeeze(1)
            log_boost = -F.binary_cross_entropy_with_logits(pred[:, K_ANGLE_BINS], tb[mb], reduction='none')
            log_prob = log_dir + log_boost

            ratio = (log_prob - old_log_prob[mb]).exp()
            a = adv[mb]
            pg_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * a).mean()

            entropy = -(F.softmax(pred[:, :K_ANGLE_BINS], dim=1) * F.log_softmax(pred[:, :K_ANGLE_BINS], dim=1)).sum(dim=1).mean()
            loss = pg_loss - args.entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            clip_frac = ((ratio - 1).abs() > args.clip_eps).float().mean().item()
            loss_sum += loss.item()
            pg_sum += pg_loss.item()
            clip_sum += clip_frac
            batches += 1

        print(f"{resume_ep + epoch + 1:>5}  {loss_sum / batches:>8.4f}  "
              f"{pg_sum / batches:>8.4f}  {clip_sum / batches:>9.3f}", flush=True)

    torch.save({"model": model.state_dict(), "ep": resume_ep + args.epochs}, args.model_path)
    print(f"Saved {args.model_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=['supervised', 'rl', 'ppo'], default='supervised')
    p.add_argument("--experience",  default="experience.jsonl")
    p.add_argument("--labels",      default='labels.npy', help="advantage labels .npy (required for --positive and ppo mode)")
    p.add_argument("--start",       type=int,   default=0)
    p.add_argument("--count",       type=int,   default=None)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--decay",       type=float, default=0)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--clip-eps",    type=float, default=0.2, help="PPO clip epsilon")
    p.add_argument("--dropout-factor", type=float, default=0)
    p.add_argument("--positive",    type=float, default=None, help="use only positive advantage examples")
    p.add_argument("--freeze-fine", action="store_true")
    p.add_argument("--freeze-body", action="store_true")
    p.add_argument("--model-path",  default=PolicyNet.save_path)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    (train_ppo if args.mode == 'ppo' else train)(args)
