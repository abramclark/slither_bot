#!/usr/bin/env python3
"""
Offline contrastive training from paired death/survival avoidance examples.

Survival examples receive positive reward (reinforce bot_script actions).
Death examples receive negative reward (penalize bot_script actions).

The signed reward scales the supervised loss:
  loss = reward * circular_loss(pred, bot_target)

Positive reward minimizes loss (push toward target).
Negative reward maximizes loss (push away from target).
"""
import argparse
import json
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BATCH_SIZE, LR, SAVE_PATH, ActorCritic, bot_script, get_flat
from runtime import load_compatible_state_dict


def load_examples(path, offset, n):
    """Load up to n examples starting at line offset from a find_examples JSONL file."""
    examples = []
    with open(path) as f:
        for line in islice(f, offset, None if n is None else offset + n):
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def examples_to_frames(examples, reward_sign):
    """Flatten examples (list of frame lists) into parallel arrays of (state, dir, boost, reward)."""
    states, dir_targets, boost_targets, rewards = [], [], [], []
    for example in examples:
        for frame in example:
            try:
                target_dir, target_boost, is_avoiding = bot_script(frame)
                x = get_flat(frame).astype(np.float32)
                x_aug = np.append(x, float(is_avoiding)).astype(np.float32)
            except Exception:
                continue
            states.append(x_aug)
            dir_targets.append(float(target_dir))
            boost_targets.append(int(target_boost))
            rewards.append(reward_sign)
    return states, dir_targets, boost_targets, rewards


def circular_loss(pred, target):
    err = (pred - target) % 2
    err = torch.where(err > 1, err - 2, err)
    return err ** 2


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic().to(device)

    resume_ep = resume_steps = 0
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        _, _, skipped = load_compatible_state_dict(model, state)
        resume_ep = ckpt.get("ep", 0) if isinstance(ckpt, dict) else 0
        resume_steps = ckpt.get("total_steps", 0) if isinstance(ckpt, dict) else 0
        print(f"Resumed from {args.model_path} (ep={resume_ep} steps={resume_steps})")
        if skipped:
            print(f"Skipped: {skipped}")
    except FileNotFoundError:
        print("Starting fresh")

    death_ex    = load_examples(args.death_path,    args.offset, args.n)
    survival_ex = load_examples(args.survival_path, args.offset, args.n)
    print(f"Death examples: {len(death_ex)}  Survival examples: {len(survival_ex)}")

    d_s, d_d, d_b, d_r = examples_to_frames(death_ex,    -1.0)
    s_s, s_d, s_b, s_r = examples_to_frames(survival_ex, +1.0)

    all_states  = torch.from_numpy(np.array(d_s + s_s, dtype=np.float32)).to(device)
    all_dirs    = torch.from_numpy(np.array(d_d + s_d, dtype=np.float32)).to(device)
    all_boosts  = torch.from_numpy(np.array(d_b + s_b, dtype=np.int64)).to(device)
    all_rewards = torch.from_numpy(np.array(d_r + s_r, dtype=np.float32)).to(device)

    n_death    = len(d_s)
    n_survival = len(s_s)
    n_total    = len(d_s) + len(s_s)
    print(f"Total frames: {n_total}  ({n_death} death, {n_survival} survival)")
    print(f"Device: {device}")

    optimizer = torch.optim.Adam(model.focus_parameters(), lr=args.lr)
    model.train()

    n = n_total
    batch_size = min(args.batch_size, n)
    total_steps = resume_steps

    print(f"\n{'Epoch':>5}  {'pos_dir':>8}  {'neg_dir':>8}  {'boost':>8}  {'grad_norm':>10}")

    for epoch in range(args.epochs):
        idx = torch.randperm(n, device=device)
        pos_dir_sum = neg_dir_sum = boost_sum = grad_norm_sum = 0.0
        pos_batches = neg_batches = boost_batches = batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
            reward = all_rewards[mb]

            pred_dir    = model.supervised_dir(all_states[mb])
            boost_logits = model.boost_focus_logits(all_states[mb])

            dir_loss   = circular_loss(pred_dir, all_dirs[mb])
            boost_loss = F.cross_entropy(boost_logits, all_boosts[mb], reduction='none')

            # Signed loss: positive reward reinforces, negative reward penalizes
            loss = (reward * dir_loss + reward * boost_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.focus_parameters(), args.grad_clip)
            optimizer.step()

            pos_mask = reward > 0
            neg_mask = reward < 0
            if pos_mask.any():
                pos_dir_sum += dir_loss[pos_mask].detach().mean().item()
                pos_batches += 1
            if neg_mask.any():
                neg_dir_sum += dir_loss[neg_mask].detach().mean().item()
                neg_batches += 1
            boost_sum  += boost_loss.detach().mean().item(); boost_batches += 1
            grad_norm_sum += gn.item(); batches += 1
            total_steps += len(mb)

        pos_str   = f"{pos_dir_sum / pos_batches:>8.4f}" if pos_batches else f"{'n/a':>8}"
        neg_str   = f"{neg_dir_sum / neg_batches:>8.4f}" if neg_batches else f"{'n/a':>8}"
        boost_str = f"{boost_sum   / boost_batches:>8.4f}" if boost_batches else f"{'n/a':>8}"
        print(f"{epoch + 1:>5}  {pos_str}  {neg_str}  {boost_str}  {grad_norm_sum / batches:>10.4f}", flush=True)

    torch.save(
        {"model": model.state_dict(), "ep": resume_ep, "total_steps": total_steps},
        args.model_path,
    )
    print(f"\nSaved {args.model_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive offline training from death/survival examples")
    p.add_argument("--death-path",    required=True, help="JSONL from find_examples --out-death")
    p.add_argument("--survival-path", required=True, help="JSONL from find_examples --out-survival")
    p.add_argument("--n",      type=int,   default=None, help="Max examples to load from each file")
    p.add_argument("--offset", type=int,   default=0,    help="Line offset into each file")
    p.add_argument("--epochs",     type=int,   default=20,       help="Training epochs")
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--grad-clip",  type=float, default=2.0)
    p.add_argument("--model-path", default=SAVE_PATH)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
