#!/usr/bin/env python3
import argparse
import json
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BATCH_SIZE, IS_AVOIDING_INDEX, LR, SAVE_PATH, ActorCritic, bot_script, get_flat
from runtime import load_compatible_state_dict


def circular_loss(pred, target):
    err = (pred - target) % 2
    err = torch.where(err > 1, err - 2, err)
    return (err ** 2).mean()


def load_checkpoint(model, device, save_path):
    resume_ep = 0
    resume_steps = 0
    try:
        ckpt = torch.load(save_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            missing, unexpected, skipped = load_compatible_state_dict(model, ckpt["model"])
            resume_ep = ckpt.get("ep", 0)
            resume_steps = ckpt.get("total_steps", 0)
        else:
            missing, unexpected, skipped = load_compatible_state_dict(model, ckpt)
        print(f"Resumed from {save_path} (ep={resume_ep} steps={resume_steps})")
        if skipped:
            print(f"Skipped incompatible keys: {skipped}")
    except FileNotFoundError:
        print(f"Starting fresh, no checkpoint at {save_path}")
    return resume_ep, resume_steps


def load_states(path, start, count):
    states = []
    dir_targets = []
    boost_targets = []
    seen_rows = 0
    used_rows = 0

    with open(path) as f:
        lines = islice(f, start, None if count is None else start + count)
        for line in lines:
            seen_rows += 1
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not isinstance(d, list) or len(d) != 4:
                continue
            target_dir, target_boost, is_avoiding = bot_script(d)
            try:
                x = get_flat(d).astype(np.float32)
            except Exception:
                continue
            x_aug = np.append(x, float(is_avoiding)).astype(np.float32)
            states.append(x_aug)
            dir_targets.append(target_dir)
            boost_targets.append(int(target_boost))
            used_rows += 1

    if not states:
        raise SystemExit("No usable supervised frames found in the selected range.")

    return np.stack(states), np.array(dir_targets, dtype=np.float32), np.array(boost_targets, dtype=np.int64), seen_rows, used_rows


def train_offline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic().to(device)
    resume_ep, resume_steps = load_checkpoint(model, device, args.model_path)

    if args.reset_focus:
        for m in [model.avoid_focus, model.food_focus, model.boost_focus]:
            for layer in m:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        print("Focus heads reset to fresh weights")

    states, dir_targets, boost_targets, seen_rows, used_rows = load_states(args.path, args.start, args.count)
    print(f"Scanned rows: {seen_rows}")
    print(f"Usable supervised frames: {used_rows}")

    sx = torch.from_numpy(states).to(device)
    sy = torch.from_numpy(dir_targets).to(device)
    sb = torch.from_numpy(boost_targets).to(device)

    optimizer = torch.optim.Adam(model.focus_parameters(), lr=args.lr)
    model.train()

    n = len(states)
    batch_size = min(args.batch_size, n)
    total_steps = resume_steps

    avoid_mask = sx[:, IS_AVOIDING_INDEX] > 0.5
    n_avoid = int(avoid_mask.sum().item())
    n_food  = n - n_avoid
    print(f"Device: {device}")
    print(f"Training focused supervised head on rows [{args.start}:{'end' if args.count is None else args.start + args.count})")
    print(f"Avoid frames: {n_avoid}  Food frames: {n_food}")
    print(f"{'Epoch':>5}  {'avoid_loss':>10}  {'food_loss':>9}  {'boost_loss':>10}  {'grad_norm':>10}  {'pred_std':>9}")

    for epoch in range(args.epochs):
        idx = torch.randperm(n, device=device)
        avoid_loss_sum = 0.0
        food_loss_sum  = 0.0
        boost_loss_sum = 0.0
        avoid_batches  = 0
        food_batches   = 0
        boost_batches  = 0
        grad_norm_sum  = 0.0
        pred_std_sum   = 0.0
        batches        = 0

        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
            mb_avoid = avoid_mask[mb]
            pred = model.supervised_dir(sx[mb])
            loss = circular_loss(pred, sy[mb])
            if mb_avoid.any():
                boost_logits = model.boost_focus_logits(sx[mb][mb_avoid])
                loss = loss + F.cross_entropy(boost_logits, sb[mb][mb_avoid])
            optimizer.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.focus_parameters(), args.grad_clip)
            optimizer.step()

            if mb_avoid.any():
                avoid_loss_sum += circular_loss(pred[mb_avoid].detach(), sy[mb][mb_avoid]).item()
                avoid_batches  += 1
                boost_loss_sum += F.cross_entropy(model.boost_focus_logits(sx[mb][mb_avoid]).detach(), sb[mb][mb_avoid]).item()
                boost_batches  += 1
            if (~mb_avoid).any():
                food_loss_sum  += circular_loss(pred[~mb_avoid].detach(), sy[mb][~mb_avoid]).item()
                food_batches   += 1

            grad_norm_sum += gn.item()
            pred_std_sum  += pred.detach().std().item()
            batches       += 1
            total_steps   += len(mb)

        avoid_loss_str = f"{avoid_loss_sum / avoid_batches:>10.4f}" if avoid_batches else f"{'n/a':>10}"
        food_loss_str  = f"{food_loss_sum  / food_batches:>9.4f}"  if food_batches  else f"{'n/a':>9}"
        boost_loss_str = f"{boost_loss_sum / boost_batches:>10.4f}" if boost_batches else f"{'n/a':>10}"
        print(f"{epoch + 1:>5}  {avoid_loss_str}  {food_loss_str}  {boost_loss_str}  {grad_norm_sum / batches:>10.4f}  {pred_std_sum / batches:>9.4f}", flush=True)

    torch.save(
        {"model": model.state_dict(), "ep": resume_ep, "total_steps": total_steps},
        args.model_path,
    )
    print(f"Saved {args.model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline supervised update for model.pt from experience.jsonl")
    parser.add_argument("--path", default="experience.jsonl", help="JSONL file to read from")
    parser.add_argument("--start", type=int, default=0, help="0-based line offset to start reading")
    parser.add_argument("--count", type=int, default=None, help="Maximum number of lines to read")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs over the selected slice")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--grad-clip", type=float, default=2.0, help="Gradient clip norm (default 2.0)")
    parser.add_argument("--reset-focus", action="store_true", help="Re-initialize focus head weights before training")
    parser.add_argument("--model-path", default=SAVE_PATH, help="Checkpoint file to load from and save to")
    return parser.parse_args()


if __name__ == "__main__":
    train_offline(parse_args())
