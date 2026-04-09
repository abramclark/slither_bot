#!/usr/bin/env python3
import argparse
import json
from itertools import islice

import numpy as np
import torch
import torch.nn as nn

from model import BATCH_SIZE, LR, SAVE_PATH, ActorCritic, bot_script, get_flat


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
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            resume_ep = ckpt.get("ep", 0)
            resume_steps = ckpt.get("total_steps", 0)
        else:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Resumed from {save_path} (ep={resume_ep} steps={resume_steps})")
        if missing:
            print(f"Missing checkpoint keys: {missing}")
        if unexpected:
            print(f"Unexpected checkpoint keys: {unexpected}")
    except FileNotFoundError:
        print(f"Starting fresh, no checkpoint at {save_path}")
    return resume_ep, resume_steps


def load_states(path, start, count):
    states = []
    targets = []
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
            target_dir, _target_boost, is_avoiding = bot_script(d)
            try:
                x = get_flat(d).astype(np.float32)
            except Exception:
                continue
            x_aug = np.append(x, float(is_avoiding)).astype(np.float32)
            states.append(x_aug)
            targets.append(target_dir)
            used_rows += 1

    if not states:
        raise SystemExit("No usable supervised frames found in the selected range.")

    return np.stack(states), np.array(targets, dtype=np.float32), seen_rows, used_rows


def train_offline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic().to(device)
    resume_ep, resume_steps = load_checkpoint(model, device, args.save_path)

    states, targets, seen_rows, used_rows = load_states(args.path, args.start, args.count)
    print(f"Scanned rows: {seen_rows}")
    print(f"Usable supervised frames: {used_rows}")

    sx = torch.from_numpy(states).to(device)
    sy = torch.from_numpy(targets).to(device)

    optimizer = torch.optim.Adam(model.dir_focus.parameters(), lr=args.lr)
    model.train()

    n = len(states)
    batch_size = min(args.batch_size, n)
    total_steps = resume_steps

    print(f"Device: {device}")
    print(f"Training focused supervised head on rows [{args.start}:{'end' if args.count is None else args.start + args.count})")
    print(f"{'Epoch':>5}  {'loss':>8}  {'grad_norm':>10}  {'pred_std':>9}")

    for epoch in range(args.epochs):
        idx = torch.randperm(n, device=device)
        epoch_loss = 0.0
        grad_norm_sum = 0.0
        pred_std_sum = 0.0
        batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
            pred = model.supervised_dir(sx[mb])
            loss = circular_loss(pred, sy[mb])
            optimizer.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.dir_focus.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            grad_norm_sum += gn.item()
            pred_std_sum += pred.detach().std().item()
            batches += 1
            total_steps += len(mb)

        print(f"{epoch + 1:>5}  {epoch_loss / batches:>8.4f}  {grad_norm_sum / batches:>10.4f}  {pred_std_sum / batches:>9.4f}")

    torch.save(
        {"model": model.state_dict(), "ep": resume_ep, "total_steps": total_steps},
        args.save_path,
    )
    print(f"Saved {args.save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline supervised update for model.pt from experience.jsonl")
    parser.add_argument("--path", default="experience.jsonl", help="JSONL file to read from")
    parser.add_argument("--start", type=int, default=0, help="0-based line offset to start reading")
    parser.add_argument("--count", type=int, default=None, help="Maximum number of lines to read")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs over the selected slice")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--save-path", default=SAVE_PATH, help="Checkpoint file to update")
    return parser.parse_args()


if __name__ == "__main__":
    train_offline(parse_args())
