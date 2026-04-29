#!/usr/bin/env python3
"""
Compute return labels for each frame in experience.jsonl.

Labels are the sum of two independent components:
  alive_norm: discounted alive returns (r=0 at terminal), normalized to mean=1 std=1
  death_disc: death_value * gamma^(T-1-t), so terminal frame gets ~death_value

Usage:
    python label_value.py < experience-t0.jsonl > value_labels.npy
    python label_value.py --gamma 0.95 --death-value -20 < experience.jsonl > labels.npy
"""
import argparse
import json
import sys

import numpy as np


def label(gamma, death_value, no_growth=False, no_death=False, death_offset=3):
    lines = []
    for raw in sys.stdin:
        raw = raw.strip()
        if raw:
            lines.append(raw)

    labels = np.full(len(lines), np.nan, dtype=np.float32)

    episodes: list[list[tuple[int, float]]] = []
    episode:  list[tuple[int, float]]       = []
    n_skipped = 0

    for i, raw in enumerate(lines):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            n_skipped += 1
            continue
        if isinstance(d, list) and len(d) == 0:
            if episode:
                episodes.append(episode)
                episode = []
        elif isinstance(d, list) and len(d) == 4:
            episode.append((i, float(d[0])))
        else:
            n_skipped += 1

    trailing = len(episode)

    deltas = [episode[t + 1][1] - episode[t][1]
              for episode in episodes
              for t in range(len(episode) - 1)]
    avg_value = float(np.mean(deltas)) if deltas else 0.0

    # Pass 1: life-only returns (r=0 at terminal) for normalization stats
    labels_alive = np.full(len(lines), np.nan, dtype=np.float32)
    for episode in episodes:
        T = len(episode)
        G = 0.0
        for t in reversed(range(T)):
            li, size_t = episode[t]
            r = 0.0 if t == T - 1 else episode[t + 1][1] - size_t - avg_value
            G = r + gamma * G
            labels_alive[li] = G

    alive_vals = labels_alive[~np.isnan(labels_alive)]
    mean, std = float(alive_vals.mean()), float(alive_vals.std())

    # Pass 2: alive_norm (mean=1) + discounted death signal
    # Frames within death_offset of the death marker are left as NaN.
    for episode in episodes:
        T = len(episode)
        for t in range(T - death_offset):
            li, _ = episode[t]
            alive_norm = 0.0 if no_growth else (labels_alive[li] - mean) / (std + 1e-8) + 1.0
            death_disc = 0.0 if no_death else death_value * (gamma ** (T - death_offset - 1 - t))
            labels[li] = alive_norm + death_disc

    valid_mask = ~np.isnan(labels)
    n_labeled = int(valid_mask.sum())

    np.save(sys.stdout.buffer, labels)

    normed = labels[valid_mask]
    print(f"Gamma:          {gamma}",              file=sys.stderr)
    print(f"Avg value/frame:{avg_value:.5f}",      file=sys.stderr)
    print(f"Episodes:       {len(episodes)}",      file=sys.stderr)
    print(f"Labeled frames: {n_labeled}",          file=sys.stderr)
    print(f"Trailing:       {trailing}",           file=sys.stderr)
    print(f"Skipped:        {n_skipped}",          file=sys.stderr)
    print(f"Alive-only: mean={mean:.3f}  std={std:.3f}", file=sys.stderr)
    print(f"Labels: mean={normed.mean():.3f}  std={normed.std():.3f}  "
          f"min={normed.min():.3f}  max={normed.max():.3f}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--death-value",  type=float, default=-20.0)
    p.add_argument("--no-growth",    action="store_true")
    p.add_argument("--no-death",     action="store_true")
    p.add_argument("--death-offset", type=int, default=3)
    args = p.parse_args()
    label(args.gamma, args.death_value, args.no_growth, args.no_death, args.death_offset)


if __name__ == "__main__":
    main()
