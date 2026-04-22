#!/usr/bin/env python3
"""
Compute value function labels for each frame in experience.jsonl.

  No death in N frames:  V = size(t+N) - size(t) - 0.053 * N
  Death at frame k <= N: V = -size(t) - 0.053 * k - 1
                             (size drops to 0; -1 is the death penalty)

The death case is always negative: -size(t) < 0, -0.053*k < 0, -1 < 0.
The survival case is positive when the snake grows faster than average.

Invalid/death-marker/trailing frames get label NaN.

Usage:
    python label_value.py < experience-t0.jsonl > value_labels.npy
    python label_value.py --horizon 30 < experience-t0.jsonl > value_labels.npy
"""
import argparse
import json
import sys

import numpy as np


def label(horizon):
    lines = []
    for raw in sys.stdin:
        raw = raw.strip()
        if raw:
            lines.append(raw)

    labels = np.full(len(lines), np.nan, dtype=np.float32)

    episode: list[tuple[int, float]] = []  # (line_idx, size)
    n_episodes = 0
    n_skipped  = 0

    for i, raw in enumerate(lines):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            n_skipped += 1
            continue

        if isinstance(d, list) and len(d) == 0:
            T = len(episode)
            for t, (li, size_t) in enumerate(episode):
                k = T - t  # frames until death (1 for the last frame)
                if k <= horizon:
                    labels[li] = -size_t - 0.053 * k - 1.0
                else:
                    _, size_end = episode[t + horizon]
                    labels[li] = size_end - size_t - 0.053 * horizon
            episode.clear()
            n_episodes += 1
        elif isinstance(d, list) and len(d) == 4:
            episode.append((i, float(d[0])))
        else:
            n_skipped += 1

    trailing = len(episode)

    np.save(sys.stdout.buffer, labels)

    valid = labels[~np.isnan(labels)]
    print(f"Horizon:        {horizon}",                  file=sys.stderr)
    print(f"Episodes:       {n_episodes}",               file=sys.stderr)
    print(f"Labeled frames: {len(valid)}",               file=sys.stderr)
    print(f"Trailing:       {trailing}",                 file=sys.stderr)
    print(f"Skipped:        {n_skipped}",                file=sys.stderr)
    if len(valid):
        n_death = int((valid < -1.0).sum())
        print(f"Death cases:    {n_death} / {len(valid)} "
              f"({100 * n_death / len(valid):.1f}%)", file=sys.stderr)
        print(f"Value stats:    mean={valid.mean():.3f}  std={valid.std():.3f}  "
              f"min={valid.min():.3f}  max={valid.max():.3f}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=20)
    args = p.parse_args()
    label(args.horizon)


if __name__ == "__main__":
    main()
