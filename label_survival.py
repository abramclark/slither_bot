#!/usr/bin/env python3
"""
Label each frame in experience.jsonl with frames-until-death.

Reads JSONL from stdin, writes int32 labels as a .npy file to stdout.
  >= 1  : frames remaining until death in that episode
  -1    : death marker, invalid line, or trailing incomplete episode (skip in training)

Episodes are delimited by [] (empty-list death markers written by server.py).
Trailing frames after the last death marker are left as -1 because we don't
know when that episode ends.

Usage:
    python label_survival.py < experience.jsonl > survival_labels.npy
"""
import json
import sys

import numpy as np


def label():
    lines = []
    for raw in sys.stdin:
        raw = raw.strip()
        if raw:
            lines.append(raw)

    labels = np.full(len(lines), -1, dtype=np.int32)

    episode: list[int] = []
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
            for rank, li in enumerate(episode):
                labels[li] = T - rank   # T, T-1, ..., 1
            episode.clear()
            n_episodes += 1
        elif isinstance(d, list) and len(d) == 4:
            episode.append(i)
        else:
            n_skipped += 1

    trailing = len(episode)

    np.save(sys.stdout.buffer, labels)

    total_labeled = int((labels >= 1).sum())
    print(f"Episodes:          {n_episodes}",          file=sys.stderr)
    print(f"Labeled frames:    {total_labeled}",       file=sys.stderr)
    print(f"Trailing (no death marker): {trailing}",   file=sys.stderr)
    print(f"Skipped/invalid:   {n_skipped}",           file=sys.stderr)
    if total_labeled:
        valid = labels[labels >= 1]
        print(f"Survival stats:  mean={valid.mean():.1f}  median={np.median(valid):.1f}  "
              f"max={valid.max()}  p90={np.percentile(valid, 90):.1f}", file=sys.stderr)


if __name__ == "__main__":
    label()
