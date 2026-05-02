#!/usr/bin/env python3
"""
Filter experience frames by proximity to death and avoidance survival.

Each episode emits two types of examples:
  Avoid segments: N-frame windows that begin when an enemy is within --avoid-factor
                  of AVOID_DIST and whose last frame is more than N/2 frames before death.
  Death window:   the last N frames before each death marker.

Use --labels / --labels-out to pass pre-computed labels from label_value.py
through the filter so output labels align with output frames.

Usage:
    python filter_experience.py < experience.jsonl > training.jsonl
    python filter_experience.py --n 30 --avoid-factor 1.2 < exp.jsonl > training.jsonl
    python filter_experience.py --labels labels.npy --labels-out filtered_labels.npy \\
        < experience.jsonl > training.jsonl
"""
import argparse
import json
import sys

import numpy as np

from model import get_flat, AVOIDX_INDEX, AVOID_DIST


def filter(in_file, n, avoid_factor):
    episode   = []
    line_idx  = 0
    death_eps = []
    avoid_eps = []

    def flush_death():
        death_eps.append(list(episode[-n:]))

    def flush_life():
        T = len(episode)
        i = 0
        while i < T:
            raw, li = episode[i]
            x = get_flat(json.loads(raw))
            end = i + n
            if (np.linalg.norm(x[AVOIDX_INDEX:AVOIDX_INDEX + 2]) < avoid_factor
                    and end <= T
                    and T - end > n):
                avoid_eps.append(list(episode[i:end]))
                i = end
            else:
                i += 1

    for raw in in_file:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped == "[]":
            if len(episode) > 10:
                flush_life()
                flush_death()
            episode.clear()
        else:
            episode.append((raw, line_idx))
        line_idx += 1

    return death_eps, avoid_eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",            type=int,   default=50)
    p.add_argument("--avoid-factor", type=float, default=1, help="Multiple of AVOID_DIST to start avoid segment")
    p.add_argument("--labels",       help="input labels .npy from label_value.py")
    p.add_argument("--labels-out",   help="output labels .npy aligned to filtered frames")
    args = p.parse_args()

    in_labels  = np.load(args.labels) if args.labels else None
    out_labels = []
    death_eps, avoid_eps = filter(sys.stdin, args.n, args.avoid_factor)

    def write_ep(frames):
        for raw, li in frames:
            sys.stdout.write(raw)
            if in_labels is not None:
                out_labels.append(in_labels[li])
        sys.stdout.write("[]\n")
        if in_labels is not None:
            out_labels.append(float("nan"))

    k = min(len(death_eps), len(avoid_eps))
    print(f"death={len(death_eps)}  avoid={len(avoid_eps)}  keeping {k} of each", file=sys.stderr)
    rng = np.random.default_rng(0)
    rng.shuffle(death_eps)
    rng.shuffle(avoid_eps)

    for ep in death_eps[:k] + avoid_eps[:k]:
        write_ep(ep)

    if args.labels_out and out_labels:
        np.save(args.labels_out, np.array(out_labels, dtype=np.float32))


if __name__ == "__main__":
    main()
