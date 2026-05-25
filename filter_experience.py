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
    python filter_experience.py --experience experience.jsonl > training.jsonl
    python filter_experience.py --n 30 --avoid-factor 1.2 --experience exp.jsonl > training.jsonl
    python filter_experience.py --labels labels.npy --labels-out filtered_labels.npy \\
        --experience experience.jsonl > training.jsonl
"""
import argparse
import json
import sys
import tempfile

import numpy as np

from environment import get_flat, AVOIDX_INDEX, AVOID_DIST


def filter(in_file, n, avoid_factor, tmp_dir='.'):
    tmp = tempfile.TemporaryFile(mode='w+', dir=tmp_dir)
    episode = []  # (tmp_offset, line_idx)
    line_idx = 0
    death_eps = []
    avoid_eps = []

    def flush_death():
        death_eps.append(list(episode[-n:]))

    def flush_life():
        tmp.flush()
        T = len(episode)
        i = 0
        while i < T:
            off, li = episode[i]
            tmp.seek(off)
            x = get_flat(json.loads(tmp.readline())[0])
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
            tmp.seek(0, 2)
            offset = tmp.tell()
            tmp.write(stripped + '\n')
            episode.append((offset, line_idx))
        line_idx += 1

    return death_eps, avoid_eps, tmp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",            type=int,   default=50)
    p.add_argument("--avoid-factor", type=float, default=1, help="Multiple of AVOID_DIST to start avoid segment")
    p.add_argument("--death-only",   action="store_true", help="output only death episodes")
    p.add_argument("--labels",       help="input labels .npy from label_value.py")
    p.add_argument("--labels-out",   help="output labels .npy aligned to filtered frames")
    p.add_argument("--idx-out",      help="output line-index .npy aligned to filtered frames")
    p.add_argument("--tmp-dir",      default=".", help="directory for temporary buffer file (default: .)")
    p.add_argument("--experience",   default="experience.jsonl")
    args = p.parse_args()

    in_labels   = np.load(args.labels) if args.labels else None
    out_labels  = []
    out_indices = []

    death_eps, avoid_eps, tmp = filter(sys.stdin, args.n, args.avoid_factor, args.tmp_dir)

    def write_ep(frames):
        for off, li in frames:
            tmp.seek(off)
            sys.stdout.write(tmp.readline())
            if in_labels is not None:
                out_labels.append(in_labels[li])
            out_indices.append(li)
        sys.stdout.write("[]\n")
        if in_labels is not None:
            out_labels.append(float("nan"))
        out_indices.append(-1)

    if args.death_only:
        print(f"death={len(death_eps)}  (death-only mode)", file=sys.stderr)
        for ep in death_eps:
            write_ep(ep)
        if args.labels_out and out_labels:
            np.save(args.labels_out, np.array(out_labels, dtype=np.float32))
        if args.idx_out:
            np.save(args.idx_out, np.array(out_indices, dtype=np.int64))
        return

    k = min(len(death_eps), len(avoid_eps))
    print(f"death={len(death_eps)}  avoid={len(avoid_eps)}  keeping {k} of each", file=sys.stderr)
    rng = np.random.default_rng(0)
    rng.shuffle(death_eps)
    if in_labels is not None:
        avoid_eps.sort(key=lambda ep: np.mean([in_labels[li] for _, li in ep]), reverse=True)
    else:
        rng.shuffle(avoid_eps)

    for ep in death_eps[:k] + avoid_eps[:k]:
        write_ep(ep)

    tmp.close()

    if args.labels_out and out_labels:
        np.save(args.labels_out, np.array(out_labels, dtype=np.float32))
    if args.idx_out:
        np.save(args.idx_out, np.array(out_indices, dtype=np.int64))


if __name__ == "__main__":
    main()
