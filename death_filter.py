#!/usr/bin/env python3
"""
Filter experience frames by proximity to death.

Death mode (default): outputs the N frames before each death marker.
Life mode (--life):   outputs N-frame segments that begin when an enemy
                      is within --avoid-dist and whose last frame is more
                      than N/2 frames before death (contrastive survival examples).

Use --labels / --out-labels to pass pre-computed labels from label_value.py
through the filter so output labels align with output frames.

Usage:
    python death_filter.py        --n 30 < experience.jsonl > near_death.jsonl
    python death_filter.py --life --n 50 --avoid-dist 300 < exp.jsonl > survival.jsonl
    python death_filter.py --life --labels labels.npy --labels-out life_labels.npy \\
        < exp.jsonl > survival.jsonl
"""
import argparse
import json
import sys

import numpy as np

from model import get_flat, AVOIDX_INDEX, AVOID_DIST


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--life",       action="store_true")
    p.add_argument("--n",          type=int,   default=50)
    p.add_argument("--avoid-factor", type=float, default=1, help="Multiple of AVOID_DIST to start episode")
    p.add_argument("--labels",     help="input labels .npy from label_value.py")
    p.add_argument("--labels-out", help="output labels .npy aligned to filtered frames")
    args = p.parse_args()
    n, life = args.n, args.life

    in_labels    = np.load(args.labels) if args.labels else None
    out_labels   = []

    # episode stores (raw_line, line_index) pairs; line_index matches label_value.py's
    # enumeration of non-empty lines (including [] markers).
    episode  = []
    line_idx = 0

    def flush_death():
        for raw, li in episode[-n:]:
            sys.stdout.write(raw)
            if in_labels is not None:
                out_labels.append(in_labels[li])
        sys.stdout.write("[]\n")
        if in_labels is not None:
            out_labels.append(float("nan"))

    def flush_life():
        T = len(episode)
        i = 0
        while i < T:
            raw, li = episode[i]
            x = get_flat(json.loads(raw.strip()))
            if not len(x):
                i += 1
                continue
            end = i + n
            if (np.linalg.norm(x[AVOIDX_INDEX:AVOIDX_INDEX + 2]) < args.avoid_factor
                    and end <= T
                    and T - end > n // 2):
                for raw, li in episode[i:end]:
                    sys.stdout.write(raw)
                    if in_labels is not None:
                        out_labels.append(in_labels[li])
                sys.stdout.write("[]\n")
                if in_labels is not None:
                    out_labels.append(float("nan"))
                i = end
            else:
                i += 1

    for raw in sys.stdin:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped == "[]":
            if life:
                flush_life()
            else:
                flush_death()
            episode.clear()
        else:
            episode.append((raw, line_idx))
        line_idx += 1

    if episode and not life:
        flush_death()

    if args.labels_out and out_labels:
        np.save(args.labels_out, np.array(out_labels, dtype=np.float32))


if __name__ == "__main__":
    main()
