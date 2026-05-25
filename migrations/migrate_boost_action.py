#!/usr/bin/env python3
"""Migrate experience.jsonl: convert action[2][-1] from -1 or 1 to output from policy"""
import json
import sys

import numpy as np
import torch

from model import PolicyNet, get_flat

pn = PolicyNet()
pn.load()
migrated = skipped = 0
for raw in sys.stdin:
    raw = raw.rstrip("\n")
    if not raw:
        continue
    d = json.loads(raw)
    if d == [] or not isinstance(d, list):
        sys.stdout.write(raw + "\n")
        skipped += 1
        continue
    state, action, improv = d
    if isinstance(action[2], list) and action[2][-1] in (-1, 1):
        action[2][-1] = pn(torch.from_numpy(get_flat(state)))[-1].item()
        migrated += 1
    else:
        skipped += 1
    sys.stdout.write(json.dumps([state, action, improv]) + "\n")

print(f"migrated={migrated}  unchanged={skipped}", file=sys.stderr)
