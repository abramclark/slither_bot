#!/usr/bin/env python3
"""Migrate experience.jsonl: convert action[2] from 8-element probs to 9-element log-probs+boost."""
import json
import sys

import numpy as np

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
    if isinstance(action[2], list) and len(action[2]) == 8:
        log_probs = np.log(np.array(action[2], dtype=np.float32)).tolist()
        log_probs.append(float(action[1]) * 2 - 1)
        action[2] = log_probs
        migrated += 1
    else:
        skipped += 1
    sys.stdout.write(json.dumps([state, action, improv]) + "\n")

print(f"migrated={migrated}  unchanged={skipped}", file=sys.stderr)
