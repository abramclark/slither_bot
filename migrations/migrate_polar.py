#!/usr/bin/env python3
"""
Migrate experience JSONL from polar format to Cartesian format.

Old format: meta angles normalized by π (range [-1, 1]); body/head/food
            positions as (angle/π, dist) polar pairs.
New format: meta angles in radians; positions as (x, y) Cartesian pairs.

Usage:
    python migrate_polar.py < old.jsonl > new.jsonl
"""
import json
import math
import sys


def p2xy(angle_pi, dist):
    a = angle_pi * math.pi
    return dist * math.cos(a), dist * math.sin(a)


def migrate_props(props):
    result = [props[0] * math.pi, props[1] * math.pi, props[2], props[3]]
    i = 4
    while i + 1 < len(props):
        result.extend(p2xy(props[i], props[i + 1]))
        i += 2
    return result


def migrate_frame(d):
    food   = [[f[0], *p2xy(f[1], f[2])] for f in d[1]]
    own    = migrate_props(d[2])
    snakes = [migrate_props(s) for s in d[3]]
    return [d[0], food, own, snakes]


for raw in sys.stdin:
    stripped = raw.strip()
    if not stripped:
        continue
    d = json.loads(stripped)
    if not d:
        sys.stdout.write("[]\n")
    else:
        sys.stdout.write(json.dumps(migrate_frame(d)) + "\n")
