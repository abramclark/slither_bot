#!/usr/bin/env python3
"""
Divide food size (d[1][i][0]) by 13 for every frame in experience.

Usage:
    python migrate_food_sz.py < experience.jsonl > experience_migrated.jsonl
"""
import json
import sys

for raw in sys.stdin:
    stripped = raw.strip()
    if not stripped or stripped == '[]':
        sys.stdout.write(raw)
        continue
    d = json.loads(stripped)
    if isinstance(d, list) and len(d) == 4:
        for food in d[1]:
            food[0] /= 13
    sys.stdout.write(json.dumps(d) + '\n')
