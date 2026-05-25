#!/usr/bin/env python3
"""Drop episodes whose last frame is an improv action."""
import json
import sys

kept = dropped = 0
episode = []

for raw in sys.stdin:
    raw = raw.rstrip("\n")
    if not raw:
        continue
    d = json.loads(raw)
    if d == []:
        if episode:
            _, _, improv = episode[-1]
            if improv:
                dropped += 1
            else:
                for frame in episode:
                    sys.stdout.write(json.dumps(frame) + "\n")
                sys.stdout.write("[]\n")
                kept += 1
        episode.clear()
    else:
        state, action, improv, *_ = d
        episode.append((state, action, improv))

print(f"kept={kept} dropped={dropped}", file=sys.stderr)
