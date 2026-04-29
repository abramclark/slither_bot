#!/usr/bin/env python3
"""Migrate a ValueNet checkpoint to the current 3-Dropout architecture.

Usage:
    python migrate_value_ckpt.py value.pt              # overwrites in place
    python migrate_value_ckpt.py value.pt value-new.pt # writes to new path
"""
import sys
import torch

src = sys.argv[1] if len(sys.argv) > 1 else "value.pt"
dst = sys.argv[2] if len(sys.argv) > 2 else src

ckpt  = torch.load(src, map_location="cpu", weights_only=True)
state = ckpt["model"]

if "head.2.weight" in state:
    # 0-Dropout: Linear at 0,2,4,6 → 3-Dropout: Linear at 0,3,6,9
    remap = {"head.2": "head.3", "head.4": "head.6", "head.6": "head.9"}
    print("Detected 0-Dropout checkpoint")
elif "head.5.weight" in state:
    # 1-Dropout: Linear at 0,3,5,7 → 3-Dropout: Linear at 0,3,6,9
    remap = {"head.5": "head.6", "head.7": "head.9"}
    print("Detected 1-Dropout checkpoint")
else:
    print("Already matches current architecture, nothing to do")
    sys.exit(0)

def remap_key(k):
    for old, new in remap.items():
        if k.startswith(old + "."):
            return new + k[len(old):]
    return k

ckpt["model"] = {remap_key(k): v for k, v in state.items()}
torch.save(ckpt, dst)
print(f"Saved → {dst}")
