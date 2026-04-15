#!/usr/bin/env python3
"""Compare model predictions vs bot_script on experience data."""
import argparse
import json
from itertools import islice

import numpy as np
import torch

from model import AVOID_DIST, IS_AVOIDING_INDEX, SAVE_PATH, ActorCritic, bot_script, get_flat
from runtime import load_compatible_state_dict


def load_model(path, device):
    model = ActorCritic().to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, _, skipped = load_compatible_state_dict(model, state)
    if skipped:
        print(f"Skipped: {skipped}")
    model.eval()
    return model


def angle_err(pred, target):
    """Circular angle error in [-1, 1] space, returned as absolute value."""
    e = (pred - target) % 2
    if e > 1:
        e -= 2
    return abs(e)


def run(args):
    device = torch.device("cpu")
    model = load_model(args.model_path, device)

    dir_errs_food  = []
    dir_errs_avoid = []
    boost_correct  = 0
    boost_total    = 0
    boost_tp = boost_fp = boost_tn = boost_fn = 0
    n_parsed = 0

    with open(args.path) as f:
        lines = islice(f, args.start, None if args.count is None else args.start + args.count)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not isinstance(d, list) or len(d) != 4:
                continue

            target_dir, target_boost, is_avoiding = bot_script(d)
            try:
                x = get_flat(d).astype(np.float32)
            except Exception:
                continue

            x_aug = np.append(x, float(is_avoiding)).astype(np.float32)
            xt = torch.from_numpy(x_aug).unsqueeze(0)

            with torch.no_grad():
                pred_dir, pred_boost, _, _, _ = model.act(xt)

            err = angle_err(pred_dir, target_dir)
            if is_avoiding:
                dir_errs_avoid.append(err)
                pred_b = pred_boost
                true_b = int(target_boost)
                if true_b == 1 and pred_b == 1: boost_tp += 1
                elif true_b == 0 and pred_b == 1: boost_fp += 1
                elif true_b == 0 and pred_b == 0: boost_tn += 1
                else:                              boost_fn += 1
                boost_correct += (pred_b == true_b)
                boost_total   += 1
            else:
                dir_errs_food.append(err)

            n_parsed += 1
            if args.count and n_parsed >= args.count:
                break

    def pct(errs, thresh):
        return 100 * (errs < thresh).mean() if len(errs) else 0.0

    def stats(errs, label):
        if not errs:
            print(f"  {label}: no samples")
            return
        a = np.array(errs)
        print(f"  {label} (n={len(a)}): "
              f"mean={a.mean():.4f}  median={np.median(a):.4f}  "
              f"p90={np.percentile(a,90):.4f}  "
              f"<0.1: {pct(a,0.1):.1f}%  <0.3: {pct(a,0.3):.1f}%  <0.5: {pct(a,0.5):.1f}%")

    print(f"\nDirection error (circular, abs, in units of π):")
    stats(dir_errs_food,  "food ")
    stats(dir_errs_avoid, "avoid")

    if boost_total:
        acc = 100 * boost_correct / boost_total
        prec = 100 * boost_tp / (boost_tp + boost_fp) if (boost_tp + boost_fp) else 0
        rec  = 100 * boost_tp / (boost_tp + boost_fn) if (boost_tp + boost_fn) else 0
        print(f"\nBoost (avoid frames only, n={boost_total}):")
        print(f"  accuracy={acc:.1f}%  precision={prec:.1f}%  recall={rec:.1f}%")
        print(f"  TP={boost_tp}  FP={boost_fp}  TN={boost_tn}  FN={boost_fn}")
        print(f"  base rate (boost=1): {100*(boost_tp+boost_fn)/boost_total:.1f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="experience.jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=5000)
    p.add_argument("--model-path", default=SAVE_PATH)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
