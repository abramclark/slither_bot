#!/usr/bin/env python3
"""
Search experience.jsonl for two types of avoidance examples:

  death    -- avoidance onset within the last N frames before death
  survival -- avoidance onset followed by at least N frames of survival

For each match, outputs the N-frame slice starting at the avoidance
transition as a JSON array (one episode per line).
"""
import argparse
import json
import statistics

from model import bot_script


def annotate_episode(frames):
    """Return list of (frame, is_avoiding) for a complete episode."""
    result = []
    for frame in frames:
        try:
            _, _, is_avoiding = bot_script(frame)
        except Exception:
            is_avoiding = False
        result.append((frame, is_avoiding))
    return result


def find_avoid_deaths(path, n, limit=None):
    """
    Yield (slice_frames, steps_before_death) for each episode where avoidance
    onset occurs within the last N frames before death.
    """
    total_episodes = 0
    matched = 0
    with open(path) as f:
        episode = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            if not d:  # death marker []
                if episode:
                    total_episodes += 1
                    annotated = annotate_episode(episode)

                    window = annotated[-n:]
                    avoid_start = None
                    for i, (_, is_avoiding) in enumerate(window):
                        if is_avoiding:
                            avoid_start = i
                            break

                    if avoid_start is not None:
                        steps_before_death = len(window) - avoid_start
                        slice_frames = [frame for frame, _ in window[avoid_start:]]
                        matched += 1
                        yield slice_frames, steps_before_death
                        if limit and matched >= limit:
                            return

                episode = []
            elif isinstance(d, list) and len(d) == 4:
                episode.append(d)


def find_avoid_survivals(path, n, limit=None):
    """
    Yield (slice_frames, frames_survived) for each avoidance transition
    followed by at least N frames of survival within the same episode.
    """
    matched = 0
    with open(path) as f:
        episode = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            if not d:  # death marker []
                if episode:
                    annotated = annotate_episode(episode)

                    prev_avoiding = False
                    for i, (_, is_avoiding) in enumerate(annotated):
                        if is_avoiding and not prev_avoiding:
                            frames_after = len(annotated) - i
                            if frames_after >= n:
                                slice_frames = [frame for frame, _ in annotated[i:i + n]]
                                matched += 1
                                yield slice_frames, frames_after
                                if limit and matched >= limit:
                                    return
                        prev_avoiding = is_avoiding

                episode = []
            elif isinstance(d, list) and len(d) == 4:
                episode.append(d)


def write_examples(examples, path):
    """Write an iterable of (slice_frames, metric) to a JSONL file."""
    count = 0
    with open(path, "w") as f:
        for slice_frames, _ in examples:
            f.write(json.dumps(slice_frames) + "\n")
            count += 1
    return count


def print_distribution(values, label):
    print(f"\n  {label}:")
    print(f"    mean={statistics.mean(values):.1f}  "
          f"median={statistics.median(values):.1f}  "
          f"min={min(values)}  max={max(values)}")
    buckets = {}
    for v in values:
        buckets[v] = buckets.get(v, 0) + 1
    print(f"    Distribution:")
    for k in sorted(buckets):
        bar = "#" * min(buckets[k], 60)
        print(f"    {k:3d}: {bar} {buckets[k]}")


def main():
    p = argparse.ArgumentParser(description="Find avoidance examples in experience.jsonl")
    p.add_argument("--path", default="experience.jsonl")
    p.add_argument("--n", type=int, default=16, help="Window size in frames")
    p.add_argument("--mode", choices=["death", "survival", "both"], default="both",
                   help="death: avoidance within N frames of death; "
                        "survival: avoidance followed by N frames of survival")
    p.add_argument("--out-death", default=None, help="Write death examples to this file (JSONL)")
    p.add_argument("--out-survival", default=None, help="Write survival examples to this file (JSONL)")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many matches per mode")
    args = p.parse_args()

    if args.mode in ("death", "both"):
        print(f"[death] Scanning {args.path!r} for avoidance onset within {args.n} frames of death...")
        examples = list(find_avoid_deaths(args.path, args.n, args.limit))
        onsets = [m for _, m in examples]
        print(f"  Matched : {len(examples)}")
        if onsets:
            print_distribution(onsets, "steps before death at avoidance onset")
        if args.out_death:
            count = write_examples(iter(examples), args.out_death)
            print(f"  Wrote {count} slices to {args.out_death!r}")

    if args.mode in ("survival", "both"):
        print(f"\n[survival] Scanning {args.path!r} for avoidance onset followed by {args.n}+ survival frames...")
        examples = list(find_avoid_survivals(args.path, args.n, args.limit))
        lengths = [m for _, m in examples]
        print(f"  Matched : {len(examples)}")
        if lengths:
            print_distribution(lengths, "frames survived after avoidance onset")
        if args.out_survival:
            count = write_examples(iter(examples), args.out_survival)
            print(f"  Wrote {count} slices to {args.out_survival!r}")


if __name__ == "__main__":
    main()
