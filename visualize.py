#!/usr/bin/env python3
"""
Visualize examples from find_examples output.
Each line in the input file is a JSON array of frames for one example.

Controls:
  Space     - next example
  Backspace - previous example
  q         - quit
"""
import argparse
import json
import math
import sys

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import AVOID_DIST, bot_script


def polar_to_xy(angle_pi, dist):
    a = angle_pi * math.pi
    return dist * math.cos(a), dist * math.sin(a)


def load_examples(path):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


ENEMY_COLORS = ['#e74c3c', '#e67e22', '#9b59b6', '#1abc9c', '#3498db']
BG = '#1a1a2e'


def draw_frame(ax, frame):
    ax.clear()
    ax.set_aspect('equal')
    lim = 700
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_facecolor(BG)

    food_dat  = frame[1]
    own_props = frame[2]
    snakes    = frame[3]

    target_dir, target_boost, is_avoiding = bot_script(frame)

    # Avoid distance circle
    circle = plt.Circle((0, 0), AVOID_DIST, color='#e74c3c', fill=False,
                         linestyle='--', linewidth=1, alpha=0.3)
    ax.add_patch(circle)

    # Food
    for food in food_dat:
        sz, angle_pi, dist = food[0], food[1], food[2]
        x, y = polar_to_xy(angle_pi, dist)
        ax.plot(x, y, 'o', color='#f7dc6f', markersize=max(2, sz * 0.4), alpha=0.7)

    # Enemies
    for ei, snake in enumerate(snakes):
        color = ENEMY_COLORS[ei % len(ENEMY_COLORS)]
        hx, hy = polar_to_xy(snake[4], snake[5])
        # Body only (head drawn separately to avoid a closing line head→tail)
        xs, ys = [], []
        for j in range(6, len(snake) - 1, 2):
            bx, by = polar_to_xy(snake[j], snake[j + 1])
            xs.append(bx)
            ys.append(by)
        if xs:
            ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.5)
        ax.plot(hx, hy, 'o', color=color, markersize=8, zorder=5)

    # Own snake — head is at (0, 0), body drawn separately
    heading = own_props[1] * math.pi
    xs, ys = [], []
    for j in range(6, len(own_props) - 1, 2):
        bx, by = polar_to_xy(own_props[j], own_props[j + 1])
        xs.append(bx)
        ys.append(by)
    if xs:
        ax.plot(xs, ys, '-', color='#2ecc71', linewidth=2.5, alpha=0.8)
    ax.plot(0, 0, 'o', color='#2ecc71', markersize=10, zorder=6)

    # Heading arrow (white)
    r = 80
    ax.annotate('', xy=(math.cos(heading) * r, math.sin(heading) * r),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # Bot target arrow (yellow)
    ta = target_dir * math.pi
    ax.annotate('', xy=(math.cos(ta) * r * 0.7, math.sin(ta) * r * 0.7),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#f1c40f', lw=1.5))

    mode = 'AVOID' if is_avoiding else 'food'
    ax.text(-lim + 20, lim - 40, mode, color='#e74c3c' if is_avoiding else '#f7dc6f',
            fontsize=12, fontweight='bold')


def main():
    p = argparse.ArgumentParser(description="Visualize find_examples output")
    p.add_argument("path", help="JSONL file from find_examples --out-death / --out-survival")
    p.add_argument("--fps", type=int, default=5)
    args = p.parse_args()

    examples = load_examples(args.path)
    if not examples:
        print("No examples found.")
        sys.exit(1)
    print(f"Loaded {len(examples)} examples. Space=next  Backspace=prev  q=quit")

    state = {'ei': 0, 'fi': 0}

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)

    def update(_):
        ei, fi = state['ei'], state['fi']
        frames = examples[ei]
        draw_frame(ax, frames[fi])
        ax.set_title(f"Example {ei + 1}/{len(examples)}  frame {fi + 1}/{len(frames)}",
                     color='white', fontsize=11)
        state['fi'] = (fi + 1) % len(frames)

    def on_key(event):
        if event.key == ' ':
            state['ei'] = (state['ei'] + 1) % len(examples)
            state['fi'] = 0
            print(f"Example {state['ei'] + 1}/{len(examples)}")
        elif event.key == 'backspace':
            state['ei'] = (state['ei'] - 1) % len(examples)
            state['fi'] = 0
            print(f"Example {state['ei'] + 1}/{len(examples)}")
        elif event.key == 'q':
            plt.close('all')
            sys.exit(0)

    fig.canvas.mpl_connect('key_press_event', on_key)
    print(f"Example 1/{len(examples)}")

    ani = animation.FuncAnimation(fig, update, interval=1000 // args.fps, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
