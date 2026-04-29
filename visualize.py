#!/usr/bin/env python3
"""
Visualize experience JSONL files.
Frames are separated by [] episode markers.

Controls:
  Space     - next episode
  Backspace - previous episode
  q         - quit
"""
import argparse
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import (
    AVOID_DIST, bot_script, get_flat,
    K_FOOD, K_SNAKE, K_SEGMENTS,
    DIRX_INDEX, HEADINGX_INDEX,
    FOOD_START_INDEX, FOOD_END_INDEX,
    AVOIDX_INDEX, OWN_SEGMENTS_INDEX,
)


def load_examples(path):
    episode = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[]':
                if episode:
                    yield episode
                    episode = []
            else:
                d = json.loads(line)
                if isinstance(d, list) and len(d) == 4:
                    episode.append(d)
    if episode:
        yield episode


ENEMY_COLORS = ['#e74c3c', '#e67e22', '#9b59b6', '#1abc9c', '#3498db']
BG = '#1a1a2e'


def _setup_ax(ax):
    ax.clear()
    ax.set_aspect('equal')
    lim = 700
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_facecolor(BG)
    ax.add_patch(plt.Circle((0, 0), AVOID_DIST, color='#e74c3c', fill=False,
                             linestyle='--', linewidth=1, alpha=0.3))
    return lim


def draw_frame(ax, frame):
    lim = _setup_ax(ax)

    food_dat  = frame[1]
    own_props = frame[2]
    snakes    = frame[3]

    target_dir, target_boost, is_avoiding = bot_script(frame)

    # Food
    for food in food_dat:
        sz, x, y = food[0], food[1], food[2]
        ax.plot(x, y, 'o', color='#f7dc6f', markersize=max(2, sz * 0.4), alpha=0.7)

    # Enemies
    for ei, snake in enumerate(snakes):
        color = ENEMY_COLORS[ei % len(ENEMY_COLORS)]
        hx, hy = snake[4], snake[5]
        xs, ys = [], []
        for j in range(6, len(snake) - 1, 2):
            xs.append(snake[j])
            ys.append(snake[j + 1])
        if xs:
            ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.5)
        ax.plot(hx, hy, 'o', color=color, markersize=8, zorder=5)

    # Own snake — head is at (0, 0), body drawn separately
    heading = own_props[1]
    xs, ys = [], []
    for j in range(6, len(own_props) - 1, 2):
        xs.append(own_props[j])
        ys.append(own_props[j + 1])
    if xs:
        ax.plot(xs, ys, '-', color='#2ecc71', linewidth=2.5, alpha=0.8)
    ax.plot(0, 0, 'o', color='#2ecc71', markersize=10, zorder=6)

    r = 80
    ax.annotate('', xy=(math.cos(heading) * r, math.sin(heading) * r),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.annotate('', xy=(math.cos(target_dir) * r * 0.7, math.sin(target_dir) * r * 0.7),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#f1c40f', lw=1.5))

    mode = 'AVOID' if is_avoiding else 'food'
    ax.text(-lim + 20, lim - 40, mode, color='#e74c3c' if is_avoiding else '#f7dc6f',
            fontsize=12, fontweight='bold')


def draw_frame_flat(ax, x):
    lim = _setup_ax(ax)

    # Food — coordinates normalized by AVOID_DIST, skip zero-padded entries
    food_xy = x[FOOD_START_INDEX + 1:FOOD_END_INDEX].reshape(K_FOOD, 2) * AVOID_DIST
    for fx, fy in food_xy:
        if fx != 0 or fy != 0:
            ax.plot(fx, fy, 'o', color='#f7dc6f', markersize=4, alpha=0.7)

    # Enemy snakes — (K_SEGMENTS+1) segments each, first is the nearest
    snake_segs = (x[AVOIDX_INDEX:AVOIDX_INDEX + K_SNAKE * (K_SEGMENTS + 1) * 2]
                  .reshape(K_SNAKE, K_SEGMENTS + 1, 2) * AVOID_DIST)
    for ei in range(K_SNAKE):
        segs = snake_segs[ei]
        if not np.any(segs):
            continue
        color = ENEMY_COLORS[ei % len(ENEMY_COLORS)]
        hx, hy = segs[0]
        body = segs[1:]
        body = body[np.any(body != 0, axis=1)]
        if len(body):
            ax.plot(body[:, 0], body[:, 1], '-', color=color, linewidth=2, alpha=0.5)
        ax.plot(hx, hy, 'o', color=color, markersize=8, zorder=5)

    # Own snake body
    own_segs = x[OWN_SEGMENTS_INDEX:].reshape(K_SEGMENTS, 2) * AVOID_DIST
    own_segs = own_segs[np.any(own_segs != 0, axis=1)]
    if len(own_segs):
        ax.plot(own_segs[:, 0], own_segs[:, 1], '-', color='#2ecc71', linewidth=2.5, alpha=0.8)
    ax.plot(0, 0, 'o', color='#2ecc71', markersize=10, zorder=6)

    r = 80
    hx_v, hy_v = x[HEADINGX_INDEX], x[HEADINGX_INDEX + 1]
    ax.annotate('', xy=(hx_v * r, hy_v * r), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    tx_v, ty_v = x[DIRX_INDEX], x[DIRX_INDEX + 1]
    ax.annotate('', xy=(tx_v * r * 0.7, ty_v * r * 0.7), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#f1c40f', lw=1.5))

    nearest_dist = np.linalg.norm(x[AVOIDX_INDEX:AVOIDX_INDEX + 2]) * AVOID_DIST
    is_avoiding = nearest_dist < AVOID_DIST
    mode = 'AVOID' if is_avoiding else 'food'
    ax.text(-lim + 20, lim - 40, mode, color='#e74c3c' if is_avoiding else '#f7dc6f',
            fontsize=12, fontweight='bold')


def main():
    p = argparse.ArgumentParser(description="Visualize find_examples output")
    p.add_argument("path", help="experience JSONL file")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--flat", action="store_true", help="draw from get_flat() features")
    args = p.parse_args()

    gen      = load_examples(args.path)
    episodes = []
    done     = [False]

    def try_load_next():
        if done[0]:
            return False
        try:
            episodes.append(next(gen))
            return True
        except StopIteration:
            done[0] = True
            return False

    if not try_load_next():
        print("No episodes found.")
        sys.exit(1)
    print("Space=next  Backspace=prev  q=quit")

    state = {'ei': 0, 'fi': 0}

    def ep_label():
        total = str(len(episodes)) + ('' if done[0] else '+')
        return f"Episode {state['ei'] + 1}/{total}"

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)

    def update(_):
        ei, fi = state['ei'], state['fi']
        frames = episodes[ei]
        frame = frames[fi]
        if args.flat:
            draw_frame_flat(ax, get_flat(frame))
        else:
            draw_frame(ax, frame)
        ax.set_title(f"{ep_label()}  frame {fi + 1}/{len(frames)}",
                     color='white', fontsize=11)
        state['fi'] = (fi + 1) % len(frames)

    def on_key(event):
        if event.key == ' ':
            next_ei = state['ei'] + 1
            if next_ei >= len(episodes):
                if not try_load_next():
                    print("No more episodes.")
                    return
            state['ei'] = next_ei
            state['fi'] = 0
            print(ep_label())
        elif event.key == 'backspace':
            state['ei'] = max(0, state['ei'] - 1)
            state['fi'] = 0
            print(ep_label())
        elif event.key == 'q':
            plt.close('all')
            sys.exit(0)

    fig.canvas.mpl_connect('key_press_event', on_key)
    print(ep_label())

    ani = animation.FuncAnimation(fig, update, interval=1000 // args.fps, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
