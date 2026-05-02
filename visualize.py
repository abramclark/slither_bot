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
    K_BIG_FOOD, K_SM_FOOD, K_FOOD, K_SNAKE, K_SEGMENTS,
    DIRX_INDEX, HEADINGX_INDEX,
    FOOD_START_INDEX, FOOD_SM_INDEX, FOOD_END_INDEX,
    AVOIDX_INDEX,
)


def load_examples(path):
    episode    = []
    line_idx   = 0
    ep_start   = 0
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == '[]':
                if episode:
                    yield episode, ep_start
                    episode = []
            else:
                d = json.loads(line)
                if isinstance(d, list) and len(d) == 4:
                    if not episode:
                        ep_start = line_idx
                    episode.append(d)
            line_idx += 1
    if episode:
        yield episode, ep_start


ENEMY_COLORS = ['#e74c3c', '#e67e22', '#9b59b6', '#1abc9c', '#3498db']
BG = '#1a1a2e'


def _setup_ax(ax):
    ax.clear()
    ax.set_aspect('equal')
    lim = 1000
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_facecolor(BG)
    ax.add_patch(plt.Circle((0, 0), AVOID_DIST, color='#e74c3c', fill=False,
                             linestyle='--', linewidth=1, alpha=0.3))
    return lim


def draw_frame(ax, frame):
    lim = _setup_ax(ax)

    me, others, food_dat, timestamp = frame
    me_meta, me_segs = me

    target_dir, target_boost, is_avoiding = bot_script(frame)

    # Food
    for food in food_dat:
        sz, x, y = food[0], food[1], food[2]
        color = '#b4340f' if sz > 0.9 else '#f7dc6f'
        ax.plot(x, y, 'o', color=color, markersize=max(2, sz * 10), alpha=0.7)

    # Enemies
    for ei, snake in enumerate(others):
        color = ENEMY_COLORS[ei % len(ENEMY_COLORS)]
        s_meta, s_segs = snake
        if s_segs:
            hx, hy = s_segs[0][0], s_segs[0][1]
            xs = [seg[0] for seg in s_segs[1:]]
            ys = [seg[1] for seg in s_segs[1:]]
            if xs:
                ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.5)
            ax.plot(hx, hy, 'o', color=color, markersize=8, zorder=5)

    # Own snake — head is at (0, 0), body drawn from segments
    heading = me_meta[1]
    if me_segs:
        xs = [seg[0] for seg in me_segs]
        ys = [seg[1] for seg in me_segs]
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

    # Big food (first K_BIG_FOOD slots), then small food
    big_xy = x[FOOD_START_INDEX:FOOD_SM_INDEX].reshape(K_BIG_FOOD, 2) * AVOID_DIST
    sm_xy  = x[FOOD_SM_INDEX:FOOD_END_INDEX].reshape(K_SM_FOOD, 2) * AVOID_DIST
    for fx, fy in big_xy:
        if fx != 0 or fy != 0:
            ax.plot(fx, fy, 'o', color='#b4340f', markersize=8, alpha=0.7)
    for fx, fy in sm_xy:
        if fx != 0 or fy != 0:
            ax.plot(fx, fy, 'o', color='#f7dc6f', markersize=4, alpha=0.7)

    # Enemy snakes — 3 segments each: closest, head, last
    snake_segs = (x[AVOIDX_INDEX:AVOIDX_INDEX + K_SNAKE * K_SEGMENTS * 2]
                  .reshape(K_SNAKE, K_SEGMENTS, 2) * AVOID_DIST)
    for ei in range(K_SNAKE):
        segs = snake_segs[ei]
        if not np.any(segs):
            continue
        color = ENEMY_COLORS[ei % len(ENEMY_COLORS)]
        closest, head, last = segs[0], segs[1], segs[2]
        ax.plot([closest[0], head[0], last[0]], [closest[1], head[1], last[1]],
                '-', color=color, linewidth=2, alpha=0.5)
        ax.plot(head[0], head[1], 'o', color=color, markersize=8, zorder=5)
        ax.plot(closest[0], closest[1], 'x', color=color, markersize=6, zorder=5)

    # Own snake body — draw head-to-tail line; last 2 elements are tail x,y
    tail = x[-2:] * AVOID_DIST
    ax.plot([0, tail[0]], [0, tail[1]], '-', color='#2ecc71', linewidth=2.5, alpha=0.8)
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
        total    = str(len(episodes)) + ('' if done[0] else '+')
        _, start = episodes[state['ei']]
        return f"Episode {state['ei'] + 1}/{total}  line {start}"

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)

    def update(_):
        ei, fi = state['ei'], state['fi']
        frames, _ = episodes[ei]
        frame = frames[fi]
        if args.flat:
            draw_frame_flat(ax, get_flat(frame))
        else:
            draw_frame(ax, frame)
        ax.set_title(f"{ep_label()}  frame {fi + 1}/{len(frames)}  line {episodes[ei][1] + fi}",
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
        elif event.key == 'right':
            frames, _ = episodes[state['ei']]
            state['fi'] = (state['fi'] + 50) % len(frames)
        elif event.key == 'left':
            frames, _ = episodes[state['ei']]
            state['fi'] = (state['fi'] - 50) % len(frames)
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
