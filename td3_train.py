#!/usr/bin/env python3
"""
Train TD3Net (actor) + twin TD3Critic networks on offline experience.
"""
import argparse
import copy
import json
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from environment import get_flat
from td3_model import TD3Net, TD3Critic


def load_transitions(experience_path, start, count, death_penalty):
    """Load (s, a, r, s_next, done) with continuous actions [dx, dy, boost]."""
    states, actions, rewards, next_states, dones = [], [], [], [], []
    episode = []

    def flush_episode():
        flat_dim = len(get_flat(episode[0][0]))
        for i, (state, action, improv) in enumerate(episode):
            is_terminal = i == len(episode) - 1
            x = get_flat(state)
            taken = improv if improv else action
            angle = float(taken[0])
            boost = float(taken[1] > 0)
            a = [np.cos(angle), np.sin(angle), boost]

            if is_terminal:
                r = death_penalty
                x_next = np.zeros(flat_dim, dtype=np.float32)
            else:
                next_state = episode[i + 1][0]
                r = next_state[0][0][2] - state[0][0][2]
                x_next = get_flat(next_state)

            states.append(x)
            actions.append(a)
            rewards.append(r)
            next_states.append(x_next)
            dones.append(float(is_terminal))

    with open(experience_path) as f:
        lines = islice(f, start, None if count is None else start + count)
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            if d == []:
                if len(episode) > 1:
                    flush_episode()
                episode.clear()
            else:
                state, action, improv, *_ = d
                episode.append((state, action, improv))

    if not states:
        raise SystemExit("No transitions found.")

    r = np.array(rewards, dtype=np.float32)
    print(f"Loaded {len(states)} transitions  "
          f"r mean={r.mean():.3f} std={r.std():.3f}  "
          f"terminal_frac={np.array(dones).mean():.3f}")

    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            r,
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32))


def polyak(src, tgt, tau):
    with torch.no_grad():
        for p, pt in zip(src.parameters(), tgt.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)


def train(args):
    actor = TD3Net()
    resume_ep = actor.load()

    q1 = TD3Critic()
    q2 = TD3Critic()
    q1_tgt = copy.deepcopy(q1)
    q2_tgt = copy.deepcopy(q2)
    q1_tgt.eval(); q2_tgt.eval()
    alpha = args.alpha

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_q1 = torch.optim.Adam(q1.parameters(), lr=args.lr)
    opt_q2 = torch.optim.Adam(q2.parameters(), lr=args.lr)

    states, actions, rewards, next_states, dones = load_transitions(
        args.experience, args.start, args.count, args.death_penalty
    )
    sx  = torch.from_numpy(states)
    sa  = torch.from_numpy(actions)
    sr  = torch.from_numpy(rewards)
    sxn = torch.from_numpy(next_states)
    sd  = torch.from_numpy(dones)

    n = len(states)
    batch_size = min(args.batch_size, n)
    print(f"n={n}  batch={batch_size}")
    print(f"{'Epoch':>5}  {'q_loss':>8}  {'pi_loss':>8}  {'alpha':>8}  {'mean_q':>8}  {'entropy':>8}")

    for epoch in range(args.epochs):
        actor.train(); q1.train(); q2.train()
        idx = torch.randperm(n)
        q_loss_sum = pi_loss_sum = 0.0
        entropy_sum = 0.0
        batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
            sx_mb = sx[mb]; sa_mb = sa[mb]; sr_mb = sr[mb]
            sxn_mb = sxn[mb]; sd_mb = sd[mb]

            # critic update
            with torch.no_grad():
                dir_next, boost_next, lp_next = actor(sxn_mb)
                a_next = torch.cat([dir_next, boost_next.unsqueeze(-1)], dim=-1)
                q_tgt_val = sr_mb + args.gamma * (1 - sd_mb) * (
                    torch.min(q1_tgt(sxn_mb, a_next), q2_tgt(sxn_mb, a_next)) - alpha * lp_next
                )

            q_loss = F.mse_loss(q1(sx_mb, sa_mb), q_tgt_val) + F.mse_loss(q2(sx_mb, sa_mb), q_tgt_val)
            opt_q1.zero_grad(); opt_q2.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(q1.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(q2.parameters(), args.grad_clip)
            opt_q1.step(); opt_q2.step()

            # actor update
            dir_new, boost_new, lp_new = actor(sx_mb)
            a_new = torch.cat([dir_new, boost_new.unsqueeze(-1)], dim=-1)
            q_pi = torch.min(q1(sx_mb, a_new), q2(sx_mb, a_new))
            pi_loss = (alpha * lp_new - q_pi).mean()
            opt_actor.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            opt_actor.step()

            polyak(q1, q1_tgt, args.tau)
            polyak(q2, q2_tgt, args.tau)

            q_loss_sum += q_loss.item()
            pi_loss_sum += pi_loss.item()
            entropy_sum += (-lp_new).mean().item()
            batches += 1

        actor.eval()
        with torch.no_grad():
            probe = sx[:2000]
            dir_e, boost_e, _ = actor(probe)
            a_e = torch.cat([dir_e, boost_e.unsqueeze(-1)], dim=-1)
            mean_q = torch.min(q1(probe, a_e), q2(probe, a_e)).mean()

        print(f"{resume_ep + epoch + 1:>5}  {q_loss_sum / batches:>8.4f}"
              f"  {pi_loss_sum / batches:>8.4f}"
              f"  {alpha:>8.4f}"
              f"  {mean_q:>8.3f}"
              f"  {entropy_sum / batches:>8.3f}", flush=True)

    torch.save({"model": actor.state_dict(), "ep": resume_ep + args.epochs}, actor.save_path)
    torch.save({"q1": q1.state_dict(), "q2": q2.state_dict()}, "sac_critics.pt")
    print(f"Saved {actor.save_path}  sac_critics.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experience",      default="experience.jsonl")
    p.add_argument("--start",           type=int,   default=0)
    p.add_argument("--count",           type=int,   default=None)
    p.add_argument("--death-penalty",   type=float, default=-1.0)
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--tau",             type=float, default=0.005)
    p.add_argument("--alpha",           type=float, default=0.2)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
