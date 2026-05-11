#!/usr/bin/env python3
"""
Train SACNet (actor) + twin SACCritic networks using Soft Actor-Critic on offline experience.
"""
import argparse
import copy
import json
import math
import os
import sys
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from model import get_flat, K_ANGLE_BINS
from sac_model import SACNet, SACCritic

def angle_to_bin(angle):
    angle = angle % (2 * math.pi)
    return int((angle / (2 * math.pi)) * K_ANGLE_BINS + 0.5) % K_ANGLE_BINS


def load_transitions(experience_path, start, count, death_penalty):
    """Load (s, a_idx, r, s_next, done) for 16 direction bins x 2 boost actions."""
    states, action_idxs, rewards, next_states, dones = [], [], [], [], []
    episode = []

    def flush_episode():
        flat_dim = len(get_flat(episode[0][0]))
        for i, (state, action, improv) in enumerate(episode):
            is_terminal = i == len(episode) - 1
            x = get_flat(state)
            taken = improv if improv else action
            a_idx = angle_to_bin(float(taken[0])) * 2 + int(taken[1] > 0)

            if is_terminal:
                r = death_penalty
                x_next = np.zeros(flat_dim, dtype=np.float32)
            else:
                next_state = episode[i + 1][0]
                r = next_state[0][0][2] - state[0][0][2]
                x_next = get_flat(next_state)

            states.append(x)
            action_idxs.append(a_idx)
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
          f"r mean={r.mean():.3f} std={r.std():.3f} max={r.max():.3f} "
          f"terminal_frac={np.array(dones).mean():.3f}")

    return (np.array(states, dtype=np.float32),
            np.array(action_idxs, dtype=np.int64),
            r,
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32))


def polyak(src, tgt, tau):
    with torch.no_grad():
        for p, pt in zip(src.parameters(), tgt.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)


def train_pretrain(args):
    actor = SACNet()
    resume_ep = actor.load()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

    states, action_idxs, *_ = load_transitions(
        args.experience, args.start, args.count, args.death_penalty
    )
    sx = torch.from_numpy(states)
    sa = torch.from_numpy(action_idxs)

    n = len(states)
    batch_size = min(args.batch_size, n)
    other_prob = (1 - args.pretrain_weight) / (K_ANGLE_BINS * 2 - 1)
    print(f"n={n}  batch={batch_size}  pretrain_weight={args.pretrain_weight:.3f}")
    print(f"{'Epoch':>5}  {'loss':>8}  {'match':>8}  {'entropy':>8}")

    for epoch in range(args.epochs):
        actor.train()
        idx = torch.randperm(n)
        loss_sum = match_sum = entropy_sum = 0.0
        batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
            logits = actor(sx[mb])
            logp = F.log_softmax(logits, dim=1)
            probs = logp.exp()

            target = torch.full_like(probs, other_prob)
            target[torch.arange(len(mb)), sa[mb]] = args.pretrain_weight

            loss = -(target * logp).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            optimizer.step()

            loss_sum += loss.item()
            match_sum += (logits.argmax(dim=1) == sa[mb]).float().mean().item()
            entropy_sum += (-(probs * logp).sum(dim=1).mean().item())
            batches += 1

        print(f"{resume_ep + epoch + 1:>5}  {loss_sum / batches:>8.4f}"
              f"  {match_sum / batches:>8.4f}"
              f"  {entropy_sum / batches:>8.4f}", flush=True)

    torch.save({"model": actor.state_dict(), "ep": resume_ep + args.epochs}, actor.save_path)
    print(f"Saved {actor.save_path}")


def train(args):
    actor = SACNet()
    resume_ep = actor.load()

    q1 = SACCritic()
    q2 = SACCritic()
    if os.path.exists("sac_critics.pt"):
        ckpt = torch.load("sac_critics.pt", weights_only=True)
        q1.load_state_dict(ckpt['q1'])
        q2.load_state_dict(ckpt['q2'])
        print("SACCritic: resumed from sac_critics.pt", file=sys.stderr)
    q1_tgt = copy.deepcopy(q1)
    q2_tgt = copy.deepcopy(q2)
    q1_tgt.eval(); q2_tgt.eval()
    alpha = args.alpha

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_q1 = torch.optim.Adam(q1.parameters(), lr=args.lr)
    opt_q2 = torch.optim.Adam(q2.parameters(), lr=args.lr)

    states, action_idxs, rewards, next_states, dones = load_transitions(
        args.experience, args.start, args.count, args.death_penalty
    )
    sx  = torch.from_numpy(states)
    sa  = torch.from_numpy(action_idxs)
    sr  = torch.from_numpy(rewards)
    sxn = torch.from_numpy(next_states)
    sd  = torch.from_numpy(dones)

    n = len(states)
    batch_size = min(args.batch_size, n)
    print(f"n={n}  batch={batch_size}")
    print(f"{'Epoch':>5}  {'q_loss':>8}  {'pi_loss':>8}  {'bc_loss':>8}  {'alpha':>8}  {'mean_q':>8}  {'entropy':>8}")

    for epoch in range(args.epochs):
        actor.train(); q1.train(); q2.train()
        idx = torch.randperm(n)
        q_loss_sum = pi_loss_sum = bc_loss_sum = 0.0
        entropy_sum = 0.0
        batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            mb = idx[start:start + batch_size]
            sx_mb = sx[mb]; sa_mb = sa[mb]; sr_mb = sr[mb]
            sxn_mb = sxn[mb]; sd_mb = sd[mb]

            # critic update
            with torch.no_grad():
                next_logits = actor(sxn_mb)
                next_logp = F.log_softmax(next_logits, dim=1)
                next_probs = next_logp.exp()
                q1_next = q1_tgt(sxn_mb)
                q2_next = q2_tgt(sxn_mb)
                q_next = torch.min(q1_next, q2_next)
                q_tgt_val = sr_mb + args.gamma * (1 - sd_mb) * (
                    next_probs * (q_next - alpha * next_logp)
                ).sum(dim=1)
            q1_taken = q1(sx_mb).gather(1, sa_mb.unsqueeze(1)).squeeze(1)
            q2_taken = q2(sx_mb).gather(1, sa_mb.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(q1_taken, q_tgt_val) + F.mse_loss(q2_taken, q_tgt_val)
            opt_q1.zero_grad(); opt_q2.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(q1.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(q2.parameters(), args.grad_clip)
            opt_q1.step(); opt_q2.step()

            # actor update
            logits = actor(sx_mb)
            logp = F.log_softmax(logits, dim=1)
            probs = logp.exp()
            with torch.no_grad():
                q_pi = torch.min(q1(sx_mb), q2(sx_mb))
            sac_loss = (probs * (alpha * logp - q_pi)).sum(dim=1).mean()
            bc_loss = F.cross_entropy(logits, sa_mb)
            pi_loss = sac_loss + args.bc_coef * bc_loss
            opt_actor.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            opt_actor.step()

            polyak(q1, q1_tgt, args.tau)
            polyak(q2, q2_tgt, args.tau)

            q_loss_sum += q_loss.item()
            pi_loss_sum += pi_loss.item()
            bc_loss_sum += bc_loss.item()
            entropy_sum += (-(probs * logp).sum(dim=1).mean().item())
            batches += 1

        actor.eval()
        with torch.no_grad():
            probe = sx[:2000]
            probe_logits = actor(probe)
            probe_logp = F.log_softmax(probe_logits, dim=1)
            probe_probs = probe_logp.exp()
            probe_q = torch.min(q1(probe), q2(probe))
            mean_q = (probe_probs * probe_q).sum(dim=1).mean()

        print(f"{resume_ep + epoch + 1:>5}  {q_loss_sum / batches:>8.4f}"
              f"  {pi_loss_sum / batches:>8.4f}"
              f"  {bc_loss_sum / batches:>8.4f}"
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
    p.add_argument("--bc-coef",         type=float, default=0)
    p.add_argument("--pretrain",        action="store_true")
    p.add_argument("--pretrain-weight", type=float, default=0.66)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    (train_pretrain if args.pretrain else train)(args)
