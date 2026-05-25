import time

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

from environment import LoadableModel


class LanderNet(LoadableModel):
    save_path = 'lander.pt'
    log4 = torch.log(torch.tensor(4))

    def __init__(self, dropout=0):
        super().__init__()
        input = 8
        trans = 16
        embed = 6
        self.head = nn.Sequential(
            nn.Linear(input, trans), nn.Tanh(),
            nn.Linear(trans, trans), nn.Tanh(),
            nn.Linear(trans, trans), nn.Tanh(),
            nn.Linear(trans, trans), nn.Tanh(),
            nn.Linear(trans, 4),
        )

    def forward(self, x):
        return self.head(x)

    def act(self, x):
        logits = self(torch.from_numpy(x))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy() - self.log4

    def act_max(self, x):
        return self(torch.from_numpy(x)).argmax().item()


def get_episode(env, model):
    steps = []
    obs, _ = env.reset()

    while True:
        action, log_prob, entropy = model.act(obs)
        obs, reward, terminal, truncated, *_ = env.step(action.item())
        steps.append(torch.stack([torch.tensor(reward), log_prob, entropy]))
        if terminal or truncated: break

    return torch.stack(steps).transpose(0, 1)


def discount(rewards, gamma=.98):
    T = len(rewards)
    returns = torch.zeros(T)
    total = 0
    for i in reversed(range(T)):
        total = rewards[i] + gamma * total
        returns[i] = total
    return returns


def train(model, episodes=100, lr=1e-4, alpha=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env = gym.make('LunarLander-v3', render_mode=None)

    for i in range(episodes):
        optimizer.zero_grad()
        rewards, log_probs, entropies = get_episode(env, model)

        returns = discount(rewards)
        returns = (returns - returns.mean()) / returns.std()
        entropy = entropies.mean()
        loss = -(returns * log_probs).sum() + alpha * entropy
        loss.backward()
        optimizer.step()
        print(f'{i:4d}: {loss.item():6.1f} {entropy.item():6.3f} {sum(rewards):6.1f}')


def td_train(model, gamma=.98, episodes=100, lr=1e-4, alpha=.05, copy_int=1000, epsilon=.5, stochastic=False, n_ahead=5):
    old_model = model.__class__()
    old_model.load_state_dict(model.state_dict())
    old_model.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env = gym.make('LunarLander-v3', render_mode=None)

    count = 0
    for i in range(episodes):
        steps = []
        with torch.no_grad():
            state, _ = env.reset()
            while True:
                logits = old_model(torch.from_numpy(state))
                if stochastic:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                else:
                    #epsilon = max(0.01, .5 - i / (episodes * 2))
                    if torch.rand(1).item() < epsilon:
                        action = torch.tensor(env.action_space.sample())
                    else:
                        action = logits.argmax()

                state2, reward, terminal, truncated, *_ = env.step(action.item())
                steps.append((state, action, logits[action], reward, terminal, truncated))
                state = state2

                count += 1
                if count >= copy_int:
                    old_model.load_state_dict(model.state_dict())
                    count = 0

                if terminal or truncated: break

        td_errors = []
        entropies = []
        rewards = []
        end = len(steps) - 1
        #rixs = torch.randperm(len(steps))
        for j in range(len(steps)):
            state, action, action_val, reward, dead, truncated = steps[j]
            rewards.append(reward)

            optimizer.zero_grad()

            logits = model(torch.from_numpy(state))
            val = logits[action]

            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1) - LanderNet.log4
            entropies.append(entropy)

            ahead = min(n_ahead, end - j)
            if n_ahead == ahead:
                target_val = steps[j + ahead][3] if steps[j + ahead][5] else steps[j + ahead][2]
            else:
                target_val = 0
            for k in reversed(range(ahead)):
                target_val = steps[j + k][3] + target_val * gamma
            td_error = (target_val - val) ** 2
            td_errors.append(td_error)

            loss = td_error - entropy * alpha
            loss.backward()
            optimizer.step()

        td_error = torch.stack(td_errors).mean()
        entropy = torch.stack(entropies).mean()
        print(f'{i:4d}: {loss:6.2f} {td_error:6.2f} {entropy:6.2f} {sum(rewards):6.1f}')


def eval(model, n=20):
    scores = torch.zeros(n)
    lengths = torch.zeros(n)
    env = gym.make('LunarLander-v3', render_mode=None)

    for i in range(n):
        obs, _ = env.reset()
        rewards = []
        while True:
            action = model.act_max(obs)
            obs, reward, terminal, trunc, *__ = env.step(action)
            rewards.append(reward)
            if terminal or trunc: break
        total = sum(rewards)
        print(total, len(rewards))
        scores[i] = total
        lengths[i] = len(rewards)

    print(f'scores: avg={scores.mean():5.1f} max={scores.max():5.1f} min={scores.min():5.1f}')
    print(f'lengths: avg={lengths.mean():5.1f} max={lengths.max():5.1f} min={lengths.min():5.1f}')
    return scores.mean()


def view(model, stochastic=False):
    env = gym.make('LunarLander-v3', render_mode='human')
    obs, _ = env.reset()
    i = 0
    while True:
        logits = model(torch.from_numpy(obs))
        if stochastic:
            dist = Categorical(logits=logits)
            action = dist.sample()
        else:
            action = logits.argmax()

        obs, reward, terminal, trunc, *__ = env.step(action.item())
        print(f'{i:3d} {reward:8.1f}   ', ' '.join(f'{n:5.2f}' for n in logits))
        if terminal or trunc: break
        i += 1

    time.sleep(1)
    env.close()

