import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment import LoadableModel


class LanderNet(LoadableModel):
    save_path = 'lander.pt'

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
        return action, dist.log_prob(action)


def get_episode(env, model):
    rewards, log_probs = [], []
    obs, _ = env.reset()

    while True:
        action, log_prob = model.act(obs)
        obs, reward, terminal, truncated, *_ = env.step(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)
        if terminal or truncated: break
    log_probs = torch.stack(log_probs)

    return rewards, log_probs


def discount(rewards, gamma=.98):
    T = len(rewards)
    returns = torch.zeros(T)
    total = 0
    for i in reversed(range(T)):
        total = rewards[i] + gamma * total
        returns[i] = total
    return returns


def train(model, episodes=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = gym.make('LunarLander-v3', render_mode=None)

    for i in range(episodes):
        optimizer.zero_grad()
        rewards, log_probs = get_episode(env, model)

        returns = discount(rewards)
        returns = (returns - returns.mean()) / returns.std()
        loss = -(returns * log_probs).sum()
        loss.backward()
        optimizer.step()
        print(f'{i:4d}: {loss.item():6.1f} {sum(rewards):6.1f}')


def td_train(model, gamma=.98, episodes=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env = gym.make('LunarLander-v3', render_mode=None)

    for i in range(episodes):
        state, _ = env.reset()

        steps = []
        with torch.no_grad():
            while True:
                logits = model(torch.from_numpy(state))
                epsilon = max(0.05, 1.0 - i / episodes)
                if torch.rand(1).item() < epsilon:
                    action = torch.tensor(env.action_space.sample())
                else:
                    action = logits.argmax()

                state2, reward, terminal, truncated, *_ = env.step(action.item())
                steps.append((state, action, logits.max(), reward))
                state = state2
                if terminal or truncated: break

        losses = []
        last_i = len(steps) - 1
        optimizer.zero_grad()
        for j, (state, action, action_val, reward) in enumerate(steps):
            next_val = 0 if j == last_i else steps[j + 1][2]
            val = model(torch.from_numpy(state))[action]
            losses.append((reward + gamma * next_val - val) ** 2)

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        print(f'{i:4d}: {loss:6.2f}')


def eval(model, n=20):
    scores = torch.zeros(n)
    env = gym.make('LunarLander-v3', render_mode=None)

    for i in range(n):
        obs, _ = env.reset()
        rewards = []
        while True:
            action = model.act(obs)[0].item()
            obs, reward, terminal, trunc, *__ = env.step(action)
            rewards.append(reward)
            if terminal or trunc: break
        total = sum(rewards)
        print(total)
        scores[i] = total

    print(f'avg={scores.mean():5.1f} max={scores.max():5.1f} min={scores.min():5.1f}')
    return scores.mean()
