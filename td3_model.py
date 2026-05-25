import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import get_flat, IN_DIM, LoadableModel

EPS = 1e-6


class TD3Net(LoadableModel):
    save_path = 'td3.pt'

    def __init__(self):
        super().__init__()
        input_dim = IN_DIM - 2
        embed = 16
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), nn.Tanh(),
            nn.Linear(input_dim * 2, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, 3),
        )
        self._act_count = 0

    def forward(self, x):
        out = self.head(x[..., 2:])
        squashed = torch.tanh(out)
        direction = F.normalize(squashed[..., :2], dim=-1, eps=EPS)
        boolean = squashed[..., 2].add(1).mul(0.5)
        return direction, boolean

    def act(self, state):
        x = torch.from_numpy(get_flat(state))
        with torch.no_grad():
            direction, boost = self.forward(x)
        angle = float(torch.atan2(direction[1], direction[0]).item())
        boost_val = float(boost.item() >= 0.5)

        self._act_count += 1
        if self._act_count % 20 == 0:
            print(f'[td3] angle={angle:.2f} boost={boost_val:.2f}')

        return [angle, boost_val]


class TD3Critic(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = IN_DIM - 2
        action_dim = 3  # dx, dy, boost
        inp = state_dim + action_dim
        embed = 16
        self.head = nn.Sequential(
            nn.Linear(inp, inp * 2), nn.Tanh(),
            nn.Linear(inp * 2, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, 1),
        )

    def forward(self, x, a):
        return self.head(torch.cat([x[..., 2:], a], dim=-1)).squeeze(-1)
