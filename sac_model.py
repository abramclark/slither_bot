import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment import get_flat, IN_DIM, LoadableModel, FINE_INDICES

K_ANGLE_BINS = 16
MID_LAYERS = [4, 6]


class SACNet(LoadableModel):
    save_path = 'sac.pt'
    offsets = torch.linspace(0, 2 * torch.pi, K_ANGLE_BINS + 1)[:-1]

    def __init__(self):
        super().__init__()
        input_dim = IN_DIM - 2
        embed = 16
        self.deterministic = False
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), nn.Tanh(),
            nn.Linear(input_dim * 2, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, K_ANGLE_BINS * 2),
        )

        # Zero-init fine columns so training warm-starts from core features only
        self.head[0].weight.data[:, FINE_INDICES] = 0
        # Identity-init MID_LAYERS so they start as pass-throughs to prevent over-fitting
        for i in MID_LAYERS:
            weights = self.head[i].weight
            nn.init.eye_(weights)
            weights.data += torch.randn_like(weights) * .01
            nn.init.zeros_(self.head[i].bias)

        self._act_count = 0

    def forward(self, x):
        return self.head(x[..., 2:])

    def act(self, state):
        x = torch.from_numpy(get_flat(state))
        with torch.no_grad():
            logits = self.forward(x)
            dist = Categorical(logits=logits)
            if self.deterministic:
                action_idx = logits.argmax(dim=-1)
                log_prob = None
            else:
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

            bin_idx = action_idx // 2
            boost = int(action_idx % 2)
            angle = float(self.offsets[bin_idx])
        lp = float(log_prob) if log_prob is not None else None

        self._act_count += 1
        if self._act_count % 20 == 0:
            lp_str = f'{lp:.2f}' if lp is not None else 'det'
            print(f'[sac] angle={angle:.2f} boost={boost} log_prob={lp_str}')

        return [angle, boost, lp]


class SACCritic(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = IN_DIM - 2
        embed = 16
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), nn.Tanh(),
            nn.Linear(input_dim * 2, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, K_ANGLE_BINS * 2),
        )

    def forward(self, x):
        return self.head(x[..., 2:])
