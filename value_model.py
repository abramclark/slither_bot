import numpy as np
import torch
import torch.nn as nn

from model import Model, ImprovisingAgent
from model import (AVOIDX_INDEX, DIRX_INDEX, BOOST_INDEX, FOOD_START_INDEX, HEADINGX_INDEX, IN_DIM,
                   FOOD_END_INDEX, OWN_SEGMENTS_INDEX, ACTION_INDICES, CORE_INDICES)

SAMPLE_COUNT = 16
ACTION_SCALAR   = 10
MID_LAYERS = [6, 9]

CORE_INDICES = ACTION_INDICES + CORE_INDICES
FINE_INDICES    = [i for i in range(IN_DIM) if i not in CORE_INDICES]

def to_core(x): return x[..., CORE_INDICES]


class ValueNet(Model):
    """Predicts V(t) = net_size_change - AVERAGE_VALUE * horizon - death_indicator."""
    save_path = "value.pt"

    def __init__(self):
        super().__init__()
        embed     = 16
        self.head = nn.Sequential(
            nn.Linear(IN_DIM, IN_DIM * 2), nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(IN_DIM * 2, embed), nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, 1),
        )

        # Zero-init fine columns so training warm-starts from core features only
        self.head[0].weight.data[:, FINE_INDICES] = 0
        # Identity-init MID_LAYERS so they start as pass-throughs to prevent over-fitting
        for i in MID_LAYERS:
            weights = self.head[i].weight
            nn.init.eye_(weights)
            weights.data += torch.randn_like(weights) * .01
            nn.init.zeros_(self.head[i].bias)

    def forward(self, x):
        x = x.clone()
        x[:, ACTION_INDICES] *= ACTION_SCALAR
        return self.head(x).squeeze(-1)

    def sample(self, x_fixed: np.ndarray):
        """Evaluate SAMPLE_COUNT heading offsets × 2 boost candidates and return all"""
        current_heading = np.atan2(x_fixed[HEADINGX_INDEX + 1], x_fixed[HEADINGX_INDEX])
        turn_angles = torch.arange(SAMPLE_COUNT) * (2 * np.pi / SAMPLE_COUNT) + current_heading

        hh = turn_angles.repeat_interleave(2)
        bb = torch.tensor([.413, 1.]).repeat(SAMPLE_COUNT)

        x_base = torch.tensor(x_fixed)
        batch  = x_base.unsqueeze(0).expand(32, -1).clone()
        batch[:, DIRX_INDEX]         = hh.cos()
        batch[:, DIRX_INDEX + 1]     = hh.sin()
        #batch[:, HEADINGX_INDEX]     = hh.cos()
        #batch[:, HEADINGX_INDEX + 1] = hh.sin()
        batch[:, BOOST_INDEX]        = bb

        with torch.no_grad():
            return self(batch), hh, bb

    def act(self, x_fixed: np.ndarray):
        """Returns: (heading: float in [-1,1], boost: int 0/1, predicted_value: float)"""
        vals, hh, bb = self.sample(x_fixed)

        # Shape: (SAMPLE_COUNT headings, 2 boosts); headings are circular
        vals_2d = vals.view(SAMPLE_COUNT, 2)
        heading_scores = vals_2d.max(dim=1).values  # best boost per heading
        smoothed = (heading_scores.roll(1) + heading_scores + heading_scores.roll(-1))

        best_hi  = smoothed.argmax().item()
        best_bi  = vals_2d[best_hi].argmax().item()

        return hh[best_hi * 2].item(), best_bi, vals_2d[best_hi, best_bi].item()


class ImprovisingValueNet(ImprovisingAgent):
    def improvise(self, x, action):
        raise NotImplemented
