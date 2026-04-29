import numpy as np
import torch
import torch.nn as nn

from model import (AVOIDX_INDEX, DIRX_INDEX, BOOST_INDEX, FOOD_START_INDEX, HEADINGX_INDEX, IN_DIM,
                   FOOD_END_INDEX, OWN_SEGMENTS_INDEX)

SAMPLE_COUNT = 16
ACTION_SCALAR   = 10
CORE_SCALAR     = 1
ACTION_INDICES  = [DIRX_INDEX, DIRX_INDEX + 1, HEADINGX_INDEX, HEADINGX_INDEX + 1, BOOST_INDEX]
CORE_INDICES    = ACTION_INDICES + [
    AVOIDX_INDEX, AVOIDX_INDEX + 1, FOOD_START_INDEX + 1, FOOD_START_INDEX + 2,
    OWN_SEGMENTS_INDEX, OWN_SEGMENTS_INDEX + 1 # first own segment is angle and distance to world edge
]
_core_set       = set(CORE_INDICES)
FINE_INDICES    = [i for i in range(IN_DIM) if i not in _core_set]

BODY_FREEZE_INDICES = list(range(OWN_SEGMENTS_INDEX + 2, IN_DIM))
FOOD_FREEZE_INDICES = list(range(FOOD_START_INDEX, FOOD_END_INDEX))

MID_LAYERS = [6, 9]


def to_core(x): return x[..., CORE_INDICES]


class Value2Net(nn.Module):
    """
    Predicts V(t) = net_size_change - AVERAGE_VALUE * horizon - death_indicator.

    Context features (all except dir/heading/boost) are processed through 3 layers,
    then action features are concatenated and passed through 2 more layers.
    This lets the network evaluate different headings/boosts against fixed context
    without recomputing context representations.
    """
    save_path = 'value2.pt'

    def __init__(self):
        super().__init__()
        embed     = 16
        self.head = nn.Sequential(
            nn.Linear(IN_DIM, IN_DIM * 2), nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(IN_DIM * 2, embed), nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, 1),
        )

        # Zero-init fine columns so training warm-starts from core features only
        self.head[0].weight.data[:, FINE_INDICES] = 0
        # Init MID_LAYERS as near pass-through / identity to prevent over-fitting
        for i in MID_LAYERS:
            weights = self.head[i].weight
            nn.init.eye_(weights)
            weights.data += torch.randn_like(weights) * .01
            nn.init.zeros_(self.head[i].bias)

    def forward(self, x):
        x = x.clone()
        x[:, ACTION_INDICES] *= ACTION_SCALAR
        x[:, CORE_INDICES]   *= CORE_SCALAR
        return self.head(x).squeeze(-1)

    def sample(self, x_fixed: np.ndarray):
        """Evaluate 16 heading offsets × 2 boost candidates and return all"""
        current_heading = np.atan2(x_fixed[HEADINGX_INDEX + 1], x_fixed[HEADINGX_INDEX])
        turn_angles = torch.arange(16) * (2 * np.pi / 16) + current_heading

        hh = turn_angles.repeat_interleave(2)
        bb = torch.tensor([.413, 1.]).repeat(16)

        x_base = torch.tensor(x_fixed)
        batch  = x_base.unsqueeze(0).expand(32, -1).clone()
        batch[:, DIRX_INDEX]         = hh.cos()
        batch[:, DIRX_INDEX + 1]     = hh.sin()
        batch[:, HEADINGX_INDEX]     = hh.cos()
        batch[:, HEADINGX_INDEX + 1] = hh.sin()
        batch[:, BOOST_INDEX]        = bb

        with torch.no_grad():
            return self(batch), hh, bb

    def act(self, x_fixed: np.ndarray):
        """Returns: (heading: float in [-1,1], boost: int 0/1, predicted_value: float)"""
        vals, hh, bb = self.sample(x_fixed)

        # Shape: (16 headings, 2 boosts); headings are circular
        vals_2d = vals.view(16, 2)
        heading_scores = vals_2d.max(dim=1).values  # best boost per heading
        smoothed = (heading_scores.roll(1) + heading_scores + heading_scores.roll(-1))

        best_hi  = smoothed.argmax().item()
        best_bi  = vals_2d[best_hi].argmax().item()

        return hh[best_hi * 2].item(), best_bi, vals_2d[best_hi, best_bi].item()
