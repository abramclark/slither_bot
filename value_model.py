import numpy as np
import torch
import torch.nn as nn

from model import (AVOID_ANGLE_INDEX, AVOID_DIST_INDEX, BOOST_INDEX, DIR_INDEX,
                   FOOD_START_INDEX, HEADING_INDEX, IN_DIM, FOOD_END_INDEX,
                   K_FOOD, K_SNAKE, K_SEGMENTS, OWN_SEGMENTS_INDEX)

ACTION_INDICES  = [DIR_INDEX, HEADING_INDEX, BOOST_INDEX]
CORE_INDICES    = ACTION_INDICES + [
    AVOID_ANGLE_INDEX, AVOID_DIST_INDEX, FOOD_START_INDEX + 1, FOOD_START_INDEX + 4,
    OWN_SEGMENTS_INDEX, OWN_SEGMENTS_INDEX + 1 # first own segment is angle and distance to world edge
]
_core_set       = set(CORE_INDICES)
FINE_INDICES    = [i for i in range(IN_DIM) if i not in _core_set]

BODY_FREEZE_INDICES = list(range(OWN_SEGMENTS_INDEX + 2, OWN_SEGMENTS_INDEX + K_SEGMENTS * 2))

VALUE_SAVE_PATH = "value.pt"


class ValueNet(nn.Module):
    """
    Predicts V(t) = net_size_change - AVERAGE_VALUE * horizon - death_indicator.

    Context features (all except dir/heading/boost) are processed through 3 layers,
    then action features are concatenated and passed through 2 more layers.
    This lets the network evaluate different headings/boosts against fixed context
    without recomputing context representations.
    """
    def __init__(self):
        super().__init__()
        embed = IN_DIM * 2
        self.head = nn.Sequential(
            nn.Linear(IN_DIM, embed), nn.Tanh(),
            nn.Linear(embed, embed),                nn.Tanh(),
            nn.Linear(embed, embed),                nn.Tanh(),
            nn.Linear(embed, embed),                nn.Tanh(),
            nn.Linear(embed, 1),
        )

        # Zero-init fine columns so training warm-starts from core features only
        self.head[0].weight.data[:, FINE_INDICES] = 0
        # Identity-init layers 2 and 3 so they start as pass-throughs
        nn.init.eye_(self.head[2].weight); nn.init.zeros_(self.head[2].bias)
        nn.init.eye_(self.head[4].weight); nn.init.zeros_(self.head[4].bias)

    def forward(self, x):
        return self.head(x).squeeze(-1)

    def sample(self, x_fixed: np.ndarray):
        """Evaluate 16 heading offsets × 2 boosts = 32 candidates and return all"""
        device = next(self.parameters()).device

        offsets = torch.linspace(-1, 1 - 1/8, 16, device=device)
        current_heading = float(x_fixed[HEADING_INDEX])
        turn_angles = (offsets + current_heading + 1.) % 2. - 1.

        # 22 candidates: each heading/dir pair × boost 0 and 1
        hh = turn_angles.repeat_interleave(2)
        bb = torch.tensor([0., 1.], device=device).repeat(16)

        x_base = torch.from_numpy(x_fixed).float().to(device)
        batch  = x_base.unsqueeze(0).expand(32, -1).clone()
        batch[:, DIR_INDEX]     = hh
        batch[:, HEADING_INDEX] = hh
        batch[:, BOOST_INDEX]   = bb

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
