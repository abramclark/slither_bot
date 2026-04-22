import numpy as np
import torch
import torch.nn as nn

from model import AVOID_ANGLE_INDEX, AVOID_DIST_INDEX, BOOST_INDEX, DIR_INDEX, HEADING_INDEX, IN_DIM

CORE_INDICES = [DIR_INDEX, HEADING_INDEX, BOOST_INDEX, AVOID_ANGLE_INDEX, AVOID_DIST_INDEX]
CORE_DIM     = len(CORE_INDICES)

_core_set    = set(CORE_INDICES)
FINE_INDICES = [i for i in range(IN_DIM) if i not in _core_set]
FINE_DIM     = len(FINE_INDICES)

VALUE_SAVE_PATH = "value.pt"


class ValueNet(nn.Module):
    """
    Predicts V(t) = net_size_change - 0.053 * horizon - death_indicator.

    Output is unbounded real-valued: positive means above-average growth with
    low death risk; negative means below-average growth or imminent death.

    Architecture mirrors SurvivalNet: core stream (5 death-predictive features)
    plus fine residual (full context), with avoid_dist as a warm-start shortcut.
    Fine output is zero-initialized so training starts from core predictions only.
    """
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(CORE_DIM, 32), nn.Tanh(),
            nn.Linear(32, 32),       nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.fine = nn.Sequential(
            nn.Linear(FINE_DIM, 64), nn.Tanh(),
            nn.Linear(64, 32),       nn.Tanh(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.fine[-1].weight)
        nn.init.zeros_(self.fine[-1].bias)

    def forward(self, x):
        core_feat = x[:, CORE_INDICES]
        fine_feat = x[:, FINE_INDICES]
        return (self.core(core_feat) + self.fine(fine_feat)).squeeze(-1)

    def act(self, x_fixed: np.ndarray, n_headings: int = 16, n_boost: int = 2):
        """
        Find the heading and boost that maximize predicted value given fixed context.

        Returns: (heading: float in [-1,1], boost: int 0/1, predicted_value: float)
        """
        device = next(self.parameters()).device

        headings = torch.linspace(-1.0, 1.0, n_headings, device=device)
        boosts   = torch.arange(n_boost, dtype=torch.float32, device=device)

        x_base = torch.from_numpy(x_fixed).float().to(device)
        batch  = x_base.unsqueeze(0).expand(n_headings * n_boost, -1).clone()

        hh = headings.repeat_interleave(n_boost)
        bb = boosts.repeat(n_headings)
        batch[:, HEADING_INDEX] = hh
        batch[:, BOOST_INDEX]   = bb

        with torch.no_grad():
            vals = self(batch)

        best_idx     = vals.argmax().item()
        best_heading = hh[best_idx].item()
        best_boost   = int(bb[best_idx].item())
        best_val     = vals[best_idx].item()

        return best_heading, best_boost, best_val
