import numpy as np
import torch
import torch.nn as nn

from model import AVOID_ANGLE_INDEX, AVOID_DIST_INDEX, BOOST_INDEX, DIR_INDEX, HEADING_INDEX, IN_DIM

# Core features: the 5 most predictive for imminent death
CORE_INDICES = [DIR_INDEX, HEADING_INDEX, BOOST_INDEX, AVOID_ANGLE_INDEX, AVOID_DIST_INDEX]
CORE_DIM     = len(CORE_INDICES)  # 5

SURVIVAL_IN_DIM = IN_DIM

# Fine-tuning stream: everything except the 5 core features
_core_set    = set(CORE_INDICES)
FINE_INDICES = [i for i in range(SURVIVAL_IN_DIM) if i not in _core_set]
FINE_DIM     = len(FINE_INDICES)  # 227

SURVIVAL_SAVE_PATH = "survival.pt"


class SurvivalNet(nn.Module):
    """
    Predicts log(frames_until_death + 1).

    Core stream (5 features): dir, heading, boost, avoid_angle, avoid_dist.
    Makes the primary prediction — these features carry the strongest signal
    for imminent death.

    Fine stream (remaining 227 features): residual correction from full context.
    Zero-initialized so training starts from core predictions only and refines
    from there.
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
        # Zero-init fine output so the network starts from core predictions alone
        nn.init.zeros_(self.fine[-1].weight)
        nn.init.zeros_(self.fine[-1].bias)
        self.out_act = nn.Softplus()

    def forward(self, x):
        core_feat = x[:, CORE_INDICES]
        fine_feat = x[:, FINE_INDICES]
        avoid_dist = x[:, AVOID_DIST_INDEX:AVOID_DIST_INDEX+1]
        return self.out_act(self.core(core_feat) + self.fine(fine_feat) + avoid_dist).squeeze(-1)

    def act(self, x_fixed: np.ndarray, n_headings: int = 16, n_boost: int = 2):
        """
        Find the heading and boost that maximize predicted survival given the fixed
        context features (snake positions, food, own body, etc.).

        Evaluates n_headings uniformly-spaced headings × n_boost boost values in a
        single batched forward pass and returns the argmax. Grid search is more robust
        than gradient ascent here because avoid_dist (the main shortcut signal) is
        fixed context and contributes no heading gradient, so ascent tends to collapse
        to a spurious local optimum.

        Returns: (heading: float in [-1,1], boost: int 0/1, predicted_log_survival: float)
        """
        device = next(self.parameters()).device

        headings = torch.linspace(-1.0, 1.0, n_headings, device=device)
        boosts   = torch.arange(n_boost, dtype=torch.float32, device=device)  # [0, 1]

        # Build (n_headings * n_boost, SURVIVAL_IN_DIM) batch
        x_base = torch.from_numpy(x_fixed).float().to(device)
        batch  = x_base.unsqueeze(0).expand(n_headings * n_boost, -1).clone()

        hh = headings.repeat_interleave(n_boost)  # [h0,h0, h1,h1, ...]
        bb = boosts.repeat(n_headings)             # [0,1, 0,1, ...]
        batch[:, HEADING_INDEX] = hh
        batch[:, BOOST_INDEX]   = bb

        with torch.no_grad():
            vals = self(batch)  # (n_headings * n_boost,)

        best_idx  = vals.argmax().item()
        best_val  = vals[best_idx].item()
        best_heading = hh[best_idx].item()
        best_boost   = int(bb[best_idx].item())

        return best_heading, best_boost, best_val
