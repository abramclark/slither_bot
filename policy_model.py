import numpy as np
import torch
import torch.nn as nn

from environment import IN_DIM, FINE_INDICES, get_flat, LoadableModel

K_ANGLE_BINS = 16
MID_LAYERS = [6, 9]


class PolicyNet(LoadableModel):
    save_path = 'policy.pt'
    offsets = torch.linspace(0, 2 * np.pi, K_ANGLE_BINS + 1)[:-1]

    def __init__(self, dropout=0):
        super().__init__()
        input = IN_DIM - 2
        embed = 16
        self.head = nn.Sequential(
            nn.Linear(input, input * 2), nn.Tanh(),
            nn.Dropout(p=.3 * dropout),
            nn.Linear(input * 2, embed), nn.Tanh(),
            nn.Dropout(p=.1 * dropout),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Dropout(p=.1 * dropout),
            nn.Linear(embed, embed), nn.Tanh(),
            nn.Linear(embed, K_ANGLE_BINS + 1),
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
        t = torch.from_numpy(get_flat(state))
        with torch.no_grad():
            pred = self.forward(t)
        probs = torch.softmax(pred[..., :K_ANGLE_BINS], dim=-1)
        bin_idx = torch.multinomial(probs, 1).item()
        angle = float(self.offsets[bin_idx].item())
        boost = int(torch.bernoulli(torch.sigmoid(pred[-1])).item())

        self._act_count += 1
        if self._act_count % 20 == 0:
            print(f'[model] bin={bin_idx} angle={angle:.2f} boost={boost} p={probs[bin_idx]:.2f}')

        return [angle, boost, pred.tolist()]
