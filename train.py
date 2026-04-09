#!/usr/bin/env python3
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import IN_DIM, make_flat_input, parse_record

class SlitherDataset(Dataset):
    def __init__(self, path):
        self.inputs = []
        self.targets = []
        raw_dirs = []

        with open(path) as f:
            for line in f:
                if line:
                    d = json.loads(line)
                    if len(d) == 4:
                        own_size, food, own_meta, own_body, snakes_meta, snakes_body, tdir, tboost = parse_record(d)
                        self.inputs.append(make_flat_input(food, own_meta, own_body, snakes_meta, snakes_body))
                        self.targets.append((tdir, tboost))
                        raw_dirs.append(tdir)

        self.inputs = np.stack(self.inputs)

        # upweight frames where dir changed from the previous frame
        self.weights = [
            3.0 if (i == 0 or raw_dirs[i] != raw_dirs[i - 1]) else 1.0
            for i in range(len(raw_dirs))
        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])
        tdir, tboost = self.targets[idx]
        return x, torch.tensor(tdir, dtype=torch.float32), torch.tensor(tboost, dtype=torch.float32)


class SlitherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(IN_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.dir_head = nn.Sequential(nn.Linear(64, 1), nn.Tanh())  # continuous -1 to 1
        self.boost_head = nn.Linear(64, 1)  # logit for boost

    def forward(self, x):
        h = self.shared(x)
        return self.dir_head(h).squeeze(-1), self.boost_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    dataset = SlitherDataset('record.jsonl')
    print(f'Loaded {len(dataset)} records')

    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_weights = [dataset.weights[i] for i in train_set.indices]
    sampler = torch.utils.data.WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=64, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = SlitherNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for x, tdir, tboost in train_loader:
            x, tdir, tboost = x.to(device), tdir.to(device), tboost.to(device)
            dir_out, boost_logit = model(x)
            loss = F.mse_loss(dir_out, tdir) + \
                   F.binary_cross_entropy_with_logits(boost_logit, tboost)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, boost_acc, dir_acc = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, tdir, tboost in val_loader:
                x, tdir, tboost = x.to(device), tdir.to(device), tboost.to(device)
                dir_out, boost_logit = model(x)
                val_loss += (F.mse_loss(dir_out, tdir) +
                             F.binary_cross_entropy_with_logits(boost_logit, tboost)).item()
                dir_pred = dir_out.round().clamp(-1, 1)
                dir_acc += (dir_pred == tdir).float().mean().item()
                boost_acc += ((boost_logit > 0) == tboost.bool()).float().mean().item()

        scheduler.step()
        nt, nv = len(train_loader), len(val_loader)
        avg_dir_acc = dir_acc / nv
        avg_boost_acc = boost_acc / nv
        print(f'Epoch {epoch+1:2d} | train={train_loss/nt:.4f} | val={val_loss/nv:.4f} | dir_acc={avg_dir_acc:.3f} | boost_acc={avg_boost_acc:.3f}')

        torch.save(model.state_dict(), 'slither_model.pt')

        if avg_boost_acc >= 0.99 and avg_dir_acc >= 0.99:
            print(f'Reached dir_acc={avg_dir_acc:.3f} boost_acc={avg_boost_acc:.3f} — stopping early.')
            break

    print('Saved slither_model.pt')


if __name__ == '__main__':
    train()
