import collections
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_FOOD = 10    # nearest food items to use
K_SNAKE = 3    # nearest snakes to use
K_SEGMENTS = 1 # examine K_SEGMENTS * 2 around nearest
K_OWN_SEGS = 7 # examine K_OWN_SEGS segments of self


def parse_record(d):
    own_size = float(d[0])
    food = np.array(d[1], dtype=np.float32)          # (N, 3): [size, angle, dist]
    self_data = d[2]
    target_dir = float(self_data[0])
    target_boost = float(self_data[2])
    own_meta = np.array(self_data[0:4], dtype=np.float32)   # [speed, ?, ?]
    own_body = np.array(self_data[6:], dtype=np.float32).reshape(-1, 2)
    snakes_meta = [np.array(s[:4], dtype=np.float32) for s in d[3]]
    snakes_body = [np.array(s[4:], dtype=np.float32).reshape(-1, 2) for s in d[3]]
    return own_size, food, own_meta, own_body, snakes_meta, snakes_body, target_dir, target_boost


def make_flat_input(food, own_meta, own_body, snakes_meta, snakes_body):
    # Mirror bot_script exactly: top-K by size, filter too-close food, sort by dist/size.
    # This guarantees the target angle is always at food_flat[1] (or heading if no valid food).
    food_flat = np.zeros(K_FOOD * 3, dtype=np.float32)
    if len(food) > 0:
        snake_scale = own_meta[3]
        big_food = food[food[:, 0].argsort()][-K_FOOD:]
        valid = big_food[big_food[:, 2] > 50 * snake_scale]
        if len(valid) > 0:
            valid = valid[(valid[:, 2] / valid[:, 0]).argsort()]
            food_flat[:len(valid) * 3] = valid.flatten()

    smallest_dists = np.array([s[:, 1].min() for s in snakes_body])
    nearest_ixs = smallest_dists.argsort()[:K_SNAKE]
    nearest_snakes = [snakes_body[i] for i in nearest_ixs]
    nearest_segments = [s[:, 1].argmin() for s in nearest_snakes]
    segments = [i + np.arange(-K_SEGMENTS, K_SEGMENTS + 1) for i in nearest_segments]
    snake_parts = [np.pad(sb, ((1, 1), (0, 0)))[segs] for sb, segs in zip(nearest_snakes, segments)]
    _snake_flat = np.concat(snake_parts).flatten() if snake_parts else np.array([], dtype=np.float32)
    snake_flat = np.zeros(K_SNAKE * (2 * K_SEGMENTS + 1) * 2, dtype=np.float32)
    snake_flat[:len(_snake_flat)] = _snake_flat
    _metas = np.concatenate([snakes_meta[i] for i in nearest_ixs]) if len(nearest_ixs) else np.array([], dtype=np.float32)
    nearest_metas = np.zeros(K_SNAKE * 4, dtype=np.float32)
    nearest_metas[:len(_metas)] = _metas

    own_skip = max(1, len(own_body) // K_OWN_SEGS)
    _segs = own_body[own_skip::own_skip][:K_OWN_SEGS].flatten()
    own_segs = np.zeros(K_OWN_SEGS * 2, dtype=np.float32)
    own_segs[:len(_segs)] = _segs

    return np.concatenate([[own_meta[1]], food_flat, snake_flat, nearest_metas, own_segs]).astype(np.float32)


IN_DIM = 1 + K_FOOD * 3 + K_SNAKE * (2 * K_SEGMENTS + 1) * 2 + K_SNAKE * 4 + K_OWN_SEGS * 2

GAMMA = 0.99
LAM = 0.95
LR = 1e-3
CLIP = 0.2
PPO_EPOCHS = 4
SL_EPOCHS = 4
BATCH_SIZE = 64
MINI_BATCH = 16
REPLAY_SIZE = 2000

SAVE_PATH = "model.pt"
AVOID_DIST = 250
HEADING_INDEX = 0
FOOD_START_INDEX = 1
FOOD_END_INDEX = FOOD_START_INDEX + K_FOOD * 3
AVOID_ANGLE_INDEX = 35
AVOID_DIST_INDEX = 36
IS_AVOIDING_INDEX = IN_DIM

AVOID_FOCUS_IN = 3               # heading + avoid_angle + avoid_dist
FOOD_FOCUS_IN  = 1 + K_FOOD * 3  # heading + K_FOOD food triplets


def avoid_focus_features(x):
    return x[..., [HEADING_INDEX, AVOID_ANGLE_INDEX, AVOID_DIST_INDEX]]


def food_focus_features(x):
    return torch.cat([x[..., [HEADING_INDEX]], x[..., FOOD_START_INDEX:FOOD_END_INDEX]], dim=-1)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(IN_DIM + 1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        # Two focused supervised heads — one per behavior mode.
        # avoid_focus trains only on is_avoiding=True frames (heading + snake geometry).
        # food_focus trains only on is_avoiding=False frames (heading + food features).
        # torch.where in dir_focus_logits routes gradients to the right head automatically.
        self.avoid_focus = nn.Sequential(
            nn.Linear(AVOID_FOCUS_IN, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.food_focus = nn.Sequential(
            nn.Linear(FOOD_FOCUS_IN, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.boost_focus = nn.Sequential(
            nn.Linear(AVOID_FOCUS_IN, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 2),
        )
        self.dir_residual = nn.Linear(128, 1)
        self.dir_log_std = nn.Parameter(torch.full((1,), -2.0))  # std ≈ 0.14 initial exploration
        self.boost_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)
        nn.init.zeros_(self.dir_residual.weight)
        nn.init.zeros_(self.dir_residual.bias)
        nn.init.zeros_(self.boost_head.weight)
        nn.init.zeros_(self.boost_head.bias)

    def forward(self, x):
        h = self.shared(x)
        dir_logits = self.dir_logits(x, h)
        dir_mean = torch.tanh(dir_logits).squeeze(-1)
        return dir_mean, self.boost_head(h) + self.boost_focus_logits(x), self.value_head(h).squeeze(-1)

    def boost_focus_logits(self, x):
        return self.boost_focus(avoid_focus_features(x))

    def dir_focus_logits(self, x):
        is_avoiding = x[..., IS_AVOIDING_INDEX:IS_AVOIDING_INDEX + 1] > 0.5
        food_feat = food_focus_features(x)
        # Shortcut: add best-food's angle directly so the model starts predicting
        # the right target on day 0; food_focus learns the residual correction.
        # food_feat[..., 2] = food_flat[1] = angle of best-ratio food item.
        food_logit = self.food_focus(food_feat) + food_feat[..., 2:3]
        return torch.where(is_avoiding,
                           self.avoid_focus(avoid_focus_features(x)),
                           food_logit)

    def dir_logits(self, x, h=None):
        if h is None:
            h = self.shared(x)
        return self.dir_focus_logits(x) + self.dir_residual(h)

    def supervised_dir(self, x):
        return torch.tanh(self.dir_focus_logits(x)).squeeze(-1)

    def focus_parameters(self):
        return (list(self.avoid_focus.parameters()) +
                list(self.food_focus.parameters()) +
                list(self.boost_focus.parameters()))

    def act(self, x):
        dir_mean, boost_logits, value = self(x)
        dir_std = self.dir_log_std.clamp(-4, 2).exp()
        dir_dist = torch.distributions.Normal(dir_mean, dir_std)
        boost_dist = torch.distributions.Categorical(logits=boost_logits)
        dir_action = dir_dist.sample().clamp(-1, 1)
        boost_idx = boost_dist.sample()
        log_prob = dir_dist.log_prob(dir_action).sum() + boost_dist.log_prob(boost_idx)
        return dir_action.item(), boost_idx.item(), log_prob.item(), value.item()

    def evaluate(self, x, dir_actions, boost_idx):
        dir_mean, boost_logits, value = self(x)
        dir_std = self.dir_log_std.clamp(-4, 2).exp()
        dir_dist = torch.distributions.Normal(dir_mean, dir_std)
        boost_dist = torch.distributions.Categorical(logits=boost_logits)
        log_prob = dir_dist.log_prob(dir_actions).sum(-1) + boost_dist.log_prob(boost_idx)
        entropy = dir_dist.entropy().sum(-1) + boost_dist.entropy()
        return log_prob, entropy, value


class PPOTrainer:
    def __init__(self, model: ActorCritic, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.lock = threading.Lock()

        self.states = []
        self.dir_actions = []
        self.boost_idxs = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        self.ep = 0
        self.total_steps = 0

    def push(self, state, dir_action, boost_idx, log_prob, value, reward, done):
        with self.lock:
            self.states.append(state)
            self.dir_actions.append(dir_action)
            self.boost_idxs.append(boost_idx)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(float(done))
            self.total_steps += 1
            if done:
                self.ep += 1

            if len(self.states) >= BATCH_SIZE:
                self._update()

    def _gae(self, rewards, values, dones, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAM * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def _update(self):
        states = np.array(self.states, dtype=np.float32)
        dir_actions = np.array(self.dir_actions, dtype=np.float32)
        boost_idxs = np.array(self.boost_idxs, dtype=np.int64)
        old_lps = np.array(self.log_probs, dtype=np.float32)
        rewards = self.rewards[:]
        values = self.values[:]
        dones = self.dones[:]

        with torch.no_grad():
            _, _, last_val = self.model(torch.from_numpy(states[-1:]).to(self.device))
        advantages, returns = self._gae(rewards, values, dones, last_val.item())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        s = torch.from_numpy(states).to(self.device)
        da = torch.from_numpy(dir_actions).to(self.device)
        bi = torch.from_numpy(boost_idxs).to(self.device)
        olp = torch.from_numpy(old_lps).to(self.device)
        adv = torch.from_numpy(advantages).to(self.device)
        ret = torch.from_numpy(returns).to(self.device)

        n = len(states)
        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(n)
            for start in range(0, n, MINI_BATCH):
                mb = idx[start:start + MINI_BATCH]
                lp, ent, val = self.model.evaluate(s[mb], da[mb], bi[mb])
                ratio = (lp - olp[mb]).exp()
                a = adv[mb]
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a).mean()
                value_loss = F.mse_loss(val, ret[mb])
                loss = policy_loss + 0.5 * value_loss - 0.01 * ent.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        ep_reward = sum(rewards)
        print(f"[PPO] ep={self.ep} steps={self.total_steps} reward={ep_reward:.2f}")
        torch.save({"model": self.model.state_dict(), "ep": self.ep, "total_steps": self.total_steps}, SAVE_PATH)

        self.states.clear()
        self.dir_actions.clear()
        self.boost_idxs.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


class SupervisedTrainer:
    def __init__(self, model: ActorCritic, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.focus_parameters(), lr=LR)
        self.lock = threading.Lock()
        self.ep = 0
        self.total_steps = 0
        self._ep_steps = 0
        self._replay_states = collections.deque(maxlen=REPLAY_SIZE)
        self._replay_dirs = collections.deque(maxlen=REPLAY_SIZE)
        self._replay_boosts = collections.deque(maxlen=REPLAY_SIZE)
        self._since_update = 0

    def step(self, state, target_dir, target_boost):
        with self.lock:
            self._replay_states.append(state)
            self._replay_dirs.append(float(target_dir))
            self._replay_boosts.append(int(target_boost))
            self.total_steps += 1
            self._ep_steps += 1
            self._since_update += 1
            if self._since_update >= BATCH_SIZE and len(self._replay_states) >= BATCH_SIZE:
                self._update()
                self._since_update = 0

    def _update(self):
        buf_size = len(self._replay_states)
        idxs = np.random.choice(buf_size, BATCH_SIZE, replace=False)
        states = torch.from_numpy(np.array([self._replay_states[i] for i in idxs], dtype=np.float32)).to(self.device)
        tds = torch.tensor([self._replay_dirs[i] for i in idxs], dtype=torch.float32).to(self.device)
        tbs = torch.tensor([self._replay_boosts[i] for i in idxs], dtype=torch.long).to(self.device)
        n_avoid = int((states[:, IS_AVOIDING_INDEX] > 0.5).sum().item())
        n_food = BATCH_SIZE - n_avoid
        total_loss = 0.0
        num_batches = 0
        for _ in range(SL_EPOCHS):
            for mb in torch.randperm(BATCH_SIZE).split(MINI_BATCH):
                dir_mean = self.model.supervised_dir(states[mb])
                angle_err = (dir_mean - tds[mb]) % 2
                angle_err = torch.where(angle_err > 1, angle_err - 2, angle_err)
                loss = (angle_err ** 2).mean()
                avoid_mb = states[mb, IS_AVOIDING_INDEX] > 0.5
                if avoid_mb.any():
                    boost_logits = self.model.boost_focus_logits(states[mb][avoid_mb])
                    loss = loss + F.cross_entropy(boost_logits, tbs[mb][avoid_mb])
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.focus_parameters(), 2.0)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        avg = total_loss / num_batches
        print(f"[SL] steps={self.total_steps} buf={buf_size} avg_loss={avg:.4f} avoid={n_avoid} food={n_food}")
        torch.save({"model": self.model.state_dict(), "ep": self.ep, "total_steps": self.total_steps}, SAVE_PATH)

    def reset_optimizer(self):
        with self.lock:
            self.optimizer = torch.optim.Adam(self.model.focus_parameters(), lr=LR)
            print("[SL] optimizer reset")

    def finish_episode(self):
        with self.lock:
            self.ep += 1
            print(f"[SL] ep={self.ep} steps={self.total_steps} ep_steps={self._ep_steps}")
            torch.save({"model": self.model.state_dict(), "ep": self.ep, "total_steps": self.total_steps}, SAVE_PATH)
            self._ep_steps = 0


def angle_sub(from_ang, to):
    d = (from_ang - to) % 2
    if d > 1:
        d -= 2
    return d


def bot_script(state_d):
    food_dat = state_d[1]   # [[sz, angle_pi, dist], ...]
    own_props = state_d[2]
    snake_scale = own_props[3]
    snakes = state_d[3]
    target_angle = heading = own_props[1]

    min_dist = float("inf")
    avoid_angle = None
    for snake in snakes:
        body = snake[4:]
        for angle, dist in zip(body[::2], body[1::2]):
            if dist < min_dist:
                min_dist = dist
                avoid_angle = angle

    boost = 0
    if min_dist < AVOID_DIST and avoid_angle is not None:
        target_angle = angle_sub(avoid_angle, 1)
        if abs(angle_sub(heading, target_angle)) < .3:
            boost = 1
            #print('BOOST', min_dist, target_angle, heading, avoid_angle)
        #else:
        #    print('AVOID', min_dist, target_angle, heading, avoid_angle)

    else:
        # Seek food with best size-per-distance value, restricted to the same
        # top-K_FOOD-by-size candidates that get_flat encodes (so the target
        # angle is always present in the feature vector).
        candidates = sorted(food_dat, key=lambda f: f[0], reverse=True)[:K_FOOD]
        best_val = float('inf')
        for f in candidates:
            sz, angle, dist = f[0], f[1], f[2]
            val = dist / sz
            if val < best_val and dist > 50 * snake_scale:
                best_val = val
                target_angle = angle
        #print('FOOD', best_val, angle, heading)

    is_avoiding = min_dist < AVOID_DIST and avoid_angle is not None
    return [target_angle, boost, is_avoiding]


def get_flat(d):
    own_size, food, own_meta, own_body, snakes_meta, snakes_body, _, _ = parse_record(d)
    return make_flat_input(food, own_meta, own_body, snakes_meta, snakes_body)
