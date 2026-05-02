import math
import collections
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_ANGLE_BINS = 8  # discrete direction output bins
K_BIG_FOOD = 10 # nearest big foods
K_SM_FOOD = 5   # nearest small foods
K_FOOD = K_BIG_FOOD + K_SM_FOOD
K_SNAKE = 4     # nearest snakes to use
K_SEGMENTS = 3 # examine K_SEGMENTS of snakes
META_SIZE = 6   # [turn_x, turn_y, heading_x * speed, heading_y * speed, speed, scale]
FOOD_SIZE = K_FOOD * 2

IN_DIM = META_SIZE + FOOD_SIZE + K_SNAKE * K_SEGMENTS * 2 + K_SNAKE * META_SIZE + 4

AVOID_DIST = 250
DIRX_INDEX       = 0 # own_meta[(0,1)] = turn to x, y
HEADINGX_INDEX   = 2 # own_meta[(2,3)] = moving towards x, y
BOOST_INDEX      = 4 # own_meta[4] = is speed (.413 to 1)
FOOD_START_INDEX = META_SIZE
FOOD_END_INDEX = FOOD_START_INDEX + FOOD_SIZE
AVOIDX_INDEX = FOOD_END_INDEX # nearest segment angle of nearest snake
OWN_SEGMENTS_INDEX = IN_DIM - K_SEGMENTS * 2

ACTION_INDICES  = [DIRX_INDEX, DIRX_INDEX + 1, BOOST_INDEX]
FOOD_SM_INDEX = FOOD_START_INDEX + K_BIG_FOOD * 2
CORE_INDICES    = [
    AVOIDX_INDEX, AVOIDX_INDEX + 1,
    OWN_SEGMENTS_INDEX, OWN_SEGMENTS_INDEX + 1 # first own segment is angle and distance to world edge
] + list(range(FOOD_START_INDEX, FOOD_END_INDEX))

FINE_INDICES    = [i - 2 for i in range(2, IN_DIM) if i not in CORE_INDICES]
MID_LAYERS = [6, 9]


def parse_record(d):
    me, others, food, timestamp = d
    # slither structure [meta, segment_coords]; [[turn_to, heading, size, scale, speed, base_speed, id], [[x, y]]]
    own_size = me[0][2]
    food = np.array(food, dtype=np.float32)          # (N, 3): [size, x, y]
    own_segs = np.array(me[1], dtype=np.float32)
    snakes_meta = [flat_meta(s[0]) for s in others]
    snakes_segs = [np.array(s[1], dtype=np.float32) for s in others]
    return own_size, flat_meta(self_data), own_segs, snakes_meta, snakes_segs, food


def flat_meta(d):
    """Convert [turn_to, heading, size, scale, speed, base_speed, id] to
    [turn_x, turn_y, heading_x * speed, heading_y * speed, boosting, scale]"""
    v = np.empty(META_SIZE, dtype=np.float32)
    v[:] = [np.cos(d[0]), np.sin(d[0]), np.cos(d[1]) * d[2], np.sin(d[1]) * d[2], d[4] > d[5], d[3]]
    return v


def make_flat_input(food, own_meta, own_segs, snakes_meta, snakes_segs):
    food_flat = np.zeros(FOOD_SIZE, dtype=np.float32)
    if len(food) > 0:
        coords = food[:, 1:]
        dists = np.linalg.norm(coords, axis=1)
        by_dist = dists.argsort()
        bigs = coords[by_dist][food[by_dist][:, 0] >= .9][:K_BIG_FOOD]
        smalls = coords[by_dist][food[by_dist][:, 0] < .9][:K_SM_FOOD]
        food_flat[:len(bigs) * 2] = bigs.flatten()
        food_flat[K_BIG_FOOD * 2:(K_BIG_FOOD + len(smalls)) * 2] = smalls.flatten()

    snakes_dists = [np.linalg.norm(s, axis=1) for s in snakes_segs]
    nearest_seg_ixs = [dists.argmin() for dists in snakes_dists]
    smallest_dists = np.array([dists[i] for i, dists in zip(nearest_seg_ixs, snakes_dists)])
    nearest_ixs = smallest_dists.argsort()[:K_SNAKE]
    nearest_snakes = [np.concat([
        snakes_segs[i][nearest_seg_ixs[i]],
        snakes_segs[i][0],
        snakes_segs[i][-1],
        #select_evenly_spaced(snakes_segs[i], K_SEGMENTS)
    ]) for i in nearest_ixs]
    _snake_flat = np.concat(nearest_snakes).flatten() if snakes_segs else []
    snake_flat = np.zeros(K_SNAKE * K_SEGMENTS * 2, dtype=np.float32)
    snake_flat[:len(_snake_flat)] = _snake_flat

    _metas = np.concat([snakes_meta[i] for i in nearest_ixs]) if snakes_segs else []
    nearest_metas = np.zeros(K_SNAKE * META_SIZE, dtype=np.float32)
    nearest_metas[:len(_metas)] = _metas

    #_own_segs = select_evenly_spaced(own_segs, K_SEGMENTS).flatten()
    #own_segs = np.zeros(K_SEGMENTS * 2, dtype=np.float32)
    #own_segs[:len(_own_segs)] = _own_segs
    own_segs = np.concat([own_segs[0], own_segs[-1]])

    food_flat  /= AVOID_DIST
    snake_flat /= AVOID_DIST
    own_segs   /= AVOID_DIST

    return np.concat([own_meta, food_flat, snake_flat, nearest_metas, own_segs]).astype(np.float32)


def select_evenly_spaced(arr, k):
    if k <= 0:
        return np.array([])
    if k == 1:
        return arr[:1]
    if k >= len(arr):
        return arr.copy()

    indices = np.round(np.linspace(0, len(arr) - 1, k)).astype(int)
    return arr[indices]


def get_flat(d):
    if not d: return d
    own_size, own_meta, own_segs, snakes_meta, snakes_segs, food = parse_record(d)
    return make_flat_input(food, own_meta, own_segs, snakes_meta, snakes_segs)


def bot_script(state):
    if not state: # dead
        return [-1, 0]

    me, snakes, food, timestamp = state
    own_props = me[0]
    snake_scale = own_props[3]
    target_angle = heading = own_props[1]

    min_dist = float("inf")
    avoid_angle = None
    for _, body in snakes:
        for x, y in body:
            dist = (x*x + y*y)**0.5
            if dist < min_dist:
                min_dist = dist
                avoid_angle = math.atan2(y, x)

    boost = 0
    if min_dist < AVOID_DIST and avoid_angle is not None:
        target_angle = angle_sub(avoid_angle, math.pi)
        if abs(angle_sub(heading, target_angle)) < 1.5: # within ~pi/2 radians
            boost = 1

    else:
        # Seek closest food, preferring large one
        best_dist = None
        best_sm_dist = float('inf')
        for f in food:
            size, x, y = f[0], f[1], f[2]
            dist = (x*x + y*y)**0.5
            if dist < (best_dist if best_dist else best_sm_dist) and dist > 50 * snake_scale:
                if size >= .9:
                    best_dist = dist
                else:
                    best_sm_dist = dist
                target_angle = math.atan2(y, x)

    is_avoiding = min_dist < AVOID_DIST and avoid_angle is not None
    return [target_angle, boost, is_avoiding]


def angle_sub(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class ImprovisingAgent():
    def __init__(self, actor, improv_length=5, improv_prob=1e-3):
        self.actor = actor
        self.improv_length = improv_length
        self.improv_prob = improv_prob

        self.improv_i = 0

    def act(self, x):
        action = self.actor(x)
        if self.should_improvise(x, action):
            self.improv_i = self.improv_length
        if self.improv_i:
            self.improv_i -= 1
            return action, self.improvise(x, action)
        else:
            return action, None

    def should_improvise(self, x, action):
        return np.random.rand() < self.improv_prob

    def improvise(self, x): raise NotImplemented


class ImprovisingScript(ImprovisingAgent):
    def __init__(self, improv_length=10, **kws):
        super().__init__(bot_script, improv_length=improv_length, **kws)
        self.mode = 0 # 0: start improvising, 1: opposite policy, 2: random choice of left / stright / right
        self.turn = 0
        self.boost = 0

    def improvise(self, x, action):
        angle, boost = action[0], action[1]
        if not self.mode:
            self.mode = np.random.randint(1, 3)
            if self.mode == 2:
                self.turn = np.random.choice([-.8, -.4, 0, .4, .8]).item()
                self.boost = np.random.randint(0, 2)

        if self.mode == 1:
            return angle + math.pi, not boost, 'opposite'
        elif self.mode == 2:
            heading = x[0][0][1]
            return heading + self.turn, self.boost, 'random'


class Model(nn.Module):
    save_path = 'model.pt'

    def load(self):
        try:
            ckpt = torch.load(self.save_path, weights_only=True)
            self.load_state_dict(ckpt['model'])
            ep = ckpt.get('ep', 0)
            print(f"{self.__class__.__name__}: resumed from {self.save_path} (ep={ep})")
        except (FileNotFoundError, EOFError, RuntimeError):
            print(f"{self.__class__.__name__}: no checkpoint at {self.save_path}, starting fresh")

        self.eval()
        return self

#    def __init__(self, dropout=0):
#        super().__init__()
#        input = IN_DIM - 2
#        embed     = 16
#        self.head = nn.Sequential(
#            nn.Linear(input, input * 2), nn.Tanh(),
#            nn.Dropout(p=.3 * dropout),
#            nn.Linear(input * 2, embed), nn.Tanh(),
#            nn.Dropout(p=.1 * dropout),
#            nn.Linear(embed, embed), nn.Tanh(),
#            nn.Dropout(p=.1 * dropout),
#            nn.Linear(embed, embed), nn.Tanh(),
#            nn.Linear(embed, K_ANGLE_BINS + 1),
#        )
#
#        # Zero-init fine columns so training warm-starts from core features only
#        self.head[0].weight.data[:, FINE_INDICES] = 0
#        # Identity-init MID_LAYERS so they start as pass-throughs to prevent over-fitting
#        for i in MID_LAYERS:
#            weights = self.head[i].weight
#            nn.init.eye_(weights)
#            weights.data += torch.randn_like(weights) * .01
#            nn.init.zeros_(self.head[i].bias)
#
#    def forward(self, x):
#        return self.head(x[..., 2:])
#
#    def act(self, x: np.ndarray):
#        t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
#        with torch.no_grad():
#            pred  = self.forward(t)
#        bins  = torch.linspace(0, 2 * np.pi, K_ANGLE_BINS + 1)[:-1]
#        probs = torch.softmax(pred[..., :K_ANGLE_BINS], dim=-1)
#        dir_x = (probs * bins.cos()).sum(dim=-1)
#        dir_y = (probs * bins.sin()).sum(dim=-1)
#        return float(np.atan2(dir_y.item(), dir_x.item())), pred[..., -1].clamp(-1, 1).item() > 0, pred
