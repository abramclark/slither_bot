import pickle

from flax import nnx
import gymnasium
import jax
from jax import numpy as np
import optax

MAX = 1000
EP_SHAPE = ((MAX, 8), (MAX,), (MAX,), (MAX, 4))


class LanderNet(nnx.Module):
    def __init__(self, rngs=nnx.Rngs(0)):
        wi = uniform(-0.45, 0.45)
        self.layers = nnx.List([
            nnx.Linear(8, 16, kernel_init=wi, rngs=rngs),
            nnx.Linear(16, 16, kernel_init=wi, rngs=rngs),
            nnx.Linear(16, 16, kernel_init=wi, rngs=rngs),
            nnx.Linear(16, 16, kernel_init=wi, rngs=rngs),
            nnx.Linear(16, 4, kernel_init=wi, rngs=rngs),
        ])

    def __call__(self, x: jax.Array) -> jax.Array:
        for l in self.layers[:-1]:
            x = jax.nn.tanh(l(x))
        return self.layers[-1](x)

    def act(self, x, stochastic, rk=jax.random.key(0)):
        logits = self(x)
        action = jax.random.categorical(rk, logits) if stochastic else logits.argmax()
        return action, logits


def loss(model, episode, gamma=.98):
    states, actions, rewards, nograd_logits, n = episode
    logits = jax.vmap(model)(states)
    mask = np.arange(MAX) < n

    returns = discount(rewards, gamma)
    ret_mean = returns.sum() / n
    ret_std = np.sqrt(((returns - ret_mean * mask) ** 2).sum() / n)
    returns = ((returns - ret_mean) * mask) / ret_std

    probs = nnx.softmax(logits)
    log_probs = np.log(probs)
    action_log_probs = log_probs[np.arange(MAX), actions.astype(np.int16)]

    entropies = -(probs * log_probs).sum(axis=1) * mask
    entropy_mean = entropies.sum() / n

    val = -sum(returns * action_log_probs)

    return val, (val, rewards.sum(), entropy_mean)

def discount(rewards, gamma):
    return jax.lax.scan(
        lambda total, r: (v := r + total * gamma, v),
        0.0, rewards, reverse=True
    )[1]

def batch_loss(model, batch, **kws):
    vals, rewards, entropies = jax.vmap(lambda *ep: loss(model, ep, **kws)[1])(*batch)
    val = vals.mean()
    return val, (val, rewards.mean(), entropies.mean())

@nnx.jit
def learn(model, optimizer, batch, **loss_kws):
    grad, aux = nnx.grad(batch_loss, has_aux=True, graph=False)(model, batch, **loss_kws)
    optimizer.update(model, grad)
    return aux


def mc_train(model, rk=jax.random.key(0), episodes=100, lr=3e-3, batch_size=5, **loss_kws):
    optimizer = nnx.Optimizer(model, optax.adabelief(learning_rate=lr), wrt=nnx.Param)
    env = gymnasium.make('LunarLander-v3', render_mode=None)

    for i in range(episodes):
        episodes = []
        for _ in range(batch_size):
            rk, sk = jax.random.split(rk)
            episodes.append(get_episode(env, model, stochastic=True, rk=sk))
        batch = jax.tree.map(lambda *xs: np.stack(xs), *episodes)
        loss_val, reward_sum, entropy_mean = learn(model, optimizer, batch, **loss_kws)
        print(f'{i:4d}: {loss_val:6.1f} {entropy_mean:6.3f} {reward_sum:6.1f}')


def get_episode(env, model, stochastic=True, rk=jax.random.key(0)):
    states, actions, rewards, logits = (np.zeros(s) for s in EP_SHAPE)
    state, _ = env.reset()

    n = 0
    while True:
        rk, sk = jax.random.split(rk)
        action, y = model.act(state, stochastic, sk)

        states = states.at[n].set(state)
        actions = actions.at[n].set(action)
        logits = logits.at[n].set(y)

        state, reward, terminal, truncated, _ = env.step(action.item())
        rewards = rewards.at[n].set(reward)

        n += 1
        if terminal or truncated: break

    return states, actions, rewards, logits, n


def score(model, n=20):
    env = gymnasium.make('LunarLander-v3', render_mode=None)

    def run_ep():
        _, _, rewards, _, n = get_episode(env, model, False)
        score = rewards.sum()
        print(score, n)
        return score

    scores = np.array([run_ep() for i in range(n)])
    return scores.mean()


def uniform(low, high):
  return lambda key, shape, dtype: jax.random.uniform(key, shape, dtype, low, high)


def entropy(a):
    probs = nnx.softmax(a)
    return -(probs * np.log(probs)).sum()


def save(model, path='lander.pickle'):
    state = nnx.to_pure_dict(nnx.split(model)[1])
    pickle.dump(state, open(path, 'wb'))

def load(model, path='lander.pickle'):
    gdef, state = nnx.split(model)
    nnx.replace_by_pure_dict(state, pickle.load(open(path, 'rb')))
    return nnx.merge(gdef, state)
