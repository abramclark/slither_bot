import pickle

from flax import nnx
import gymnasium
import jax
from jax import numpy as np
import optax

MAX = 1000
EP_SHAPE = ((MAX, 24), (MAX, 4), (MAX,), (MAX, 8))


class PolicyCritic(nnx.Module):
    def __init__(self, input=24, trans=48, hidden=3, output=4, rngs=nnx.Rngs(0)):
        self.policy = mlp(input, trans, hidden, output * 2, rngs)
        self.value  = mlp(input, trans, hidden, 1,          rngs)

    def __call__(self, x): return infer(self.policy, x)
    def critic(self, x): return infer(self.value, x)

    def act(self, x, stochastic, rk=jax.random.key(0)):
        out = self(x)
        d = out.shape[-1] // 2
        mu, log_sigma = out[:d], np.clip(out[d:], -4, 2)
        sigma = np.exp(log_sigma)
        noise = jax.random.normal(rk, mu.shape) * sigma if stochastic else 0.0
        return jax.nn.tanh(mu + noise), out

def infer(mlp, x):
    for l in mlp[:-1]:
        x = jax.nn.tanh(l(x))
    return mlp[-1](x)

def mlp(input, trans, hidden, output, rngs):
    wi = uniform(-0.45, 0.45)
    return nnx.List(
        [nnx.Linear(input, trans, kernel_init=wi, rngs=rngs)] +
        [nnx.Linear(trans, trans, kernel_init=wi, rngs=rngs) for _ in range(hidden)] +
        [nnx.Linear(trans, output, kernel_init=wi, rngs=rngs)]
    )


def loss(model, episode, gamma=.98, critic=True):
    states, actions, rewards, nograd_logits, n = episode
    out = jax.vmap(model)(states)
    d = out.shape[-1] // 2
    mu, log_sigma = out[:, :d], np.clip(out[:, d:], -4, 2)
    sigma = np.exp(log_sigma)
    mask = np.arange(MAX) < n

    returns = discount(rewards, gamma)
    ret_mean = returns.sum() / n
    ret_std = np.sqrt(((returns - ret_mean * mask) ** 2).sum() / n)
    returns = ((returns - ret_mean) * mask) / ret_std

    raw = np.arctanh(np.clip(actions, -1 + 1e-6, 1 - 1e-6))
    action_log_probs = (
        -0.5 * ((raw - mu) / sigma) ** 2 - log_sigma
        - np.log(1 - actions ** 2 + 1e-6)
    ).sum(axis=1) * mask

    entropies = (log_sigma + 0.5 * np.log(2 * np.pi * np.e)).sum(axis=1) * mask
    entropy_mean = entropies.sum() / n

    val = returns * action_log_probs
    if critic:
        criticisms = jax.vmap(model.critic)(states).squeeze() * mask
        critic_mean = criticisms.sum() / n
        critic_std = np.sqrt(((returns - ret_mean * mask) ** 2).sum() / n)
        criticisms = ((criticisms - critic_mean) * mask) / critic_std
        val = val - criticisms
    val = -sum(val)

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

def critic_loss(model, episode, gamma=.98, critic=1):
    states, actions, rewards, nograd_logits, n = episode
    mask = np.arange(MAX) < n
    vals = jax.vmap(model.critic)(states).squeeze() * mask

    returns = discount(rewards, gamma)
    ret_mean = returns.sum() / n
    ret_std = np.sqrt(((returns - ret_mean * mask) ** 2).sum() / n)
    returns = ((returns - ret_mean) * mask) / ret_std

    return ((returns - vals) ** 2).mean()

def critic_batch_loss(model, batch, **kws):
    vals = jax.vmap(lambda *ep: critic_loss(model, ep, **kws))(*batch)
    return vals.mean()

@nnx.jit
def critic_learn(model, optimizer, batch, **loss_kws):
    val, grad = nnx.value_and_grad(lambda m, b: critic_batch_loss(m, b, **loss_kws))(model, batch)
    optimizer.update(model, grad)
    return val

@nnx.jit(static_argnames=('critic', 'gamma'))
def learn(model, optimizer, batch, **loss_kws):
    grad, aux = nnx.grad(batch_loss, has_aux=True, graph=False)(model, batch, **loss_kws)
    optimizer.update(model, grad)
    return aux

def mc_train(model, episodes=100, lr=3e-3, critic_lr=3e-2, batch_size=5, rk=jax.random.key(0), critic=True, policy=True, **loss_kws):
    optimizer = nnx.Optimizer(model, optax.adabelief(learning_rate=lr), wrt=nnx.Param)
    critic_optimizer = nnx.Optimizer(model, optax.adabelief(learning_rate=critic_lr), wrt=nnx.Param)
    env = gymnasium.make('BipedalWalker-v3', render_mode=None)

    print(f'ep: loss entropy critic_loss reward_sum')
    for i in range(episodes):
        rk, sk = jax.random.split(rk)
        batch = get_batch(env, model, batch_size, sk, stochastic=True)

        critic_loss = critic_learn(model, critic_optimizer, batch, **loss_kws) if critic else 0
        loss_val, reward_sum, entropy_mean = learn(model, optimizer, batch, critic=critic, **loss_kws) if policy else (0, 0, 0)

        print(f'{i:4d}: {loss_val:6.1f} {entropy_mean:6.3f} {critic_loss:6.3f} {reward_sum:6.1f}')

        if critic:
            while critic_loss > 0.5:
                rk, sk = jax.random.split(rk)
                batch = get_batch(env, model, batch_size, sk, stochastic=True)
                critic_loss = critic_learn(model, critic_optimizer, batch, **loss_kws)
                reward_sum = batch[2].sum(axis=1).mean()
                print(f'{i:4d}: critic {critic_loss:6.3f} {reward_sum:6.1f}')
                i += 1


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

        state, reward, terminal, truncated, _ = env.step(action)
        rewards = rewards.at[n].set(reward)

        n += 1
        if terminal or truncated: break

    return states, actions, rewards, logits, n

def get_batch(env, model, batch_size, rk, stochastic=True):
    episodes = []
    for _ in range(batch_size):
        rk, sk = jax.random.split(rk)
        episodes.append(get_episode(env, model, stochastic=stochastic, rk=sk))
    return jax.tree.map(lambda *xs: np.stack(xs), *episodes)


def score(model, n=20):
    env = gymnasium.make('BipedalWalker-v3', render_mode=None)

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
