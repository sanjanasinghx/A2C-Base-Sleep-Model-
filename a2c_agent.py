"""
a2c_agent.py
============
Advantage Actor-Critic (A2C) for the 2-action SleepEnv.

A single shared MLP trunk feeds two heads:
    actor  -> logits over {WAKE, SLEEP}      (policy pi(a|s))
    critic -> scalar state value V(s)

The agent IS the basal-ganglia controller: actor <-> striatal action
selection, critic's TD error <-> dopaminergic teaching signal. No GEN
pathway logic and no hand-coded thresholds.

It also exposes  mu(s) = pi(SLEEP | s)  -- the scalar "sleep drive" intended
to drive the downstream Wilson-Cowan thalamocortical oscillator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sleep_env import SleepEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 2, n_actions: int = 2, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        logits, value = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    @torch.no_grad()
    def mu(self, obs: np.ndarray) -> float:
        """Sleep drive mu = pi(SLEEP | s). The bridge to the oscillator."""
        logits, _ = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        return float(torch.softmax(logits, dim=-1)[SleepEnv.SLEEP].item())


def compute_gae(rewards, values, last_value, gamma=0.99, lam=0.95):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae, next_value = 0.0, last_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def train(episodes=1200, gamma=0.99, lam=0.95, lr=2e-3, value_coef=0.5,
          entropy_coef=0.01, entropy_coef_final=0.001, seed=0, log_every=100):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SleepEnv()
    net = ActorCritic(obs_dim=env.observation_space.shape[0], n_actions=2)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    history_return, running = [], None
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_b, act_b, val_b, rew_b = [], [], [], []
        done = False
        while not done:
            a, _, v = net.act(obs)
            nxt, r, term, trunc, _ = env.step(a)
            obs_b.append(obs); act_b.append(a); val_b.append(v); rew_b.append(r)
            obs = nxt; done = term or trunc

        with torch.no_grad():
            _, last_v = net.forward(torch.as_tensor(obs, dtype=torch.float32))
        last_value = 0.0 if term else float(last_v.item())
        adv, ret = compute_gae(rew_b, val_b, last_value, gamma, lam)

        obs_t = torch.as_tensor(np.array(obs_b), dtype=torch.float32)
        act_t = torch.as_tensor(act_b, dtype=torch.long)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        logits, values = net.forward(obs_t)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_t)
        entropy = dist.entropy().mean()
        ent_coef = entropy_coef + (entropy_coef_final - entropy_coef) * (ep / episodes)

        loss = (-(logp * adv_t).mean()
                + value_coef * F.mse_loss(values, ret_t)
                - ent_coef * entropy)

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()

        ep_ret = float(np.sum(rew_b)); history_return.append(ep_ret)
        running = ep_ret if running is None else 0.98 * running + 0.02 * ep_ret
        if (ep + 1) % log_every == 0:
            print(f"ep {ep+1:4d} | return {ep_ret:7.2f} | smoothed {running:7.2f} "
                  f"| entropy {entropy.item():.3f} | critic {F.mse_loss(values, ret_t).item():.3f}")
    return net, history_return


@torch.no_grad()
def rollout_greedy(net, seed=123):
    """Greedy episode; returns S, E, action, reward, and mu = pi(SLEEP|s)."""
    env = SleepEnv()
    obs, _ = env.reset(seed=seed)
    S_h, E_h, a_h, r_h, mu_h = [], [], [], [], []
    done = False
    while not done:
        logits, _ = net.forward(torch.as_tensor(obs, dtype=torch.float32))
        a = int(torch.argmax(logits).item())
        mu = float(torch.softmax(logits, dim=-1)[SleepEnv.SLEEP].item())
        S_h.append(obs[0]); E_h.append(obs[1]); mu_h.append(mu)
        obs, r, term, trunc, _ = env.step(a)
        a_h.append(a); r_h.append(r); done = term or trunc
    return (np.array(S_h), np.array(E_h), np.array(a_h),
            np.array(r_h), np.array(mu_h))


if __name__ == "__main__":
    net, _ = train()
    S, E, A, R, MU = rollout_greedy(net)
    frac = np.bincount(A, minlength=2) / len(A)
    print(f"\ngreedy return {R.sum():.2f} | WAKE {frac[0]:.2f} SLEEP {frac[1]:.2f}"
          f" | mu range [{MU.min():.2f}, {MU.max():.2f}]")
