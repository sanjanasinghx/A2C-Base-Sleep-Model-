"""
Gymnasium environment for the basal-ganglia sleep-wake model.

State is s = (S, E): S is homeostatic sleep pressure, E is energy reserves.
No circadian term. Two actions: 0 = WAKE, 1 = SLEEP. The policy is learned
by the actor-critic in a2c_agent.py, so there are no thresholds in here.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SleepEnv(gym.Env):
    """Two-variable (S, E) sleep-wake environment with two macro-actions."""

    metadata = {"render_modes": ["ansi"]}

    WAKE = 0
    SLEEP = 1

    def __init__(self, max_steps: int = 200, render_mode: str | None = None):
        super().__init__()

        self.k2 = 0.15                  # tiredness / clearance factor
        self.intrinsic_gamma = 0.90     # not used in the reward (RL discount is in a2c_agent.py)
        self.R_env = 1.0                # environmental payoff (world richness)
        self.bankruptcy_penalty = -3.0  # penalty when energy is exhausted
        self.exploration_bonus = 0.05   # small bonus for being awake

        self.wake_E_drain = 0.04
        self.sleep_E_gain = 0.06

        # S is coupled to E: adenosine builds with energy spent and clears with
        # energy recovered.
        #   WAKE:  S += adenosine_gain  * wake_E_drain
        #   SLEEP: S -= adenosine_clear * sleep_E_gain
        # With gain == clear (0.375), dS = -0.375 * dE every step, so
        # S = 0.1 + 0.375 * (1 - E) exactly, and S stays in [0.1, 0.4375].
        self.adenosine_gain = 0.375     # S gained per unit energy drained
        self.adenosine_clear = 0.375    # S cleared per unit energy recovered

        self.max_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        self.S = 0.1
        self.E = 1.0
        self.t = 0

    def _obs(self) -> np.ndarray:
        return np.array([self.S, self.E], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.S = 0.1
        self.E = 1.0
        self.t = 0
        return self._obs(), {}

    def step(self, action: int):
        self.t += 1
        action = int(action)

        if action == self.WAKE:
            self.E -= self.wake_E_drain
            self.S += self.adenosine_gain * self.wake_E_drain
            reward = (self.bankruptcy_penalty if self.E <= 0
                      else self.R_env + self.exploration_bonus)
        else:  # SLEEP
            self.E += self.sleep_E_gain
            self.S -= self.adenosine_clear * self.sleep_E_gain
            reward = self.k2 * self.S

        self.S = float(np.clip(self.S, 0.0, 1.0))
        self.E = float(np.clip(self.E, 0.0, 1.0))

        # running out of energy ends the episode, so the penalty fires once
        terminated = (self.E <= 0)
        truncated = self.t >= self.max_steps
        info = {"S": self.S, "E": self.E, "action": action}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        return f"t={self.t:3d}  S={self.S:.3f}  E={self.E:.3f}"


try:
    gym.register(id="SleepEnv-v0", entry_point="sleep_env:SleepEnv")
except gym.error.Error:
    pass


if __name__ == "__main__":
    env = SleepEnv()
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(env.max_steps):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    print(f"random-policy return: {total:.2f}  |  final {env.render()}")
