"""
sleep_env.py
============
A clean Gymnasium environment for the basal-ganglia sleep-wake model.

Design (per the simple/clean formulation):
  * State is the 2-D vector  s = (S, E):
        S = homeostatic sleep pressure   ("tiredness meter")
        E = energy reserves              ("battery life")
    The exogenous circadian phase C is omitted.
  * Two macro-actions only -- no hand-built GEN pathway logic:
        0 = WAKE   (stay awake / exploit the environment)
        1 = SLEEP  (recover)
    An Actor-Critic agent (see a2c_agent.py) learns the policy directly;
    there are no hard-coded thresholds or macro-action locks in the env.
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

        # ---- Model constants (from the original framework) ----
        self.k2 = 0.15                  # tiredness / clearance factor
        self.intrinsic_gamma = 0.90     # intrinsic discount from the original
                                        #   framework, kept as a model constant.
                                        #   It does NOT enter the per-step sleep
                                        #   reward (r_sleep = k2*S); temporal
                                        #   credit uses the RL discount in
                                        #   a2c_agent.py.
        self.R_env = 1.0                # environmental payoff (world richness)
        self.bankruptcy_penalty = -3.0  # penalty when energy is exhausted
        self.exploration_bonus = 0.05   # tiny constant bonus for being awake

        # Per-action dynamics.
        self.wake_E_drain = 0.04
        self.sleep_E_gain = 0.06

        # --- Coupled (S, E) dynamics (replaces the old INDEPENDENT S rules) ---
        # Theory: S (adenosine pressure) and E are "two readings of one
        # process" -- adenosine accrues in proportion to energy spent and
        # clears in proportion to energy recovered. So S is driven by the
        # SAME quantities that move E, not by its own decoupled rule:
        #   WAKE:  S += adenosine_gain  * wake_E_drain
        #   SLEEP: S -= adenosine_clear * sleep_E_gain
        #
        # Make gain == clear == c (= 0.375). Then the per-unit-energy coupling is
        # IDENTICAL going up and coming down, so dS = -c * dE on every step. That
        # integrates to an exact affine relation between S and E:
        #       S = 0.1 + 0.375 * (1 - E)
        # i.e. S is a co-readout of E -- bounded in [0.1, 0.4375] over E in [1, 0],
        # with no drift and no reliance on clipping. (This is what makes S and E
        # "two readings of one process": the proportionality constant is the same
        # in both directions.) NOTE the per-STEP magnitudes still differ because
        # the energy steps differ: WAKE dS = 0.375*0.04 = 0.015, SLEEP dS =
        # 0.375*0.06 = 0.0225 -- but the per-unit-energy rate (0.375) is shared,
        # which is the invariant that matters.
        self.adenosine_gain = 0.375     # S gained per unit energy drained
        self.adenosine_clear = 0.375    # S cleared per unit energy recovered (== gain)

        # Old DECOUPLED S constants -- retained for reference, no longer used
        # (replaced by the coupled dynamics above):
        # self.wake_S_rise = 0.10       # was: S += k2*(1-S)*wake_S_rise
        # self.sleep_S_clear = 0.15     # was: S -= k2*S*sleep_S_clear

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
            # Coupled dynamics: spending energy accrues adenosine pressure.
            self.E -= self.wake_E_drain
            self.S += self.adenosine_gain * self.wake_E_drain
            reward = (self.bankruptcy_penalty if self.E <= 0
                      else self.R_env + self.exploration_bonus)
        else:  # SLEEP
            # Coupled dynamics: recovering energy clears adenosine pressure.
            self.E += self.sleep_E_gain
            self.S -= self.adenosine_clear * self.sleep_E_gain
            reward = self.k2 * self.S

        self.S = float(np.clip(self.S, 0.0, 1.0))
        self.E = float(np.clip(self.E, 0.0, 1.0))

        # Change 1: bankruptcy is TERMINAL. When energy is exhausted the episode
        # ends; because E only decreases while awake, the bankruptcy_penalty
        # computed above fires exactly ONCE, on this terminating WAKE step.
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
