"""
run_experiment.py
=================
Train the Actor-Critic on the 2-action SleepEnv and produce figures:

  1. learning_curve.png  -- episode return vs. training episode
  2. learned_rollout.png -- S, E, the sleep-drive mu = pi(SLEEP|s), and the
                            chosen action over one greedy episode
  3. policy_map.png      -- greedy action over the (S, E) plane (the learned
                            wake/sleep boundary)
  4. consolidated_bouts.png (optional) -- the same learned agent with a
                            deterministic macro-lock that commits to a sleep
                            bout until pressure clears, giving clean rhythms
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

from sleep_env import SleepEnv
from a2c_agent import ActorCritic, train, rollout_greedy

WAKE_C, SLEEP_C = "#6C4AB6", "#B79CED"


def plot_learning_curve(returns, path):
    returns = np.asarray(returns)
    sm = np.convolve(returns, np.ones(25) / 25, mode="valid")
    plt.figure(figsize=(9, 4))
    plt.plot(returns, color=SLEEP_C, alpha=0.4, lw=1, label="episode return")
    plt.plot(np.arange(len(sm)) + 12, sm, color=WAKE_C, lw=2.2,
             label="25-episode moving average")
    plt.xlabel("Training episode"); plt.ylabel("Episode return")
    plt.title("Actor-Critic learning curve on SleepEnv", fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_rollout(S, E, A, MU, path,
                 title="Learned greedy policy (Actor-Critic)"):
    n = len(S)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(S, color="crimson", lw=2, label="Homeostatic Sleep Pressure (S)")
    ax1.plot(E, color="teal", lw=2, label="Energy Reserves (E)")
    ax1.set_ylabel("State magnitude"); ax1.set_title(title, fontweight="bold")
    ax1.legend(loc="upper right"); ax1.grid(True, alpha=0.3)

    # mu = pi(SLEEP|s), the sleep-drive signal, with sleep periods shaded.
    ax2.fill_between(range(n), 0, (A == 1).astype(float), step="mid",
                     color=SLEEP_C, alpha=0.35, label="Sleep period")
    ax2.plot(MU, color=WAKE_C, lw=2, label=r"$\mu = \pi(\mathrm{SLEEP}\,|\,s)$")
    ax2.set_ylim(-0.02, 1.02); ax2.set_xlabel("Time step")
    ax2.set_ylabel("Sleep drive  /  action")
    ax2.set_title(r"Sleep drive $\mu$ (bridge to the Wilson-Cowan oscillator)",
                  fontweight="bold")
    ax2.legend(loc="upper right"); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_policy_map(net, path, res=160):
    s_vals = np.linspace(0, 1, res); e_vals = np.linspace(0, 1, res)
    grid = np.zeros((res, res), dtype=int)
    with torch.no_grad():
        for i, e in enumerate(e_vals):
            batch = np.stack([s_vals, np.full_like(s_vals, e)], axis=1)
            logits, _ = net.forward(torch.as_tensor(batch, dtype=torch.float32))
            grid[i] = torch.argmax(logits, dim=1).numpy()
    from matplotlib.colors import ListedColormap
    plt.figure(figsize=(7, 6))
    plt.imshow(grid, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
               cmap=ListedColormap([WAKE_C, SLEEP_C]), vmin=0, vmax=1)
    cbar = plt.colorbar(ticks=[0.25, 0.75]); cbar.ax.set_yticklabels(["Wake", "Sleep"])
    plt.xlabel("Homeostatic Sleep Pressure (S)"); plt.ylabel("Energy Reserves (E)")
    plt.title("Learned wake/sleep boundary over (S, E)", fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


@torch.no_grad()
def rollout_macrolock(net, clearance=0.15, seed=123):
    """Learned entry decision + deterministic commit-until-cleared lock."""
    env = SleepEnv(); obs, _ = env.reset(seed=seed)
    S_h, E_h, A_h, R_h, MU_h = [], [], [], [], []
    in_bout, done = False, False
    while not done:
        logits, _ = net.forward(torch.as_tensor(obs, dtype=torch.float32))
        mu = float(torch.softmax(logits, dim=-1)[SleepEnv.SLEEP].item())
        greedy = int(torch.argmax(logits).item())
        if in_bout:
            a = SleepEnv.WAKE if obs[0] <= clearance else SleepEnv.SLEEP
            if a == SleepEnv.WAKE:
                in_bout = False
        else:
            a = greedy
            if a == SleepEnv.SLEEP:
                in_bout = True
        S_h.append(obs[0]); E_h.append(obs[1]); MU_h.append(mu)
        obs, r, term, trunc, _ = env.step(a)
        A_h.append(a); R_h.append(r); done = term or trunc
    return (np.array(S_h), np.array(E_h), np.array(A_h),
            np.array(R_h), np.array(MU_h))


def _greedy_policy_grid(net, res=160):
    """Argmax action over the (S, E) plane; grid[i] is row for energy e_vals[i]."""
    s_vals = np.linspace(0, 1, res); e_vals = np.linspace(0, 1, res)
    grid = np.zeros((res, res), dtype=int)
    with torch.no_grad():
        for i, e in enumerate(e_vals):
            batch = np.stack([s_vals, np.full_like(s_vals, e)], axis=1)
            logits, _ = net.forward(torch.as_tensor(batch, dtype=torch.float32))
            grid[i] = torch.argmax(logits, dim=1).numpy()
    return grid, e_vals


def verify_fix(net, S, E, returns):
    """Fail-closed verification of the gain==clear=0.375 co-readout fix.

    Returns (all_passed, summary_dict). Prints each check. The caller must NOT
    save figures/weights unless all_passed is True.
    """
    S = np.asarray(S, dtype=np.float64); E = np.asarray(E, dtype=np.float64)
    max_S = float(S.max())
    ident_err = float(np.max(np.abs(S - (0.1 + 0.375 * (1.0 - E)))))
    plateau = float(np.mean(returns[-100:]))

    grid, e_vals = _greedy_policy_grid(net)
    sleep_rows = np.where((grid == 1).any(axis=1))[0]   # energy rows containing SLEEP
    has_sleep = sleep_rows.size > 0
    max_sleep_E = float(e_vals[sleep_rows.max()]) if has_sleep else 0.0
    # "single clean boundary, sleep only at low E": the sleep energy-rows are a
    # contiguous band starting at E=0 (i.e. the low-E floor), not scattered.
    contiguous_low = bool(has_sleep and sleep_rows.min() == 0
                          and sleep_rows.max() == sleep_rows.size - 1)
    sleep_low_E = bool(has_sleep and max_sleep_E < 0.6)

    checks = [
        ("max(S) < 0.5 (drift gone)",            max_S < 0.5,
         f"max(S)={max_S:.4f}"),
        ("S == 0.1+0.375(1-E) within 1e-6",      ident_err < 1e-6,
         f"max|err|={ident_err:.2e}"),
        ("final-100 mean return in [100,130]",   100.0 <= plateau <= 130.0,
         f"plateau={plateau:.2f}"),
        ("policy map: single low-E sleep band",  contiguous_low and sleep_low_E,
         f"sleep up to E={max_sleep_E:.3f}, contiguous={contiguous_low}"),
    ]
    print("\n" + "=" * 60 + "\nVERIFICATION (fail-closed)\n" + "=" * 60)
    all_passed = True
    for name, ok, detail in checks:
        all_passed &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")
    return all_passed, {"max_S": max_S, "ident_err": ident_err,
                        "plateau": plateau, "max_sleep_E": max_sleep_E}


if __name__ == "__main__":
    import sys

    net, returns = train(episodes=1200, seed=0)

    S, E, A, R, MU = rollout_greedy(net, seed=123)
    frac = np.bincount(A, minlength=2) / len(A)
    print(f"\ngreedy return {R.sum():.2f} | WAKE {frac[0]:.2f} SLEEP {frac[1]:.2f}"
          f" | mu in [{MU.min():.2f}, {MU.max():.2f}]")

    passed, m = verify_fix(net, S, E, returns)
    if not passed:
        print("\nSTOP: one or more verification checks FAILED. Not saving "
              "figures or weights. Re-inspect the fix before proceeding.")
        sys.exit(1)

    # Checks passed -> safe to persist artifacts.
    torch.save(net.state_dict(), "sleep_actor_critic.pt")
    plot_learning_curve(returns, "learning_curve.png")
    plot_rollout(S, E, A, MU, "learned_rollout.png")
    plot_policy_map(net, "policy_map.png")

    Sl, El, Al, Rl, MUl = rollout_macrolock(net)
    sw = int(np.sum(Al[1:] != Al[:-1]))
    print(f"macro-lock rollout: return {Rl.sum():.1f} | switches {sw} "
          f"| sleep frac {np.mean(Al == 1):.2f}")
    plot_rollout(Sl, El, Al, MUl, "consolidated_bouts.png",
                 title="Learned entry + macro-lock: consolidated sleep bouts")
    print("saved figures + sleep_actor_critic.pt")

    # ---- One-paragraph numeric summary (old vs new) ----
    OLD_MAX_S = 0.955   # greedy-rollout max(S) under the old gain!=clear dynamics
    print("\n" + "-" * 60 + "\nSUMMARY\n" + "-" * 60)
    print(
        f"With adenosine_clear set equal to adenosine_gain (c=0.375), the greedy "
        f"rollout's max(S) drops from ~{OLD_MAX_S:.3f} (old, saturating) to "
        f"{m['max_S']:.4f} -- the drift to the ceiling is gone. S now spans "
        f"[{float(np.min(S)):.4f}, {m['max_S']:.4f}], matching the exact affine "
        f"law S = 0.1 + 0.375*(1 - E) to within {m['ident_err']:.1e}, so S is a "
        f"bounded co-readout of E rather than a clipped, drifting variable. The "
        f"final 100-episode mean return is {m['plateau']:.2f}, i.e. the learning "
        f"plateau is intact.")
