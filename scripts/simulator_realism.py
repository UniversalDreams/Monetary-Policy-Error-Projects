"""
scripts/simulator_realism.py — Simulator realism check.

Runs the COVID counterfactual environment using the real Fed's actual rate
path as the policy. Compares simulated inflation and unemployment against
real BLS/FRED data to quantify how well the MacroSimulator captures
historical dynamics.

Usage:
    .venv/Scripts/python scripts/simulator_realism.py
    .venv/Scripts/python scripts/simulator_realism.py --out runs/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from covid_env import CovidEnv, COVID_DATA, _N_REAL


def _month_label(t: int) -> str:
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][t % 12]
    return f"{month}'{str(2020 + t // 12)[2:]}"


def run(out_dir: str) -> None:
    env = CovidEnv(mode="counterfactual")
    obs, _ = env.reset(seed=0)

    pi_hist, u_hist, rate_hist = [], [], []
    done = False

    while not done:
        t_idx = min(env.t, _N_REAL - 1)
        target_rate  = COVID_DATA["effr"][t_idx]
        current_rate = float(obs["macro"][2])

        # Greedy: pick the action that moves current_rate closest to real EFFR
        best_action, best_diff = 3, float("inf")
        for idx, delta in env.action_mapping.items():
            diff = abs(np.clip(current_rate + delta, 0.0, 20.0) - target_rate)
            if diff < best_diff:
                best_diff  = diff
                best_action = idx

        pi_hist.append(float(obs["macro"][0]))
        u_hist.append(float(obs["macro"][1]))
        rate_hist.append(current_rate)

        obs, _, term, trunc, _ = env.step(best_action)
        done = term or trunc

    # Trim to real-data window (Jan 2020 – Dec 2025)
    pi_hist   = pi_hist[:_N_REAL]
    u_hist    = u_hist[:_N_REAL]
    rate_hist = rate_hist[:_N_REAL]

    n_steps     = _N_REAL
    all_months  = list(range(n_steps))
    real_months = list(range(_N_REAL))

    # ── Summary stats (over the real-data window only) ────────────────────────
    sim_pi = np.array(pi_hist[:_N_REAL])
    sim_u  = np.array(u_hist[:_N_REAL])
    real_pi = np.array(COVID_DATA["cpi"][:_N_REAL])
    real_u  = np.array(COVID_DATA["unemployment"][:_N_REAL])

    pi_mae  = float(np.mean(np.abs(sim_pi - real_pi)))
    u_mae   = float(np.mean(np.abs(sim_u  - real_u)))
    pi_rmse = float(np.sqrt(np.mean((sim_pi - real_pi) ** 2)))
    u_rmse  = float(np.sqrt(np.mean((sim_u  - real_u)  ** 2)))

    print(f"Simulator realism  ({_N_REAL} months, Jan 2020 – {_month_label(_N_REAL - 1)})")
    print(f"  Inflation    MAE={pi_mae:.2f}pp   RMSE={pi_rmse:.2f}pp")
    print(f"  Unemployment MAE={u_mae:.2f}pp   RMSE={u_rmse:.2f}pp")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Phase shading
    for ax in axes:
        ax.axvspan(2,  5,  color="gold",   alpha=0.18, label="Lockdown")
        ax.axvspan(14, 34, color="salmon",  alpha=0.18, label="Supply chain")
        if n_steps > _N_REAL:
            ax.axvline(_N_REAL - 1, color="gray", linestyle=":", linewidth=1.2,
                       label="Real data ends")
        ax.grid(True, alpha=0.3)

    # Inflation
    ax = axes[0]
    ax.plot(real_months, COVID_DATA["cpi"], "r--", lw=1.5, label="Real CPI")
    ax.plot(all_months,  pi_hist,           "r-",  lw=2,   label=f"Simulated pi  (MAE={pi_mae:.2f}pp)")
    ax.axhline(2.0, color="red", ls=":", alpha=0.4)
    ax.set_ylabel("Inflation (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Simulator realism: real Fed rate path as policy", fontweight="bold")

    # Unemployment
    ax = axes[1]
    ax.plot(real_months, COVID_DATA["unemployment"], "b--", lw=1.5, label="Real U-3")
    ax.plot(all_months,  u_hist,                     "b-",  lw=2,   label=f"Simulated u  (MAE={u_mae:.2f}pp)")
    ax.axhline(4.0, color="blue", ls=":", alpha=0.4)
    ax.set_ylabel("Unemployment (%)")
    ax.legend(loc="upper right", fontsize=8)

    # Policy rate
    ax = axes[2]
    ax.step(real_months, COVID_DATA["effr"], "g--", lw=1.5, where="post",
            label="Real EFFR")
    ax.step(all_months,  rate_hist,          "g-",  lw=2,   where="post",
            label="Simulated rate (discrete actions, max +-0.75pp/month)")
    ax.set_ylabel("Policy Rate (%)")
    ax.set_xlabel("Month")
    ax.legend(loc="upper left", fontsize=8)

    ticks = list(range(0, n_steps, 6))
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels([_month_label(t) for t in ticks], rotation=45, fontsize=8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "simulator_realism.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Simulator realism check vs real COVID data")
    parser.add_argument("--out", type=str, default="runs",
                        help="Output directory for the plot (default: runs/)")
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
