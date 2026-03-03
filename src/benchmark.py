"""
src/benchmark.py — Evaluate and compare all conditions + Taylor Rule baseline.

Loads saved PPO models from disk and evaluates on fixed seeds:
  - taylor_rule : classical rule-based policy (no training required)
  - baseline    : PPO trained without LLM belief state
  - llm         : PPO trained with state-keyed LLM belief DB

Usage (run directory — auto-discovers models):
    .venv/Scripts/python src/benchmark.py --run runs/20260301_143022_lstm

Usage (manual model paths):
    .venv/Scripts/python src/benchmark.py \\
        --base-model    runs/paper_run/baseline/model \\
        --offline-model runs/paper_run/llm/model \\
        --db            data/state_belief_db.json \\
        --out           runs/paper_run
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_SB3_CONTRIB = True
except ImportError:
    _HAS_SB3_CONTRIB = False

sys.path.insert(0, os.path.dirname(__file__))

import config
from fed_env import FedEnvBase, StateKeyedLLMWrapper, MockLLMObservationWrapper


# ─────────────────────────────────────────────────────────────
# METADATA HELPER
# ─────────────────────────────────────────────────────────────

def _save_metadata(run_dir: str, meta: dict) -> None:
    """Atomic JSON write of metadata.json."""
    path = os.path.join(run_dir, "metadata.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────
# TAYLOR RULE
# ─────────────────────────────────────────────────────────────

def _taylor_action(obs: dict, action_mapping: dict) -> int:
    """
    Standard Taylor Rule: r* = 2 + π + 0.5(π - 2) - 0.5(u - 4)
    Maps desired continuous delta to nearest discrete action.
    """
    pi, u, current_rate = obs["macro"]
    target_rate = 2.0 + pi + 0.5 * (pi - 2.0) - 0.5 * (u - 4.0)
    desired_delta = target_rate - current_rate
    best_action, min_diff = 3, float("inf")
    for idx, delta in action_mapping.items():
        diff = abs(delta - desired_delta)
        if diff < min_diff:
            min_diff = diff
            best_action = idx
    return best_action


def evaluate_taylor_rule(n_seeds: int) -> list[float]:
    rewards = []
    env = FedEnvBase(llm_dim=config.LLM_DIM)
    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        ep_reward, done = 0.0, False
        while not done:
            action = _taylor_action(obs, env.unwrapped.action_mapping)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return rewards


# ─────────────────────────────────────────────────────────────
# PPO EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_ppo(model, env_factory, n_seeds: int, policy: str = "mlp") -> list[float]:
    rewards = []
    for seed in range(n_seeds):
        env = env_factory()
        obs, _ = env.reset(seed=seed)
        ep_reward, done = 0.0, False
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        while not done:
            if policy == "lstm":
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_start, deterministic=True
                )
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        env.close()
    return rewards


# ─────────────────────────────────────────────────────────────
# TRAJECTORY COLLECTION  (one representative episode per condition)
# ─────────────────────────────────────────────────────────────

def _collect_trajectory_taylor(seed: int = 0) -> dict:
    env = FedEnvBase(llm_dim=config.LLM_DIM)
    obs, _ = env.reset(seed=seed)
    shock_start = env.unwrapped.shock_start
    shock_end   = env.unwrapped.shock_end
    pi_hist, u_hist, rate_hist = [], [], []
    done = False
    while not done:
        pi, u, rate = obs["macro"]
        pi_hist.append(float(pi))
        u_hist.append(float(u))
        rate_hist.append(float(rate))
        action = _taylor_action(obs, env.unwrapped.action_mapping)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    return {"pi": pi_hist, "u": u_hist, "rate": rate_hist,
            "shock_start": shock_start, "shock_end": shock_end}


def _collect_trajectory_ppo(model, env_factory, seed: int = 0,
                             policy: str = "mlp") -> dict:
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    shock_start = env.unwrapped.shock_start
    shock_end   = env.unwrapped.shock_end
    pi_hist, u_hist, rate_hist = [], [], []
    done = False
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)
    while not done:
        pi, u, rate = obs["macro"]
        pi_hist.append(float(pi))
        u_hist.append(float(u))
        rate_hist.append(float(rate))
        if policy == "lstm":
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    env.close()
    return {"pi": pi_hist, "u": u_hist, "rate": rate_hist,
            "shock_start": shock_start, "shock_end": shock_end}


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

_CONDITION_STYLES = {
    "taylor_rule": {"color": "green",     "label": "Taylor Rule"},
    "baseline":    {"color": "purple",    "label": "Baseline PPO"},
    "oracle":      {"color": "steelblue", "label": "Oracle PPO"},
    "llm":         {"color": "darkorange","label": "LLM PPO"},
}


def plot_reward_comparison(results: dict[str, list[float]], out_dir: str) -> None:
    """Bar chart of mean ± std reward for each condition."""
    conditions = list(results.keys())
    means = [np.mean(results[c]) for c in conditions]
    stds  = [np.std(results[c])  for c in conditions]
    colors = [_CONDITION_STYLES.get(c, {}).get("color", "steelblue") for c in conditions]
    labels = [_CONDITION_STYLES.get(c, {}).get("label", c) for c in conditions]

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(conditions)), 5))
    x = range(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.8, width=0.5)
    ax.axhline(means[0], color=colors[0], linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title(f"Benchmark Reward Comparison  ({len(results[conditions[0]])} seeds)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:+.1f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_rewards.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_training_curves(run_dir: str, out_dir: str, window: int = 100) -> None:
    """Smoothed episode reward over training for each condition that has a training.csv."""
    # conditions that live in subdirs of run_dir
    cond_dirs = {
        "baseline": os.path.join(run_dir, "baseline", "training.csv"),
        "oracle":   os.path.join(run_dir, "oracle",   "training.csv"),
        "llm":      os.path.join(run_dir, "llm",      "training.csv"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for cond, csv_path in cond_dirs.items():
        if not os.path.exists(csv_path):
            continue
        episodes, rewards = [], []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                episodes.append(int(row["episode"]))
                rewards.append(float(row["ep_reward"]))

        style = _CONDITION_STYLES.get(cond, {"color": "steelblue", "label": cond})
        rewards_arr = np.array(rewards)
        # raw (faint)
        ax.plot(episodes, rewards_arr, color=style["color"], alpha=0.15, linewidth=0.8)
        # rolling mean
        if len(rewards_arr) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(rewards_arr, kernel, mode="valid")
            ax.plot(episodes[window - 1:], smoothed,
                    color=style["color"], linewidth=2, label=style["label"])
        else:
            ax.plot(episodes, rewards_arr,
                    color=style["color"], linewidth=2, label=style["label"])
        plotted = True

    if not plotted:
        plt.close()
        return

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title(f"Training Curves  (smoothed over {window} episodes)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_trajectories(trajectories: dict[str, dict], out_dir: str) -> None:
    """Dual-panel trajectory plot (macro vars + policy rate) for each condition."""
    conditions = list(trajectories.keys())
    n = len(conditions)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), sharex="col")
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, cond in enumerate(conditions):
        h  = trajectories[cond]
        style = _CONDITION_STYLES.get(cond, {"color": "steelblue", "label": cond})
        t  = range(len(h["pi"]))
        ss, se = h["shock_start"], h["shock_end"]

        ax_top = axes[0][col]
        ax_top.plot(t, h["pi"], label="Inflation (%)",   color="red",  linewidth=2)
        ax_top.plot(t, h["u"],  label="Unemployment (%)", color="blue", linewidth=2)
        ax_top.axvspan(ss, se, color="gray", alpha=0.2, label="Supply Shock")
        ax_top.axhline(2.0, color="red",  linestyle="--", alpha=0.4, linewidth=1)
        ax_top.axhline(4.0, color="blue", linestyle="--", alpha=0.4, linewidth=1)
        ax_top.set_title(style["label"], fontweight="bold", fontsize=12)
        ax_top.set_ylabel("Rate (%)")
        ax_top.legend(loc="upper left", fontsize=8)
        ax_top.grid(True, alpha=0.3)

        ax_bot = axes[1][col]
        ax_bot.step(t, h["rate"], label="Policy Rate", color=style["color"],
                    linewidth=2, where="post")
        ax_bot.axvspan(ss, se, color="gray", alpha=0.2)

        # First rate hike after shock onset
        rates = h["rate"]
        first_hike = next(
            (i for i in range(ss + 1, len(rates)) if rates[i] > rates[i - 1]),
            None,
        )
        if first_hike is not None:
            latency = first_hike - ss
            ax_bot.axvline(first_hike, color="black", linestyle=":", linewidth=2,
                           label=f"Reaction lag: {latency}mo")

        ax_bot.set_xlabel("Timestep (months)")
        ax_bot.set_ylabel("Interest Rate (%)")
        ax_bot.legend(loc="upper left", fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle("Representative Episode Trajectories (seed 42)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────

def _print_table(results: dict[str, list[float]], n_seeds: int) -> str:
    taylor_mean = np.mean(results["taylor_rule"])
    lines = [
        f"Benchmark — {n_seeds} evaluation seeds",
        "─" * 56,
        f"{'condition':<16} {'mean':>9} {'std':>8} {'Δtaylor':>10}",
        "─" * 56,
    ]
    for name, rewards in results.items():
        mean, std = np.mean(rewards), np.std(rewards)
        delta = mean - taylor_mean if name != "taylor_rule" else 0.0
        delta_str = f"{delta:+.2f}" if name != "taylor_rule" else "—"
        lines.append(f"{name:<16} {mean:>+9.2f} {std:>8.2f} {delta_str:>10}")
    lines.append("─" * 56)
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark all conditions vs Taylor Rule")
    parser.add_argument("--run",           type=str, default=None,
                        help="Path to run directory (auto-discovers models and metadata)")
    parser.add_argument("--base-model",    type=str, default=None,
                        help="Path to baseline PPO model (without .zip); overrides --run discovery")
    parser.add_argument("--offline-model", type=str, default=None,
                        help="Path to LLM PPO model (without .zip); overrides --run discovery")
    parser.add_argument("--oracle-model",  type=str, default=None,
                        help="Path to oracle PPO model (without .zip); overrides --run discovery")
    parser.add_argument("--db",            type=str, default=config.DEFAULT_STATE_DB_PATH,
                        help=f"State-keyed DB for LLM eval (default {config.DEFAULT_STATE_DB_PATH})")
    parser.add_argument("--out",           type=str, default=None,
                        help="Output directory (default: run dir if --run provided, else runs/)")
    parser.add_argument("--seeds",         type=int, default=config.EVAL_SEEDS,
                        help=f"Number of evaluation seeds (default {config.EVAL_SEEDS})")
    parser.add_argument("--policy",        type=str, default="mlp",
                        choices=["mlp", "lstm"],
                        help="Policy architecture used during training (default: mlp)")
    parser.add_argument("--ema",           action="store_true",
                        help="Use best_model_ema.zip instead of best_model.zip during auto-discovery")
    args = parser.parse_args()

    # ── Resolve run dir, model paths, and metadata ─────────────
    run_meta = None
    run_dir = args.run

    if run_dir:
        meta_path = os.path.join(run_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                run_meta = json.load(f)
            # Inherit policy from metadata if user didn't explicitly set it
            if args.policy == "mlp" and run_meta.get("policy"):
                args.policy = run_meta["policy"]

        # Auto-discover model paths (explicit args take precedence).
        # Prefer best_model_ema.zip (--ema) or best_model.zip over final model.zip.
        for arg_attr, cond_name in [
            ("base_model",    "baseline"),
            ("offline_model", "llm"),
            ("oracle_model",  "oracle"),
        ]:
            if getattr(args, arg_attr) is not None:
                continue
            cond_dir    = os.path.join(run_dir, cond_name)
            ema_path    = os.path.join(cond_dir, "best_model_ema")
            best_path   = os.path.join(cond_dir, "best_model")
            final_path  = os.path.join(cond_dir, "model")

            if args.ema and os.path.exists(ema_path + ".zip"):
                setattr(args, arg_attr, ema_path)
                print(f"[auto] {cond_name}: using best_model_ema.zip")
            elif os.path.exists(best_path + ".zip"):
                setattr(args, arg_attr, best_path)
                print(f"[auto] {cond_name}: using best_model.zip")
            elif os.path.exists(final_path + ".zip"):
                setattr(args, arg_attr, final_path)
                print(f"[auto] {cond_name}: using model.zip")

        # Use db_path from metadata if user didn't override
        if run_meta and args.db == config.DEFAULT_STATE_DB_PATH:
            args.db = run_meta.get("db_path", args.db)

    out_dir = args.out or run_dir or config.DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    def _load_model(path: str):
        if args.policy == "lstm":
            if not _HAS_SB3_CONTRIB:
                raise ImportError("sb3_contrib required for LSTM. pip install sb3-contrib")
            return RecurrentPPO.load(path)
        return PPO.load(path)

    results:     dict[str, list[float]] = {}
    trajectories: dict[str, dict]       = {}

    # ── Taylor Rule ───────────────────────────────────────────
    print(f"\n>>> Taylor Rule  ({args.seeds} seeds) …")
    results["taylor_rule"] = evaluate_taylor_rule(args.seeds)
    trajectories["taylor_rule"] = _collect_trajectory_taylor(seed=42)
    print(f"    mean={np.mean(results['taylor_rule']):+.2f}  "
          f"std={np.std(results['taylor_rule']):.2f}")

    # ── Baseline PPO ──────────────────────────────────────────
    if args.base_model:
        print(f"\n>>> Baseline PPO  loading from {args.base_model} …")
        base_model = _load_model(args.base_model)
        print(f"    Evaluating on {args.seeds} seeds …")
        results["baseline"] = evaluate_ppo(
            base_model,
            lambda: FedEnvBase(llm_dim=config.LLM_DIM),
            args.seeds,
            args.policy,
        )
        trajectories["baseline"] = _collect_trajectory_ppo(
            base_model, lambda: FedEnvBase(llm_dim=config.LLM_DIM), seed=42, policy=args.policy
        )
        print(f"    mean={np.mean(results['baseline']):+.2f}  "
              f"std={np.std(results['baseline']):.2f}")

    # ── LLM PPO ───────────────────────────────────────────────
    if args.offline_model:
        if not os.path.exists(args.db):
            print(f"\nERROR: DB not found at {args.db}")
            print("Run build_state_db.py first, or pass --db <path>")
            sys.exit(1)
        print(f"\n>>> LLM PPO  loading from {args.offline_model} …")
        llm_model = _load_model(args.offline_model)
        print(f"    Evaluating on {args.seeds} seeds …")
        db_path = args.db
        results["llm"] = evaluate_ppo(
            llm_model,
            lambda: StateKeyedLLMWrapper(FedEnvBase(llm_dim=config.LLM_DIM), db_path=db_path),
            args.seeds,
            args.policy,
        )
        trajectories["llm"] = _collect_trajectory_ppo(
            llm_model,
            lambda: StateKeyedLLMWrapper(FedEnvBase(llm_dim=config.LLM_DIM), db_path=db_path),
            seed=42, policy=args.policy,
        )
        print(f"    mean={np.mean(results['llm']):+.2f}  "
              f"std={np.std(results['llm']):.2f}")

    # ── Oracle PPO ────────────────────────────────────────────
    if args.oracle_model:
        print(f"\n>>> Oracle PPO  loading from {args.oracle_model} …")
        oracle_model = _load_model(args.oracle_model)
        print(f"    Evaluating on {args.seeds} seeds …")
        results["oracle"] = evaluate_ppo(
            oracle_model,
            lambda: MockLLMObservationWrapper(FedEnvBase(llm_dim=config.LLM_DIM)),
            args.seeds,
            args.policy,
        )
        trajectories["oracle"] = _collect_trajectory_ppo(
            oracle_model,
            lambda: MockLLMObservationWrapper(FedEnvBase(llm_dim=config.LLM_DIM)),
            seed=42, policy=args.policy,
        )
        print(f"    mean={np.mean(results['oracle']):+.2f}  "
              f"std={np.std(results['oracle']):.2f}")

    # ── Table ─────────────────────────────────────────────────
    table = _print_table(results, args.seeds)
    print("\n" + table)

    txt_path = os.path.join(out_dir, "benchmark.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved → {txt_path}")

    # ── Per-seed CSV ──────────────────────────────────────────
    csv_path = os.path.join(out_dir, "benchmark.csv")
    conditions = list(results.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + conditions)
        for seed in range(args.seeds):
            writer.writerow([seed] + [round(results[c][seed], 4) for c in conditions])
    print(f"Saved → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────
    if run_dir:
        plot_training_curves(run_dir, out_dir)
    plot_reward_comparison(results, out_dir)
    plot_trajectories(trajectories, out_dir)

    # ── Update metadata.json ──────────────────────────────────
    if run_meta is not None and run_dir:
        run_meta["benchmark"] = {
            "seeds": args.seeds,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "conditions": {
                name: {
                    "mean": round(float(np.mean(rewards)), 4),
                    "std":  round(float(np.std(rewards)), 4),
                }
                for name, rewards in results.items()
            },
        }
        _save_metadata(run_dir, run_meta)
        print(f"Metadata updated → {os.path.join(run_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
