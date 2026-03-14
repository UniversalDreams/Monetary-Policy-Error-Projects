"""
src/benchmark_custom.py — Evaluate and compare all conditions + Taylor Rule baseline.

Loads saved custom PPO models (.pt state dicts) from disk and evaluates on fixed seeds:
  - taylor_rule : classical rule-based policy (no training required)
  - baseline    : PPO trained without LLM belief state
  - llm         : PPO trained with state-keyed LLM belief DB
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

import config
from fed_env import FedEnvBase, StateKeyedLLMWrapper, MockLLMObservationWrapper
from covid_env import CovidEnv, covid_eval as _covid_eval
from network import ActorCritic, LSTMActorCritic


# -------------------------------------------------------------
# METADATA HELPER
# -------------------------------------------------------------

def _save_metadata(run_dir: str, meta: dict) -> None:
    """Atomic JSON write of metadata.json."""
    path = os.path.join(run_dir, "metadata.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp, path)


# -------------------------------------------------------------
# CUSTOM MODEL ADAPTER
# -------------------------------------------------------------

def _load_obs_rms(vecnorm_path: str):
    """Load obs running stats from a VecNormalize .pkl file. Returns obs_rms or None."""
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return None
    try:
        import pickle
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        return getattr(vn, "obs_rms", None)
    except (RecursionError, Exception):
        # SB3 VecNormalize pkl can trigger infinite recursion in __getattr__
        # on some versions. Safe to return None since norm_obs=False during training.
        return None


def _normalize_obs(obs: dict, obs_rms, clip: float = 10.0) -> dict:
    result = {}
    for key, val in obs.items():
        if obs_rms is not None and hasattr(obs_rms, "__getitem__") and key in obs_rms:
            rms = obs_rms[key]
            normalized = (np.asarray(val) - rms.mean) / np.sqrt(rms.var + 1e-8)
            result[key] = np.clip(normalized, -clip, clip).astype(np.float32)
        else:
            result[key] = val
    return result


def _obs_to_tensor(obs: dict, device: str, obs_rms=None, clip: float = 10.0) -> torch.Tensor:
    """Convert single-env obs dict to (1, obs_dim) tensor (llm then macro)."""
    if obs_rms is not None:
        obs = _normalize_obs(obs, obs_rms, clip)
    llm   = torch.tensor(obs["llm_belief"], dtype=torch.float32, device=device)
    macro = torch.tensor(obs["macro"],      dtype=torch.float32, device=device)
    return torch.cat([llm, macro], dim=-1).unsqueeze(0)  # (1, obs_dim)


class _ModelAdapter:
    """Wraps ActorCritic / LSTMActorCritic to expose a predict() interface."""

    def __init__(self, ac: nn.Module, policy: str, device: str, obs_rms=None):
        self._ac = ac
        self._policy = policy
        self._device = device
        self._obs_rms = obs_rms

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs_t = _obs_to_tensor(obs, self._device, obs_rms=self._obs_rms)
        with torch.no_grad():
            if self._policy == "lstm":
                ep_t = torch.tensor(episode_start, dtype=torch.float32, device=self._device)
                action, _, _, new_state = self._ac.get_action_and_value(obs_t, state, ep_t)
                return action.cpu().numpy()[0], new_state
            else:
                action, _, _, _ = self._ac.get_action_and_value(obs_t, None, None)
                return action.cpu().numpy()[0], None


def _load_model(path: str, policy: str, device: str,
                vecnorm_path: str = None,
                lstm_hidden_size: int = None) -> _ModelAdapter:
    """Instantiate custom network, load .pt state dict, and wrap in _ModelAdapter."""
    # Fixed dims: obs_dim=8 (5 llm + 3 macro), act_dim=7
    obs_dim, act_dim = 8, 7
    hidden = lstm_hidden_size or config.LSTM_HIDDEN_SIZE
    if policy == "lstm":
        ac = LSTMActorCritic(obs_dim, act_dim, lstm_hidden_size=hidden, n_lstm_layers=1).to(device)
    else:
        ac = ActorCritic(obs_dim, act_dim).to(device)
    # Try loading; if size mismatch, fall back to hidden=64 (old runs)
    try:
        ac.load_state_dict(torch.load(path + ".pt", map_location=device, weights_only=True))
    except RuntimeError:
        if policy == "lstm" and hidden != 64:
            ac = LSTMActorCritic(obs_dim, act_dim, lstm_hidden_size=64, n_lstm_layers=1).to(device)
            ac.load_state_dict(torch.load(path + ".pt", map_location=device, weights_only=True))
        else:
            raise
    ac.eval()
    obs_rms = _load_obs_rms(vecnorm_path)
    return _ModelAdapter(ac, policy, device, obs_rms=obs_rms)


# -------------------------------------------------------------
# TAYLOR RULE
# -------------------------------------------------------------

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


# -------------------------------------------------------------
# PPO EVALUATION
# -------------------------------------------------------------

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


# -------------------------------------------------------------
# TRAJECTORY COLLECTION  (one representative episode per condition)
# -------------------------------------------------------------

def _collect_trajectory_taylor(seed: int = 0) -> dict:
    env = FedEnvBase(llm_dim=config.LLM_DIM)
    obs, _ = env.reset(seed=seed)
    shock_start = env.unwrapped.shock_start
    shock_end   = env.unwrapped.shock_end
    pi_hist, u_hist, rate_hist = [], [], []
    ep_reward = 0.0
    done = False
    while not done:
        pi, u, rate = obs["macro"]
        pi_hist.append(float(pi))
        u_hist.append(float(u))
        rate_hist.append(float(rate))
        action = _taylor_action(obs, env.unwrapped.action_mapping)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        done = terminated or truncated
    env.close()
    return {"pi": pi_hist, "u": u_hist, "rate": rate_hist,
            "shock_start": shock_start, "shock_end": shock_end,
            "ep_reward": ep_reward}


def _collect_trajectory_ppo(model, env_factory, seed: int = 0,
                             policy: str = "mlp") -> dict:
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    shock_start = env.unwrapped.shock_start
    shock_end   = env.unwrapped.shock_end
    pi_hist, u_hist, rate_hist = [], [], []
    ep_reward = 0.0
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
        obs, reward, terminated, truncated, _ = env.step(int(action))
        ep_reward += reward
        done = terminated or truncated
    env.close()
    return {"pi": pi_hist, "u": u_hist, "rate": rate_hist,
            "shock_start": shock_start, "shock_end": shock_end,
            "ep_reward": ep_reward}


# -------------------------------------------------------------
# SEED SCANNER
# -------------------------------------------------------------

def find_seed(criterion, max_scan: int = 500) -> int:
    """Return first seed in [0, max_scan) where criterion(shock_start, shock_end, shock_scale, max_steps) is True."""
    env = FedEnvBase(llm_dim=config.LLM_DIM)
    for seed in range(max_scan):
        env.reset(seed=seed)
        u = env.unwrapped
        if criterion(u.shock_start, u.shock_end, u.shock_scale, u.max_steps):
            env.close()
            return seed
    env.close()
    raise RuntimeError(f"No matching seed found in [0, {max_scan})")


def _find_scenario_seeds() -> dict[str, int]:
    return {
        "No-Shock":       find_seed(lambda ss, se, sc, ms: ss > ms),
        "Min Shock":      find_seed(lambda ss, se, sc, ms: ss <= ms and sc <= config.SHOCK_SCALE_MIN + 0.1),
        "Standard Shock": 42,
        "Intense Shock":  find_seed(lambda ss, se, sc, ms: ss <= ms and sc >= 1.3),
    }


# -------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------

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
    print(f"Saved -> {path}")


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
    print(f"Saved -> {path}")


def plot_trajectories(trajectories: dict[str, dict], out_dir: str,
                      title: str = "Representative Episode Trajectories (seed 42)",
                      filename: str = "benchmark_trajectories.png") -> None:
    """Dual-panel trajectory plot (macro vars + policy rate) for each condition."""
    conditions = list(trajectories.keys())
    n = len(conditions)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), sharex="col", sharey="row")
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
        ep_reward = h.get("ep_reward")
        reward_str = f"  (reward: {ep_reward:+.0f})" if ep_reward is not None else ""
        ax_top.set_title(style["label"] + reward_str, fontweight="bold", fontsize=12)
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

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


# -------------------------------------------------------------
# REPORTING
# -------------------------------------------------------------

def _print_table(results: dict[str, list[float]], n_seeds: int) -> str:
    taylor_mean = np.mean(results["taylor_rule"])
    lines = [
        f"Benchmark — {n_seeds} evaluation seeds",
        "-" * 56,
        f"{'condition':<16} {'mean':>9} {'std':>8} {'d_taylor':>10}",
        "-" * 56,
    ]
    for name, rewards in results.items():
        mean, std = np.mean(rewards), np.std(rewards)
        delta = mean - taylor_mean if name != "taylor_rule" else 0.0
        delta_str = f"{delta:+.2f}" if name != "taylor_rule" else "—"
        lines.append(f"{name:<16} {mean:>+9.2f} {std:>8.2f} {delta_str:>10}")
    lines.append("-" * 56)
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------
# COVID BENCHMARK
# -------------------------------------------------------------

def plot_covid_trajectories(
    covid_results: dict[str, dict],
    out_dir: str,
) -> None:
    """Overlay plot: agent trajectory vs real Fed path for each condition."""
    from covid_env import COVID_DATA, _N_REAL
    real_months = list(range(_N_REAL))
    real_cpi    = COVID_DATA["cpi"]
    real_u      = COVID_DATA["unemployment"]
    real_effr   = COVID_DATA["effr"]

    # Use actual trajectory length (may be >36 if episode extended past real data)
    n_steps  = len(covid_results[next(iter(covid_results))]["trajectories"][0]["pi"])
    all_months = list(range(n_steps))

    # phase shading spans
    phase1 = (2, 5)    # lockdown
    phase2 = (14, 34)  # supply chain

    conditions = list(covid_results.keys())
    n = len(conditions)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 10), sharex="col", sharey="row")
    if n == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]

    for col, cond in enumerate(conditions):
        style = _CONDITION_STYLES.get(cond, {"color": "steelblue", "label": cond})
        traj  = covid_results[cond]["trajectories"][0]
        ep_r  = traj["ep_reward"]

        for ax_row in axes:
            ax = ax_row[col]
            ax.axvspan(*phase1, color="gold",   alpha=0.18, label="Lockdown")
            ax.axvspan(*phase2, color="salmon",  alpha=0.18, label="Supply chain")
            if n_steps > _N_REAL:
                ax.axvline(_N_REAL - 1, color="gray", linestyle=":", linewidth=1.2,
                           label="Real data ends")
            ax.grid(True, alpha=0.3)

        # — Inflation —
        ax = axes[0][col]
        ax.plot(real_months, real_cpi, color="red", linestyle="--", linewidth=1.5,
                label="Real CPI")
        ax.plot(all_months, traj["pi"], color="red", linewidth=2, label="Agent sim pi")
        ax.axhline(2.0, color="red", linestyle=":", alpha=0.4)
        ax.set_title(f"{style['label']}\n(reward: {ep_r:+.0f})", fontweight="bold")
        ax.set_ylabel("Inflation (%)")
        ax.legend(loc="upper left", fontsize=7)

        # — Unemployment —
        ax = axes[1][col]
        ax.plot(real_months, real_u,   color="blue", linestyle="--", linewidth=1.5,
                label="Real U-3")
        ax.plot(all_months, traj["u"], color="blue", linewidth=2, label="Agent sim u")
        ax.axhline(4.0, color="blue", linestyle=":", alpha=0.4)
        ax.set_ylabel("Unemployment (%)")
        ax.legend(loc="upper right", fontsize=7)

        # — Policy rate —
        ax = axes[2][col]
        ax.step(real_months, real_effr,    color="green", linestyle="--",
                linewidth=1.5, where="post", label="Real EFFR")
        ax.step(all_months,  traj["rate"], color=style["color"],
                linewidth=2,   where="post", label="Agent rate")
        ax.set_ylabel("Policy Rate (%)")
        ax.set_xlabel("Month (0=Jan 2020)")
        ax.legend(loc="upper left", fontsize=7)

    # x-tick labels: every 6 months, starting Jan 2020
    tick_step = 6
    ticks = list(range(0, n_steps, tick_step))
    def _month_label(t):
        year  = 2020 + (t // 12)
        month = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"][t % 12]
        return f"{month}'{str(year)[2:]}"
    labels = [_month_label(t) for t in ticks]
    for col in range(n):
        axes[2][col].set_xticks(ticks)
        axes[2][col].set_xticklabels(labels, rotation=45, fontsize=7)

    end_year = 2020 + (n_steps // 12)
    fig.suptitle(f"COVID-Era Benchmark  (Jan 2020 - {_month_label(n_steps - 1)})",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "covid_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def _run_covid_benchmark(
    args,
    out_dir: str,
    base_model=None,
    llm_model=None,
    oracle_model=None,
    run_meta: dict | None = None,
) -> None:
    """Evaluate all available conditions on CovidEnv and save table + plots."""
    n = args.covid_runs
    db_path = args.db
    policy  = args.policy

    covid_results: dict[str, dict] = {}

    print(f"\n{'='*56}")
    print(f"COVID-ERA BENCHMARK  (counterfactual, {n} runs each)")
    print(f"{'='*56}")

    print("\n>>> Taylor Rule ...")
    covid_results["taylor_rule"] = _covid_eval(model=None, n_runs=n, policy=policy)
    print(f"    mean={covid_results['taylor_rule']['mean_reward']:+.2f}  "
          f"std={covid_results['taylor_rule']['std_reward']:.2f}")

    if base_model is not None:
        print("\n>>> Baseline PPO ...")
        covid_results["baseline"] = _covid_eval(
            model=base_model,
            env_factory=lambda: CovidEnv(),
            n_runs=n, policy=policy,
        )
        print(f"    mean={covid_results['baseline']['mean_reward']:+.2f}  "
              f"std={covid_results['baseline']['std_reward']:.2f}")

    if llm_model is not None and os.path.exists(db_path):
        print("\n>>> LLM PPO ...")
        _db = db_path
        covid_results["llm"] = _covid_eval(
            model=llm_model,
            env_factory=lambda: StateKeyedLLMWrapper(CovidEnv(), db_path=_db),
            n_runs=n, policy=policy,
        )
        print(f"    mean={covid_results['llm']['mean_reward']:+.2f}  "
              f"std={covid_results['llm']['std_reward']:.2f}")

    if oracle_model is not None:
        print("\n>>> Oracle PPO ...")
        covid_results["oracle"] = _covid_eval(
            model=oracle_model,
            env_factory=lambda: MockLLMObservationWrapper(CovidEnv()),
            n_runs=n, policy=policy,
        )
        print(f"    mean={covid_results['oracle']['mean_reward']:+.2f}  "
              f"std={covid_results['oracle']['std_reward']:.2f}")

    # — Table —
    taylor_mean = covid_results["taylor_rule"]["mean_reward"]
    lines = [
        f"COVID Benchmark — {n} runs each",
        "-" * 56,
        f"{'condition':<16} {'mean':>9} {'std':>8} {'d_taylor':>10}",
        "-" * 56,
    ]
    for name, res in covid_results.items():
        mean, std = res["mean_reward"], res["std_reward"]
        delta = mean - taylor_mean if name != "taylor_rule" else 0.0
        delta_str = f"{delta:+.2f}" if name != "taylor_rule" else "—"
        lines.append(f"{name:<16} {mean:>+9.2f} {std:>8.2f} {delta_str:>10}")
    lines.append("-" * 56)
    table = "\n".join(lines) + "\n"
    print("\n" + table)

    txt_path = os.path.join(out_dir, "covid_benchmark.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved -> {txt_path}")

    # — CSV —
    csv_path = os.path.join(out_dir, "covid_benchmark.csv")
    conds = list(covid_results.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + conds)
        for i in range(n):
            writer.writerow([i] + [round(covid_results[c]["rewards"][i], 4) for c in conds])
    print(f"Saved -> {csv_path}")

    # — Plot —
    plot_covid_trajectories(covid_results, out_dir)

    # — Update metadata —
    if run_meta is not None:
        run_meta.setdefault("covid_benchmark", {}).update({
            "runs": n,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "conditions": {
                name: {"mean": round(res["mean_reward"], 4), "std": round(res["std_reward"], 4)}
                for name, res in covid_results.items()
            },
        })


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark all conditions vs Taylor Rule")
    parser.add_argument("--run",           type=str, default=None,
                        help="Path to run directory (auto-discovers models and metadata)")
    parser.add_argument("--base-model",    type=str, default=None,
                        help="Path to baseline PPO model (without .pt); overrides --run discovery")
    parser.add_argument("--offline-model", type=str, default=None,
                        help="Path to LLM PPO model (without .pt); overrides --run discovery")
    parser.add_argument("--oracle-model",  type=str, default=None,
                        help="Path to oracle PPO model (without .pt); overrides --run discovery")
    parser.add_argument("--db",            type=str, default=config.DEFAULT_STATE_DB_PATH,
                        help=f"State-keyed DB for LLM eval (default {config.DEFAULT_STATE_DB_PATH})")
    parser.add_argument("--out",           type=str, default=None,
                        help="Output directory (default: run dir if --run provided, else runs_custom/)")
    parser.add_argument("--seeds",         type=int, default=config.EVAL_SEEDS,
                        help=f"Number of evaluation seeds (default {config.EVAL_SEEDS})")
    parser.add_argument("--policy",        type=str, default="mlp",
                        choices=["mlp", "lstm"],
                        help="Policy architecture used during training (default: mlp)")
    parser.add_argument("--ema",           action="store_true",
                        help="Use best_model_ema.pt instead of best_model.pt during auto-discovery")
    parser.add_argument("--covid",         action="store_true",
                        help="Also benchmark all conditions on the COVID-era environment (Jan 2020–Dec 2022)")
    parser.add_argument("--covid-runs",    type=int, default=1,
                        help="Number of independent runs per condition for COVID eval (default: 1)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Resolve run dir, model paths, and metadata -------------
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
        # Prefer best_model_ema.pt (--ema) or best_model.pt over final model.pt.
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

            if args.ema and os.path.exists(ema_path + ".pt"):
                setattr(args, arg_attr, ema_path)
                print(f"[auto] {cond_name}: using best_model_ema.pt")
            elif os.path.exists(best_path + ".pt"):
                setattr(args, arg_attr, best_path)
                print(f"[auto] {cond_name}: using best_model.pt")
            elif os.path.exists(final_path + ".pt"):
                setattr(args, arg_attr, final_path)
                print(f"[auto] {cond_name}: using model.pt")

        # Build a map from condition dir to vecnormalize path, co-located with the chosen model.
        # e.g. best_model.pt -> best_model_vecnormalize.pkl; model.pt -> vecnormalize.pkl
        _vecnorm_map: dict[str, str] = {}
        for arg_attr, cond_name in [
            ("base_model",    "baseline"),
            ("offline_model", "llm"),
            ("oracle_model",  "oracle"),
        ]:
            model_path = getattr(args, arg_attr)
            if model_path is None:
                continue
            candidate = model_path + "_vecnormalize.pkl"
            if not os.path.exists(candidate):
                # fall back to the condition-level vecnormalize.pkl
                candidate = os.path.join(run_dir, cond_name, "vecnormalize.pkl")
            _vecnorm_map[arg_attr] = candidate if os.path.exists(candidate) else None

        # Use db_path from metadata if user didn't override
        if run_meta and args.db == config.DEFAULT_STATE_DB_PATH:
            args.db = run_meta.get("db_path", args.db)

    # Read lstm_hidden_size from metadata for backward compat
    lstm_hidden_size = run_meta.get("lstm_hidden_size") if run_meta else None

    out_dir = args.out or run_dir or config.DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    base_model = llm_model = oracle_model = None
    results:          dict[str, list[float]]      = {}
    scenario_seeds  = _find_scenario_seeds()
    print(f"Scenario seeds: {scenario_seeds}")
    traj_by_scenario: dict[str, dict[str, dict]] = {s: {} for s in scenario_seeds}

    # -- Taylor Rule -------------------------------------------
    print(f"\n>>> Taylor Rule  ({args.seeds} seeds) ...")
    results["taylor_rule"] = evaluate_taylor_rule(args.seeds)
    for scen_label, scen_seed in scenario_seeds.items():
        traj_by_scenario[scen_label]["taylor_rule"] = _collect_trajectory_taylor(seed=scen_seed)
    print(f"    mean={np.mean(results['taylor_rule']):+.2f}  "
          f"std={np.std(results['taylor_rule']):.2f}")

    # -- Baseline PPO ------------------------------------------
    if args.base_model:
        print(f"\n>>> Baseline PPO  loading from {args.base_model} ...")
        _base_vecnorm = _vecnorm_map.get("base_model") if run_dir else None
        base_model = _load_model(args.base_model, args.policy, device,
                                 vecnorm_path=_base_vecnorm,
                                 lstm_hidden_size=lstm_hidden_size)
        print(f"    Evaluating on {args.seeds} seeds ...")
        results["baseline"] = evaluate_ppo(
            base_model,
            lambda: FedEnvBase(llm_dim=config.LLM_DIM),
            args.seeds,
            args.policy,
        )
        for scen_label, scen_seed in scenario_seeds.items():
            traj_by_scenario[scen_label]["baseline"] = _collect_trajectory_ppo(
                base_model, lambda: FedEnvBase(llm_dim=config.LLM_DIM),
                seed=scen_seed, policy=args.policy,
            )
        print(f"    mean={np.mean(results['baseline']):+.2f}  "
              f"std={np.std(results['baseline']):.2f}")

    # -- LLM PPO -----------------------------------------------
    if args.offline_model:
        if not os.path.exists(args.db):
            print(f"\nERROR: DB not found at {args.db}")
            print("Run build_state_db.py first, or pass --db <path>")
            sys.exit(1)
        print(f"\n>>> LLM PPO  loading from {args.offline_model} ...")
        _llm_vecnorm = _vecnorm_map.get("offline_model") if run_dir else None
        llm_model = _load_model(args.offline_model, args.policy, device,
                                vecnorm_path=_llm_vecnorm,
                                lstm_hidden_size=lstm_hidden_size)
        print(f"    Evaluating on {args.seeds} seeds ...")
        db_path = args.db
        results["llm"] = evaluate_ppo(
            llm_model,
            lambda: StateKeyedLLMWrapper(FedEnvBase(llm_dim=config.LLM_DIM), db_path=db_path),
            args.seeds,
            args.policy,
        )
        for scen_label, scen_seed in scenario_seeds.items():
            traj_by_scenario[scen_label]["llm"] = _collect_trajectory_ppo(
                llm_model,
                lambda: StateKeyedLLMWrapper(FedEnvBase(llm_dim=config.LLM_DIM), db_path=db_path),
                seed=scen_seed, policy=args.policy,
            )
        print(f"    mean={np.mean(results['llm']):+.2f}  "
              f"std={np.std(results['llm']):.2f}")

    # -- Oracle PPO --------------------------------------------
    if args.oracle_model:
        print(f"\n>>> Oracle PPO  loading from {args.oracle_model} ...")
        _oracle_vecnorm = _vecnorm_map.get("oracle_model") if run_dir else None
        oracle_model = _load_model(args.oracle_model, args.policy, device,
                                   vecnorm_path=_oracle_vecnorm,
                                   lstm_hidden_size=lstm_hidden_size)
        print(f"    Evaluating on {args.seeds} seeds ...")
        results["oracle"] = evaluate_ppo(
            oracle_model,
            lambda: MockLLMObservationWrapper(FedEnvBase(llm_dim=config.LLM_DIM)),
            args.seeds,
            args.policy,
        )
        for scen_label, scen_seed in scenario_seeds.items():
            traj_by_scenario[scen_label]["oracle"] = _collect_trajectory_ppo(
                oracle_model,
                lambda: MockLLMObservationWrapper(FedEnvBase(llm_dim=config.LLM_DIM)),
                seed=scen_seed, policy=args.policy,
            )
        print(f"    mean={np.mean(results['oracle']):+.2f}  "
              f"std={np.std(results['oracle']):.2f}")

    # -- Table -------------------------------------------------
    table = _print_table(results, args.seeds)
    print("\n" + table)

    txt_path = os.path.join(out_dir, "benchmark.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved -> {txt_path}")

    # -- Per-seed CSV ------------------------------------------
    csv_path = os.path.join(out_dir, "benchmark.csv")
    conditions = list(results.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + conditions)
        for seed in range(args.seeds):
            writer.writerow([seed] + [round(results[c][seed], 4) for c in conditions])
    print(f"Saved -> {csv_path}")

    # -- Plots -------------------------------------------------
    if run_dir:
        plot_training_curves(run_dir, out_dir)
    plot_reward_comparison(results, out_dir)
    for scen_label, trajs in traj_by_scenario.items():
        slug = scen_label.lower().replace(" ", "_").replace("-", "_")
        plot_trajectories(
            trajs, out_dir,
            title=f"Representative Episode — {scen_label}  (seed {scenario_seeds[scen_label]})",
            filename=f"benchmark_trajectories_{slug}.png",
        )

    # -- COVID benchmark ---------------------------------------
    if args.covid:
        _run_covid_benchmark(
            args, out_dir,
            base_model=base_model,
            llm_model=llm_model,
            oracle_model=oracle_model,
            run_meta=run_meta,
        )

    # -- Update metadata.json ----------------------------------
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
        print(f"Metadata updated -> {os.path.join(run_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
