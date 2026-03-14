"""
src/train_custom.py — from-scratch PPO training (MLP or LSTM) without SB3.

Both policies use the same agent.learn(total_timesteps) call — no branching.

Usage:
    .venv/Scripts/python src/train_custom.py --policy mlp  --base-episodes 5  --run-name smoke_mlp
    .venv/Scripts/python src/train_custom.py --policy lstm --base-episodes 5  --run-name smoke_lstm
    .venv/Scripts/python src/train_custom.py --condition both --base-episodes 500 --run-name paper_run
"""

import argparse
import collections
import csv
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for background saving
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(__file__))

import config
from fed_env import FedEnvBase, StateKeyedLLMWrapper, MockLLMObservationWrapper
from network import ActorCritic, LSTMActorCritic
from ppo import PPOAgent
from ppo_recurrent import RecurrentPPOAgent


# ─────────────────────────────────────────────────────────────
# LIVE TRAINING CURVE PLOT
# ─────────────────────────────────────────────────────────────

_CONDITION_STYLES = {
    "baseline": {"color": "purple",     "label": "Baseline PPO"},
    "oracle":   {"color": "steelblue",  "label": "Oracle PPO"},
    "llm":      {"color": "darkorange", "label": "LLM PPO"},
}


def _plot_training_curves(run_dir: str, window: int = 100) -> None:
    """Regenerate training_curves.png in run_dir from all available training.csv files."""
    cond_csvs = {
        cond: os.path.join(run_dir, cond, "training.csv")
        for cond in ("baseline", "oracle", "llm")
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for cond, csv_path in cond_csvs.items():
        if not os.path.exists(csv_path):
            continue
        episodes, rewards = [], []
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    episodes.append(int(row["episode"]))
                    rewards.append(float(row["ep_reward"]))
        except Exception:
            continue
        if not rewards:
            continue

        style = _CONDITION_STYLES[cond]
        rewards_arr = np.array(rewards)
        ax.plot(episodes, rewards_arr, color=style["color"], alpha=0.15, linewidth=0.8)
        if len(rewards_arr) >= window:
            kernel   = np.ones(window) / window
            smoothed = np.convolve(rewards_arr, kernel, mode="valid")
            ax.plot(episodes[window - 1:], smoothed,
                    color=style["color"], linewidth=2, label=style["label"])
        else:
            ax.plot(episodes, rewards_arr,
                    color=style["color"], linewidth=2, label=style["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title(f"Training Curves  (smoothed over {window} episodes)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_curves(run_dir: str, window: int = 100) -> None:
    """Regenerate loss_curves.png from all available losses.csv files."""
    cond_csvs = {
        cond: os.path.join(run_dir, cond, "losses.csv")
        for cond in ("baseline", "oracle", "llm")
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plotted = False

    for cond, csv_path in cond_csvs.items():
        if not os.path.exists(csv_path):
            continue
        steps, p_losses, v_losses = [], [], []
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    steps.append(int(row["global_step"]))
                    p_losses.append(float(row["policy_loss"]))
                    v_losses.append(float(row["value_loss"]))
        except Exception:
            continue
        if not steps:
            continue

        style = _CONDITION_STYLES[cond]
        for ax, vals, title in zip(
            axes,
            [p_losses, v_losses],
            ["Policy Loss", "Value Loss"],
        ):
            arr = np.array(vals)
            ax.plot(steps, arr, color=style["color"], alpha=0.15, linewidth=0.8)
            if len(arr) >= window:
                kernel = np.ones(window) / window
                smoothed = np.convolve(arr, kernel, mode="valid")
                ax.plot(steps[window - 1:], smoothed,
                        color=style["color"], linewidth=2, label=style["label"])
            else:
                ax.plot(steps, arr, color=style["color"], linewidth=2,
                        label=style["label"])
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Global Step", fontsize=10)
            ax.grid(True, alpha=0.3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    fig.suptitle(f"Loss Curves  (smoothed over {window} updates)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_agent(policy: str, envs, device: str,
                lr_start=config.LR, lr_end=config.LR, lr_decay_start=1.0,
                clip_range=config.BASELINE_CLIP_RANGE,
                ent_coef_start=config.BASELINE_ENT_COEF,
                ent_coef_end=0.0,
                vf_coef=0.5):
    """Build actor-critic network + agent for the given policy type."""
    macro_dim = envs.observation_space["macro"].shape[0]
    llm_dim   = envs.observation_space["llm_belief"].shape[0]
    obs_dim   = macro_dim + llm_dim
    act_dim   = envs.action_space.n

    if policy == "lstm":
        ac = LSTMActorCritic(obs_dim, act_dim, lstm_hidden_size=config.LSTM_HIDDEN_SIZE, n_lstm_layers=1).to(device)
        return RecurrentPPOAgent(envs, ac, device, lr_start, lr_end, lr_decay_start,
                                 clip_range=clip_range,
                                 ent_coef_start=ent_coef_start,
                                 ent_coef_end=ent_coef_end,
                                 vf_coef=vf_coef), ac
    else:
        ac = ActorCritic(obs_dim, act_dim).to(device)
        return PPOAgent(envs, ac, device, lr_start, lr_end, lr_decay_start,
                        clip_range=clip_range,
                        ent_coef_start=ent_coef_start,
                        ent_coef_end=ent_coef_end), ac


def _pending_condition(name: str) -> dict:
    return {
        "status": "pending",
        "episodes_trained": 0,
        "total_timesteps": 0,
        "final_reward_mean": None,
        "final_reward_std": None,
        "model_path": f"{name}/model.pt",
        "training_csv": f"{name}/training.csv",
        "completed_at": None,
    }


def _save_metadata(run_dir: str, meta: dict) -> None:
    path = os.path.join(run_dir, "metadata.json")
    tmp  = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp, path)


def _init_run(run_dir: str, args) -> dict:
    for sub in ("baseline/checkpoints", "llm/checkpoints", "oracle/checkpoints"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    meta = {
        "run_id":        os.path.basename(run_dir),
        "created_at":    datetime.now().isoformat(timespec="seconds"),
        "policy":        args.policy,
        "seed":          args.seed,
        "lstm_hidden_size": config.LSTM_HIDDEN_SIZE,
        "base_episodes": args.base_episodes,
        "db_path":       args.db,
        "config":        {k: v for k, v in vars(config).items() if k.isupper()},
        "conditions": {
            "baseline": _pending_condition("baseline"),
            "llm":      _pending_condition("llm"),
            "oracle":   _pending_condition("oracle"),
        },
    }
    _save_metadata(run_dir, meta)
    return meta


def _last100_stats(csv_path):
    """Return (mean, std) of ep_reward over the last 100 rows in csv_path."""
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        rewards = [float(r["ep_reward"]) for r in rows[-100:]]
        if not rewards:
            return None, None
        return float(np.mean(rewards)), float(np.std(rewards))
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────
# BEST-MODEL TRACKER
# ─────────────────────────────────────────────────────────────

class BestModelTracker:
    """
    Saves best_model.pt (rolling avg) and best_model_ema.pt (EMA) whenever
    a new best is reached.
    """

    def __init__(self, save_path: str, tag: str, window: int = 100, envs=None):
        self._save_path = save_path  # no extension; we add .pt / _ema.pt
        self._tag = tag
        self._envs = envs
        self._alpha = 2.0 / (window + 1)
        self._history = collections.deque(maxlen=window)
        self._ema = None
        self._best_avg = -np.inf
        self._best_ema = -np.inf

    def _save_vecnormalize(self, suffix: str) -> None:
        if isinstance(self._envs, VecNormalize):
            self._envs.save(self._save_path + suffix + "_vecnormalize.pkl")

    def update(self, ep_reward: float, ac: nn.Module) -> None:
        self._history.append(ep_reward)

        # Rolling average — only save when window is full
        if len(self._history) == self._history.maxlen:
            avg = float(np.mean(self._history))
            if avg > self._best_avg:
                self._best_avg = avg
                torch.save(ac.state_dict(), self._save_path + ".pt")
                self._save_vecnormalize("")
                print(f"[{self._tag}] new best avg={avg:.2f} -> best_model.pt", flush=True)

        # EMA
        if self._ema is None:
            self._ema = ep_reward
        else:
            self._ema = self._alpha * ep_reward + (1.0 - self._alpha) * self._ema

        if self._ema > self._best_ema:
            self._best_ema = self._ema
            torch.save(ac.state_dict(), self._save_path + "_ema.pt")
            self._save_vecnormalize("_ema")
            print(f"[{self._tag}] new best EMA={self._ema:.2f} -> best_model_ema.pt", flush=True)


# ─────────────────────────────────────────────────────────────
# EPISODE LOGGER
# ─────────────────────────────────────────────────────────────

class EpisodeLogger:
    """
    Writes per-episode CSV rows and keeps meta["conditions"][condition_key]["episodes_trained"]
    up to date.
    """

    _HEADER = [
        "episode", "total_steps", "ep_reward",
        "avg_P_supply", "avg_hawkishness", "avg_uncertainty",
        "ent_coef", "elapsed_s",
    ]

    def __init__(self, csv_path: str, tag: str,
                 meta: dict, condition_key: str, run_dir: str):
        self._tag = tag
        self._meta = meta
        self._condition_key = condition_key
        self._run_dir = run_dir
        self._ep_count = 0
        self._start_time = time.time()

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._csv_file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(self._HEADER)

    def on_episode(self, ep_reward: float, ep_norm_reward: float, global_step: int,
                   ent_coef: float, llm_belief) -> None:
        self._ep_count += 1
        elapsed = time.time() - self._start_time

        if llm_belief is not None and len(llm_belief) > 0 and llm_belief.ndim == 2:
            avg_p_supply    = float(np.mean(llm_belief[:, 1]))
            avg_hawkishness = float(np.mean(llm_belief[:, 3]))
            avg_uncertainty = float(np.mean(llm_belief[:, 4]))
        else:
            avg_p_supply = avg_hawkishness = avg_uncertainty = None

        self._writer.writerow([
            self._ep_count,
            global_step,
            round(ep_reward, 4),
            round(avg_p_supply,    4) if avg_p_supply    is not None else "",
            round(avg_hawkishness, 4) if avg_hawkishness is not None else "",
            round(avg_uncertainty, 4) if avg_uncertainty is not None else "",
            round(ent_coef, 6),
            round(elapsed, 1),
        ])
        self._csv_file.flush()

        print(
            f"[{self._tag}] ep {self._ep_count}  raw_reward={ep_reward:.2f}"
            f"  norm_reward={ep_norm_reward:.4f}"
            f"  steps={global_step}  elapsed={elapsed:.0f}s",
            flush=True,
        )

        self._meta["conditions"][self._condition_key]["episodes_trained"] = self._ep_count
        if self._ep_count % 100 == 0:
            _save_metadata(self._run_dir, self._meta)
            _plot_training_curves(self._run_dir)

    def close(self) -> None:
        self._csv_file.close()


# ─────────────────────────────────────────────────────────────
# LOSS LOGGER
# ─────────────────────────────────────────────────────────────

class LossLogger:
    """Writes one CSV row per PPO update (one call to _update_policy())."""

    _HEADER = ["update", "global_step", "policy_loss", "value_loss", "entropy_loss", "explained_variance"]

    def __init__(self, csv_path: str):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._csv_file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(self._HEADER)
        self._update_count = 0

    def on_update(self, losses: dict, global_step: int) -> None:
        self._update_count += 1
        expl_var = losses.get("explained_variance", "")
        self._writer.writerow([
            self._update_count,
            global_step,
            round(losses["policy_loss"],  6),
            round(losses["value_loss"],   6),
            round(losses["entropy_loss"], 6),
            round(expl_var, 6) if expl_var != "" else "",
        ])
        self._csv_file.flush()

    def close(self) -> None:
        self._csv_file.close()


# ─────────────────────────────────────────────────────────────
# CONDITION RUNNER
# ─────────────────────────────────────────────────────────────

_LR_CONFIG = {
    "baseline": (config.BASELINE_LR, config.BASELINE_LR_END, config.BASELINE_LR_DECAY_START),
    "oracle":   (config.ORACLE_LR,   config.ORACLE_LR_END,   config.ORACLE_LR_DECAY_START),
    "llm":      (config.LLM_LR,      config.LLM_LR_END,      config.LLM_LR_DECAY_START),
}

_COND_CONFIG = {
    "baseline": (config.BASELINE_CLIP_RANGE, config.BASELINE_ENT_COEF, config.BASELINE_ENT_COEF_END, config.BASELINE_VF_COEF),
    "oracle":   (config.ORACLE_CLIP_RANGE,   config.ORACLE_ENT_COEF,   config.ORACLE_ENT_COEF_END,   config.ORACLE_VF_COEF),
    "llm":      (config.LLM_CLIP_RANGE,      config.LLM_ENT_COEF,      config.LLM_ENT_COEF_END,      config.LLM_VF_COEF),
}


def _run_condition(condition_key, envs, policy, device,
                   total_timesteps, cond_dir, run_dir, meta):
    """Build agent, wire callbacks, train, save final model + metadata."""
    lr_start, lr_end, lr_decay_start = _LR_CONFIG.get(condition_key, (config.LR, config.LR, 1.0))
    clip_range, ent_coef_start, ent_coef_end, vf_coef = _COND_CONFIG.get(
        condition_key, (config.BASELINE_CLIP_RANGE, config.BASELINE_ENT_COEF, 0.0, 0.5))
    agent, ac = _make_agent(policy, envs, device, lr_start, lr_end, lr_decay_start,
                            clip_range=clip_range,
                            ent_coef_start=ent_coef_start,
                            ent_coef_end=ent_coef_end,
                            vf_coef=vf_coef)
    os.makedirs(os.path.join(cond_dir, "checkpoints"), exist_ok=True)

    logger = EpisodeLogger(
        os.path.join(cond_dir, "training.csv"),
        tag=condition_key,
        meta=meta,
        condition_key=condition_key,
        run_dir=run_dir,
    )
    loss_logger = LossLogger(os.path.join(cond_dir, "losses.csv"))
    tracker = BestModelTracker(
        os.path.join(cond_dir, "best_model"),
        tag=condition_key,
        window=500,
        envs=envs,
    )

    def on_episode(ep_reward, ep_norm_reward, global_step, ent_coef, llm_belief):
        logger.on_episode(ep_reward, ep_norm_reward, global_step, ent_coef, llm_belief)
        tracker.update(ep_reward, ac)

    def on_update(losses, global_step):
        loss_logger.on_update(losses, global_step)
        _plot_loss_curves(run_dir)

    def on_checkpoint(ac_, step):
        path = os.path.join(cond_dir, "checkpoints", f"model_{step}.pt")
        torch.save(ac_.state_dict(), path)
        if isinstance(envs, VecNormalize):
            envs.save(os.path.join(cond_dir, "checkpoints", f"vecnormalize_{step}.pkl"))
        print(f"[{condition_key}] checkpoint -> {path}", flush=True)

    try:
        agent.learn(total_timesteps, on_episode=on_episode,
                    on_checkpoint=on_checkpoint, on_update=on_update)
    finally:
        logger.close()
        loss_logger.close()

    if isinstance(envs, VecNormalize):
        envs.save(os.path.join(cond_dir, "vecnormalize.pkl"))
    torch.save(ac.state_dict(), os.path.join(cond_dir, "model.pt"))
    mean, std = _last100_stats(os.path.join(cond_dir, "training.csv"))
    meta["conditions"][condition_key].update({
        "status": "completed",
        "total_timesteps": total_timesteps,
        "final_reward_mean": mean,
        "final_reward_std": std,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    })
    _save_metadata(run_dir, meta)
    envs.close()
    print(f">>> {condition_key} model saved to {cond_dir}/model.pt\n")
    return ac


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="From-scratch PPO training (MLP or LSTM)")
    parser.add_argument("--base-episodes", type=int,  default=config.DEFAULT_BASE_EPISODES)
    parser.add_argument("--out",           type=str,  default=config.DEFAULT_OUT)
    parser.add_argument("--run-name",      type=str,  default=None)
    parser.add_argument("--seed",          type=int,  default=config.DEFAULT_SEED)
    parser.add_argument("--condition",     type=str,  default="both",
                        choices=["base", "offline", "oracle", "both", "all"])
    parser.add_argument("--policy",        type=str,  default="mlp",
                        choices=["mlp", "lstm"])
    parser.add_argument("--db",            type=str,  default=config.DEFAULT_STATE_DB_PATH)
    args = parser.parse_args()

    device = config.PPO_DEVICE if torch.cuda.is_available() else "cpu"

    run_id  = args.run_name or (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.policy}")
    run_dir = os.path.join(args.out, run_id)
    meta    = _init_run(run_dir, args)

    total_timesteps = args.base_episodes * config.MAX_STEPS

    print(f"\n{'='*60}")
    print(f"  Run: {run_id}")
    print(f"  policy={args.policy}  device={device}  seed={args.seed}")
    print(f"  condition={args.condition}  base_episodes={args.base_episodes}")
    print(f"  run_dir={run_dir}")
    print(f"{'='*60}\n")

    # ── LLM offline ───────────────────────────────────────────
    if args.condition in ("offline", "both", "all"):
        print(">>> Condition: LLM insight (state-keyed belief DB)")
        if not os.path.exists(args.db):
            print(f"ERROR: DB not found at {args.db}")
            print("Run: python src/build_state_db.py first")
            sys.exit(1)
        meta["conditions"]["llm"]["status"] = "running"
        _save_metadata(run_dir, meta)
        _db = args.db
        envs = DummyVecEnv([
            lambda: StateKeyedLLMWrapper(FedEnvBase(llm_dim=config.LLM_DIM), db_path=_db)
            for _ in range(config.N_ENVS)
        ])
        envs = VecNormalize(envs, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=config.GAMMA)
        _run_condition("llm", envs, args.policy, device,
                       total_timesteps, os.path.join(run_dir, "llm"), run_dir, meta)

    # ── Baseline ──────────────────────────────────────────────
    if args.condition in ("base", "both", "all"):
        print(">>> Condition: Baseline (zero LLM belief)")
        meta["conditions"]["baseline"]["status"] = "running"
        _save_metadata(run_dir, meta)
        envs = DummyVecEnv([lambda: FedEnvBase(llm_dim=config.LLM_DIM)
                            for _ in range(config.N_ENVS)])
        envs = VecNormalize(envs, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=config.GAMMA)
        _run_condition("baseline", envs, args.policy, device,
                       total_timesteps, os.path.join(run_dir, "baseline"), run_dir, meta)

    # ── Oracle ────────────────────────────────────────────────
    if args.condition in ("oracle", "all"):
        print(">>> Condition: Oracle (perfect belief via MockLLMObservationWrapper)")
        meta["conditions"]["oracle"]["status"] = "running"
        _save_metadata(run_dir, meta)
        envs = DummyVecEnv([
            lambda: MockLLMObservationWrapper(FedEnvBase(llm_dim=config.LLM_DIM))
            for _ in range(config.N_ENVS)
        ])
        envs = VecNormalize(envs, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=config.GAMMA)
        _run_condition("oracle", envs, args.policy, device,
                       total_timesteps, os.path.join(run_dir, "oracle"), run_dir, meta)

    print(f"\nRun complete -> {run_dir}")


if __name__ == "__main__":
    main()
