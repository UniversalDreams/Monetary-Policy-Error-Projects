"""
src/train.py — PPO training: baseline (no LLM) vs LLM insight (offline DB).

Runs two conditions sequentially:
  1. baseline  — FedEnvBase with zero LLM belief vector
  2. llm       — StateKeyedLLMWrapper with offline belief DB

Usage:
    .venv/Scripts/python src/train.py --condition base --policy lstm --base-episodes 10 --run-name smoke_test
    .venv/Scripts/python src/train.py --condition both --out runs/ --run-name paper_run
"""

import argparse
import collections
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import RecurrentPPO


sys.path.insert(0, os.path.dirname(__file__))

import config
from fed_env import FedEnvBase, StateKeyedLLMWrapper, MockLLMObservationWrapper


# ─────────────────────────────────────────────────────────────
# METADATA HELPERS
# ─────────────────────────────────────────────────────────────

def _pending_condition(name: str) -> dict:
    return {
        "status": "pending",
        "episodes_trained": 0,
        "total_timesteps": 0,
        "final_reward_mean": None,
        "final_reward_std": None,
        "model_path": f"{name}/model.zip",
        "training_csv": f"{name}/training.csv",
        "completed_at": None,
    }


def _save_metadata(run_dir: str, meta: dict) -> None:
    """Atomic JSON write of metadata.json."""
    path = os.path.join(run_dir, "metadata.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp, path)


def _init_run(run_dir: str, args) -> dict:
    """Create run directory structure and write initial metadata.json."""
    for sub in ("baseline/checkpoints", "llm/checkpoints", "oracle/checkpoints"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    meta = {
        "run_id": os.path.basename(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": args.policy,
        "seed": args.seed,
        "base_episodes": args.base_episodes,
        "llm_model": args.model,
        "db_path": args.db,
        "config": {k: v for k, v in vars(config).items() if k.isupper()},
        "conditions": {
            "baseline": _pending_condition("baseline"),
            "llm": _pending_condition("llm"),
            "oracle": _pending_condition("oracle"),
        },
        "benchmark": None,
    }
    _save_metadata(run_dir, meta)
    return meta


# ─────────────────────────────────────────────────────────────
# EPISODE LOGGER CALLBACK
# ─────────────────────────────────────────────────────────────

class EpisodeLogger(BaseCallback):
    """
    Logs one CSV row per completed episode and updates run metadata.
      episode, total_steps, ep_reward, avg_P_supply, avg_hawkishness,
      avg_uncertainty, elapsed_s
    """

    def __init__(self, csv_path: str, tag: str = "",
                 meta: dict = None, condition_key: str = None, run_dir: str = None):
        super().__init__()
        self.csv_path = csv_path
        self.tag = tag
        self._meta = meta
        self._condition_key = condition_key
        self._run_dir = run_dir
        self._ep_reward = 0.0
        self._ep_P_supply: list[float] = []
        self._ep_hawkishness: list[float] = []
        self._ep_uncertainty: list[float] = []
        self._ep_count = 0
        self._ep_step = 0
        self._start_time = time.time()
        self._writer = None
        self._file = None

    def _on_training_start(self) -> None:
        parent = os.path.dirname(self.csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "episode", "total_steps", "ep_reward",
            "avg_P_supply", "avg_hawkishness", "avg_uncertainty",
            "ent_coef", "elapsed_s"
        ])
        self._file.flush()

    def _on_step(self) -> bool:
        self._ep_step += 1

        rewards = self.locals.get("rewards", [0.0])
        self._ep_reward += float(np.mean(rewards))

        obs = self.locals.get("new_obs") or self.locals.get("obs_tensor")
        if obs is not None and isinstance(obs, dict) and "llm_belief" in obs:
            belief = obs["llm_belief"]
            if hasattr(belief, "shape") and belief.shape[-1] >= 5:
                b = np.atleast_2d(belief)[0]
                self._ep_P_supply.append(float(b[1]))
                self._ep_hawkishness.append(float(b[3]))
                self._ep_uncertainty.append(float(b[4]))

        dones = self.locals.get("dones", [False])
        if any(dones):
            self._ep_count += 1
            elapsed = time.time() - self._start_time
            avg_P_supply    = float(np.mean(self._ep_P_supply))    if self._ep_P_supply    else 0.0
            avg_hawkishness = float(np.mean(self._ep_hawkishness)) if self._ep_hawkishness else 0.0
            avg_uncertainty = float(np.mean(self._ep_uncertainty)) if self._ep_uncertainty else 0.0

            ent_coef = float(self.model.ent_coef) if self.model is not None else 0.0

            print(
                f"[{self.tag}] -- ep {self._ep_count} done  "
                f"total_r={self._ep_reward:.2f}  "
                f"avg_P_sup={avg_P_supply:.2f}  "
                f"avg_unc={avg_uncertainty:.2f}  "
                f"ent_coef={ent_coef:.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

            self._writer.writerow([
                self._ep_count,
                self.num_timesteps,
                round(self._ep_reward, 4),
                round(avg_P_supply, 4),
                round(avg_hawkishness, 4),
                round(avg_uncertainty, 4),
                round(ent_coef, 6),
                round(elapsed, 1),
            ])
            self._file.flush()

            # Update metadata
            if self._meta is not None and self._condition_key is not None:
                cond = self._meta["conditions"][self._condition_key]
                cond["episodes_trained"] = self._ep_count
                if self._ep_count % 100 == 0 and self._run_dir:
                    _save_metadata(self._run_dir, self._meta)

            # Reset accumulators
            self._ep_reward = 0.0
            self._ep_step = 0
            self._ep_P_supply.clear()
            self._ep_hawkishness.clear()
            self._ep_uncertainty.clear()

        return True

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()


class EntropyAnnealCallback(BaseCallback):
    """Linearly decays model.ent_coef from ent_coef_start to 0 over training."""
    def __init__(self, ent_coef_start: float, total_timesteps: int):
        super().__init__()
        self.ent_coef_start = ent_coef_start
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        self.model.ent_coef = float(self.ent_coef_start * (1.0 - progress))
        return True


class BestModelCallback(BaseCallback):
    """
    Saves two best-model files per condition:
      - best_model.zip     — best rolling average (uniform window)
      - best_model_ema.zip — best EMA (recent episodes weighted more)
    """
    def __init__(self, save_path: str, tag: str = "", window: int = 100):
        super().__init__()
        self.save_path = save_path          # e.g. cond_dir/best_model
        self.ema_path  = save_path + "_ema" # e.g. cond_dir/best_model_ema
        self.tag = tag
        self.window = window
        self._alpha = 2.0 / (window + 1)   # EMA decay equivalent to `window` episodes
        self._best_avg = -np.inf
        self._best_ema = -np.inf
        self._ep_reward = 0.0
        self._history: collections.deque = collections.deque(maxlen=window)
        self._ema: float | None = None      # None until first episode completes

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        self._ep_reward += float(np.mean(rewards))

        dones = self.locals.get("dones", [False])
        if any(dones):
            r = self._ep_reward

            # Rolling average
            self._history.append(r)
            if len(self._history) == self.window:
                avg = float(np.mean(self._history))
                if avg > self._best_avg:
                    self._best_avg = avg
                    self.model.save(self.save_path)
                    print(f"[{self.tag}] * best avg({self.window}): {avg:.2f} -> {self.save_path}.zip", flush=True)

            # EMA
            self._ema = r if self._ema is None else self._alpha * r + (1 - self._alpha) * self._ema
            if self._ema > self._best_ema:
                self._best_ema = self._ema
                self.model.save(self.ema_path)
                print(f"[{self.tag}] * best ema({self.window}):  {self._ema:.2f} -> {self.ema_path}.zip", flush=True)

            self._ep_reward = 0.0
        return True


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _last100_stats(csv_path: str) -> tuple[float | None, float | None]:
    """Read final 100 episode rewards from training CSV and return (mean, std)."""
    if not os.path.exists(csv_path):
        return None, None
    rewards = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["ep_reward"]))
    last100 = rewards[-100:] if rewards else []
    if not last100:
        return None, None
    return round(float(np.mean(last100)), 4), round(float(np.std(last100)), 4)


_CONDITION_STYLES = {
    "taylor_rule": {"color": "green",      "label": "Taylor Rule"},
    "baseline":    {"color": "purple",     "label": "Baseline PPO"},
    "oracle":      {"color": "steelblue",  "label": "Oracle PPO"},
    "llm":         {"color": "darkorange", "label": "LLM PPO"},
}


def _update_training_curves(run_dir: str, smooth: int = 100) -> None:
    """Read all condition CSVs and redraw training_curves.png (mirrors benchmark.plot_training_curves)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for cond in ("baseline", "oracle", "llm"):
        csv_path = os.path.join(run_dir, cond, "training.csv")
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

        style = _CONDITION_STYLES.get(cond, {"color": "steelblue", "label": cond})
        rewards_arr = np.array(rewards)
        ax.plot(episodes, rewards_arr, color=style["color"], alpha=0.15, linewidth=0.8)
        if len(rewards_arr) >= smooth:
            kernel   = np.ones(smooth) / smooth
            smoothed = np.convolve(rewards_arr, kernel, mode="valid")
            ax.plot(episodes[smooth - 1:], smoothed,
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
    ax.set_title(f"Training Curves  (smoothed over {smooth} episodes)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(run_dir, "training_curves.png")
    tmp = plot_path + ".tmp.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, plot_path)


class LivePlotCallback(BaseCallback):
    """Redraws training_curves.png after every rollout (one PPO update cycle)."""
    def __init__(self, run_dir: str, smooth: int = 100):
        super().__init__()
        self.run_dir = run_dir
        self.smooth  = smooth

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        _update_training_curves(self.run_dir, self.smooth)

    def _on_training_end(self) -> None:
        _update_training_curves(self.run_dir, self.smooth)


def _tb_log(out: str, name: str):
    """Return tensorboard log path only if tensorboard is installed."""
    try:
        import tensorboard  # noqa: F401
        return os.path.join(out, "tb", name)
    except ImportError:
        return None


def make_ppo(env, seed: int, out: str, name: str, policy: str = "mlp", condition: str = "baseline"):
    if policy == "lstm":
        cond_key = condition.upper()
        lr_start    = getattr(config, f"{cond_key}_LR")
        lr_end      = getattr(config, f"{cond_key}_LR_END")
        decay_start = getattr(config, f"{cond_key}_LR_DECAY_START")
        ent_coef    = getattr(config, f"{cond_key}_ENT_COEF")
        clip_range  = getattr(config, f"{cond_key}_CLIP_RANGE")
        return RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            learning_rate=lambda p: (
                lr_start if p > decay_start
                else lr_end + (lr_start - lr_end) * (p / decay_start)
            ),
            n_steps=config.LSTM_N_STEPS,
            batch_size=config.LSTM_BATCH_SIZE,
            n_epochs=config.LSTM_N_EPOCHS,
            gamma=config.GAMMA,
            ent_coef=ent_coef,
            clip_range=clip_range,
            verbose=0,
            seed=seed,
            device="cuda",
            tensorboard_log=_tb_log(out, name),
        )
    return PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config.LR,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        verbose=0,
        seed=seed,
        device="cpu",
        tensorboard_log=_tb_log(out, name),
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PPO training: baseline vs LLM insight")
    parser.add_argument("--base-episodes", type=int, default=config.DEFAULT_BASE_EPISODES,
                        help=f"Episodes for baseline/llm conditions (default {config.DEFAULT_BASE_EPISODES})")
    parser.add_argument("--model",         type=str, default=config.DEFAULT_MODEL,
                        help=f"Ollama model name (default {config.DEFAULT_MODEL})")
    parser.add_argument("--out",           type=str, default=config.DEFAULT_OUT,
                        help="Runs root directory (default: runs/)")
    parser.add_argument("--run-name",      type=str, default=None,
                        help="Run identifier (default: YYYYMMDD_HHMMSS_{policy})")
    parser.add_argument("--seed",          type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--condition",     type=str, default="both",
                        choices=["base", "offline", "oracle", "both", "all"],
                        help="Which condition(s) to run (default: both; "
                             "'both'=base+offline, 'all'=base+offline+oracle)")
    parser.add_argument("--policy",        type=str, default="mlp",
                        choices=["mlp", "lstm"],
                        help="Policy architecture: mlp (default) or lstm")
    parser.add_argument("--db",            type=str, default=config.DEFAULT_STATE_DB_PATH,
                        help=f"State-keyed DB path for LLM condition "
                             f"(default {config.DEFAULT_STATE_DB_PATH})")
    args = parser.parse_args()

    run_id = args.run_name or (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.policy}")
    run_dir = os.path.join(args.out, run_id)

    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "oracle" not in meta["conditions"]:
            meta["conditions"]["oracle"] = _pending_condition("oracle")
        os.makedirs(os.path.join(run_dir, "oracle", "checkpoints"), exist_ok=True)
        _save_metadata(run_dir, meta)
    else:
        meta = _init_run(run_dir, args)

    base_total_timesteps = args.base_episodes * config.MAX_STEPS

    print(f"\n{'='*60}")
    print(f"  Run: {run_id}")
    print(f"  policy={args.policy}  seed={args.seed}  condition={args.condition}")
    print(f"  base_episodes={args.base_episodes}  run_dir={run_dir}")
    print(f"{'='*60}\n")

    # ── Condition: Baseline (zero belief) ─────────────────────
    if args.condition in ("base", "both", "all"):
        print(">>> Condition: Baseline (zero LLM belief)")
        cond_dir = os.path.join(run_dir, "baseline")
        meta["conditions"]["baseline"]["status"] = "running"
        _save_metadata(run_dir, meta)

        base_env = DummyVecEnv([lambda: FedEnvBase(llm_dim=config.LLM_DIM) for _ in range(config.N_ENVS)])
        base_model = make_ppo(base_env, args.seed, cond_dir, "baseline", args.policy, condition="baseline")
        base_model.learn(
            total_timesteps=base_total_timesteps,
            callback=[
                CheckpointCallback(
                    save_freq=config.CHECKPOINT_FREQ // config.N_ENVS,
                    save_path=os.path.join(cond_dir, "checkpoints"),
                    name_prefix="model",
                ),
                EpisodeLogger(
                    os.path.join(cond_dir, "training.csv"),
                    tag="baseline",
                    meta=meta,
                    condition_key="baseline",
                    run_dir=run_dir,
                ),
                EntropyAnnealCallback(config.BASELINE_ENT_COEF, base_total_timesteps),
                BestModelCallback(os.path.join(cond_dir, "best_model"), tag="baseline"),
                LivePlotCallback(run_dir),
            ],
        )
        base_model.save(os.path.join(cond_dir, "model"))

        mean, std = _last100_stats(os.path.join(cond_dir, "training.csv"))
        meta["conditions"]["baseline"].update({
            "status": "completed",
            "total_timesteps": base_total_timesteps,
            "final_reward_mean": mean,
            "final_reward_std":  std,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        _save_metadata(run_dir, meta)
        base_env.close()
        print(f">>> Baseline model saved to {cond_dir}/model.zip\n")

    # ── Condition: Oracle (perfect hardcoded belief) ──────────
    if args.condition in ("oracle", "all"):
        print(">>> Condition: Oracle (MockLLMObservationWrapper — perfect belief)")
        cond_dir = os.path.join(run_dir, "oracle")
        meta["conditions"]["oracle"]["status"] = "running"
        _save_metadata(run_dir, meta)

        oracle_env = DummyVecEnv([lambda: MockLLMObservationWrapper(FedEnvBase(llm_dim=config.LLM_DIM)) for _ in range(config.N_ENVS)])
        oracle_model = make_ppo(oracle_env, args.seed, cond_dir, "oracle", args.policy, condition="oracle")
        oracle_model.learn(
            total_timesteps=base_total_timesteps,
            callback=[
                CheckpointCallback(
                    save_freq=config.CHECKPOINT_FREQ // config.N_ENVS,
                    save_path=os.path.join(cond_dir, "checkpoints"),
                    name_prefix="model",
                ),
                EpisodeLogger(
                    os.path.join(cond_dir, "training.csv"),
                    tag="oracle",
                    meta=meta,
                    condition_key="oracle",
                    run_dir=run_dir,
                ),
                EntropyAnnealCallback(config.ORACLE_ENT_COEF, base_total_timesteps),
                BestModelCallback(os.path.join(cond_dir, "best_model"), tag="oracle"),
                LivePlotCallback(run_dir),
            ],
        )
        oracle_model.save(os.path.join(cond_dir, "model"))

        mean, std = _last100_stats(os.path.join(cond_dir, "training.csv"))
        meta["conditions"]["oracle"].update({
            "status": "completed",
            "total_timesteps": base_total_timesteps,
            "final_reward_mean": mean,
            "final_reward_std":  std,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        _save_metadata(run_dir, meta)
        oracle_env.close()
        print(f">>> Oracle model saved to {cond_dir}/model.zip\n")

    # ── Condition: LLM insight (state-keyed DB) ───────────────
    if args.condition in ("offline", "both", "all"):
        print(">>> Condition: LLM insight (state-keyed belief DB)")
        if not os.path.exists(args.db):
            print(f"ERROR: DB not found at {args.db}")
            print("Run: python src/build_state_db.py first")
            sys.exit(1)
        cond_dir = os.path.join(run_dir, "llm")
        meta["conditions"]["llm"]["status"] = "running"
        _save_metadata(run_dir, meta)

        _db = args.db
        llm_env = DummyVecEnv([lambda: StateKeyedLLMWrapper(FedEnvBase(llm_dim=config.LLM_DIM), db_path=_db) for _ in range(config.N_ENVS)])
        llm_model = make_ppo(llm_env, args.seed, cond_dir, "llm", args.policy, condition="llm")
        llm_model.learn(
            total_timesteps=base_total_timesteps,
            callback=[
                CheckpointCallback(
                    save_freq=config.CHECKPOINT_FREQ // config.N_ENVS,
                    save_path=os.path.join(cond_dir, "checkpoints"),
                    name_prefix="model",
                ),
                EpisodeLogger(
                    os.path.join(cond_dir, "training.csv"),
                    tag="llm",
                    meta=meta,
                    condition_key="llm",
                    run_dir=run_dir,
                ),
                EntropyAnnealCallback(config.LLM_ENT_COEF, base_total_timesteps),
                BestModelCallback(os.path.join(cond_dir, "best_model"), tag="llm"),
                LivePlotCallback(run_dir),
            ],
        )
        llm_model.save(os.path.join(cond_dir, "model"))

        mean, std = _last100_stats(os.path.join(cond_dir, "training.csv"))
        total_misses = sum(e._misses for e in llm_env.envs)
        miss_rate = (total_misses / max(1, base_total_timesteps * config.N_ENVS)) * 100
        meta["conditions"]["llm"].update({
            "status": "completed",
            "total_timesteps": base_total_timesteps,
            "final_reward_mean": mean,
            "final_reward_std":  std,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        _save_metadata(run_dir, meta)
        llm_env.close()
        print(f">>> LLM model saved to {cond_dir}/model.zip")
        print(f">>> Miss rate: {total_misses} misses = {miss_rate:.1f}%\n")

    print(f"\nRun complete -> {run_dir}")
    print(f"Next: python src/benchmark.py --run {run_dir} --db {args.db}")


if __name__ == "__main__":
    main()
