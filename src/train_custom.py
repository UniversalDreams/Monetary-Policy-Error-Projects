"""
src/train_custom.py — from-scratch PPO training (MLP or LSTM) without SB3.

Both policies use the same agent.learn(total_timesteps) call — no branching.

Usage:
    .venv/Scripts/python src/train_custom.py --policy mlp  --base-episodes 5  --run-name smoke_mlp
    .venv/Scripts/python src/train_custom.py --policy lstm --base-episodes 5  --run-name smoke_lstm
    .venv/Scripts/python src/train_custom.py --condition both --base-episodes 500 --run-name paper_run
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(__file__))

import config
from fed_env import FedEnvBase, StateKeyedLLMWrapper, MockLLMObservationWrapper
from network import ActorCritic, LSTMActorCritic
from ppo import PPOAgent
from ppo_recurrent import RecurrentPPOAgent


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_agent(policy: str, envs, device: str):
    """Build actor-critic network + agent for the given policy type."""
    macro_dim = envs.observation_space["macro"].shape[0]
    llm_dim   = envs.observation_space["llm_belief"].shape[0]
    obs_dim   = macro_dim + llm_dim
    act_dim   = envs.action_space.n

    if policy == "lstm":
        ac = LSTMActorCritic(obs_dim, act_dim, lstm_hidden_size=64, n_lstm_layers=1).to(device)
        return RecurrentPPOAgent(envs, ac, device), ac
    else:
        ac = ActorCritic(obs_dim, act_dim).to(device)
        return PPOAgent(envs, ac, device), ac


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

    # ── Baseline ──────────────────────────────────────────────
    if args.condition in ("base", "both", "all"):
        print(">>> Condition: Baseline (zero LLM belief)")
        cond_dir = os.path.join(run_dir, "baseline")
        meta["conditions"]["baseline"]["status"] = "running"
        _save_metadata(run_dir, meta)

        envs = DummyVecEnv([lambda: FedEnvBase(llm_dim=config.LLM_DIM)
                            for _ in range(config.N_ENVS)])

        agent, ac = _make_agent(args.policy, envs, device)
        agent.learn(
            total_timesteps,
            csv_log_path=os.path.join(cond_dir, "training.csv"),
            tag="baseline",
        )

        torch.save(ac.state_dict(), os.path.join(cond_dir, "model.pt"))
        meta["conditions"]["baseline"].update({
            "status": "completed",
            "total_timesteps": total_timesteps,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        _save_metadata(run_dir, meta)
        envs.close()
        print(f">>> Baseline model saved to {cond_dir}/model.pt\n")

    # ── Oracle ────────────────────────────────────────────────
    if args.condition in ("oracle", "all"):
        print(">>> Condition: Oracle (perfect belief via MockLLMObservationWrapper)")
        cond_dir = os.path.join(run_dir, "oracle")
        meta["conditions"]["oracle"]["status"] = "running"
        _save_metadata(run_dir, meta)

        envs = DummyVecEnv([
            lambda: MockLLMObservationWrapper(FedEnvBase(llm_dim=config.LLM_DIM))
            for _ in range(config.N_ENVS)
        ])

        agent, ac = _make_agent(args.policy, envs, device)
        agent.learn(
            total_timesteps,
            csv_log_path=os.path.join(cond_dir, "training.csv"),
            tag="oracle",
        )

        torch.save(ac.state_dict(), os.path.join(cond_dir, "model.pt"))
        meta["conditions"]["oracle"].update({
            "status": "completed",
            "total_timesteps": total_timesteps,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        _save_metadata(run_dir, meta)
        envs.close()
        print(f">>> Oracle model saved to {cond_dir}/model.pt\n")

    # ── LLM offline ───────────────────────────────────────────
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
        envs = DummyVecEnv([
            lambda: StateKeyedLLMWrapper(FedEnvBase(llm_dim=config.LLM_DIM), db_path=_db)
            for _ in range(config.N_ENVS)
        ])

        agent, ac = _make_agent(args.policy, envs, device)
        agent.learn(
            total_timesteps,
            csv_log_path=os.path.join(cond_dir, "training.csv"),
            tag="llm",
        )

        torch.save(ac.state_dict(), os.path.join(cond_dir, "model.pt"))
        meta["conditions"]["llm"].update({
            "status": "completed",
            "total_timesteps": total_timesteps,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        _save_metadata(run_dir, meta)
        envs.close()
        print(f">>> LLM model saved to {cond_dir}/model.pt\n")

    print(f"\nRun complete -> {run_dir}")


if __name__ == "__main__":
    main()
