import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

import config
from fed_env import FedEnvBase, StateKeyedLLMWrapper, MockLLMObservationWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from ppo import PPOAgent
from network import ActorCritic
from train import _init_run, _save_metadata


def make_manual_ppo(envs, device):
    """Instantiates custom ActorCritic and PPOAgent."""
    macro_dim = envs.observation_space["macro"].shape[0]
    llm_dim = envs.observation_space["llm_belief"].shape[0]
    obs_dim = macro_dim + llm_dim
    act_dim = envs.action_space.n

    actor_critic = ActorCritic(obs_dim, act_dim).to(device)
    agent = PPOAgent(envs, actor_critic, device)

    return agent, actor_critic


def main():
    parser = argparse.ArgumentParser(description="Manual PPO training: baseline vs LLM vs Oracle")
    parser.add_argument("--base-episodes", type=int, default=config.DEFAULT_BASE_EPISODES)
    parser.add_argument("--out", type=str, default="manual_ppo_results")
    parser.add_argument("--condition", type=str, default="all", choices=["base", "offline", "oracle", "both", "all"])
    parser.add_argument("--db", type=str, default=config.DEFAULT_STATE_DB_PATH)
    args = parser.parse_args()

    run_dir = args.out

    args.policy = "mlp"
    args.model = config.DEFAULT_MODEL
    args.seed = config.DEFAULT_SEED

    # load existing metadata
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        # Ensure oracle key exists in older metadata
        if "oracle" not in meta["conditions"]:
            meta["conditions"]["oracle"] = {"status": "pending"}
    else:
        meta = _init_run(run_dir, args)
        if "oracle" not in meta["conditions"]:
            meta["conditions"]["oracle"] = {"status": "pending"}

    base_total_timesteps = args.base_episodes * config.MAX_STEPS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  PPO RUN  (Device: {device})")
    print(f"  Target Directory: {run_dir}")
    print(f"{'=' * 60}\n")

    # Baseline (zero belief)
    if args.condition in ("base", "both"):
        print("Condition: Baseline (zero LLM belief)")
        cond_dir = os.path.join(run_dir, "baseline")
        os.makedirs(cond_dir, exist_ok=True)

        meta["conditions"]["baseline"]["status"] = "running"
        _save_metadata(run_dir, meta)

        base_envs = DummyVecEnv([lambda: FedEnvBase(llm_dim=config.LLM_DIM) for _ in range(config.N_ENVS)])
        base_agent, base_network = make_manual_ppo(base_envs, device)

        print(f"Training Baseline for {base_total_timesteps} timesteps...")
        base_agent.learn(total_timesteps=base_total_timesteps, cond_dir=cond_dir)

        model_path = os.path.join(cond_dir, "model_weights.pth")
        torch.save(base_network.state_dict(), model_path)

        meta["conditions"]["baseline"]["status"] = "completed"
        _save_metadata(run_dir, meta)
        print(f"Baseline weights saved to {model_path}\n")

    # Oracle (perfect hardcoded belief)
    if args.condition in ("oracle", "all"):
        print("Condition: Oracle (MockLLMObservationWrapper — perfect belief)")
        cond_dir = os.path.join(run_dir, "oracle")
        os.makedirs(cond_dir, exist_ok=True)

        meta["conditions"]["oracle"]["status"] = "running"
        _save_metadata(run_dir, meta)

        oracle_envs = DummyVecEnv([
            lambda: MockLLMObservationWrapper(FedEnvBase(llm_dim=config.LLM_DIM))
            for _ in range(config.N_ENVS)
        ])
        oracle_agent, oracle_network = make_manual_ppo(oracle_envs, device)

        print(f"Training Oracle condition for {base_total_timesteps} timesteps...")
        oracle_agent.learn(total_timesteps=base_total_timesteps, cond_dir=cond_dir)

        model_path = os.path.join(cond_dir, "model_weights.pth")
        torch.save(oracle_network.state_dict(), model_path)

        meta["conditions"]["oracle"]["status"] = "completed"
        _save_metadata(run_dir, meta)
        print(f">>> Oracle weights saved to {model_path}\n")

    # LLM insight (state-keyed DB)
    if args.condition in ("offline", "both", "all"):
        print("Condition: LLM insight (state-keyed belief DB)")
        cond_dir = os.path.join(run_dir, "llm")
        os.makedirs(cond_dir, exist_ok=True)

        meta["conditions"]["llm"]["status"] = "running"
        _save_metadata(run_dir, meta)

        llm_envs = DummyVecEnv([
            lambda: StateKeyedLLMWrapper(FedEnvBase(llm_dim=config.LLM_DIM), db_path=args.db)
            for _ in range(config.N_ENVS)
        ])
        llm_agent, llm_network = make_manual_ppo(llm_envs, device)

        print(f"Training LLM condition for {base_total_timesteps} timesteps...")
        llm_agent.learn(total_timesteps=base_total_timesteps, cond_dir=cond_dir)

        model_path = os.path.join(cond_dir, "model_weights.pth")
        torch.save(llm_network.state_dict(), model_path)

        meta["conditions"]["llm"]["status"] = "completed"
        _save_metadata(run_dir, meta)
        print(f"LLM weights saved to {model_path}\n")

    print(f"\nManual Run complete. Weights saved to -> {run_dir}")


if __name__ == "__main__":
    main()