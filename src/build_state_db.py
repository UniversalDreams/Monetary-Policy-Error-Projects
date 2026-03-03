"""
src/build_state_db.py — Build a state-keyed offline belief-state DB.

Keys each unique (pi, u, rate) macro state to a 5-dim LLM belief vector,
enabling StateKeyedLLMWrapper to look up beliefs for any episode trajectory
without live inference at training time.

Two-phase sampling strategy:
  Phase 1: N/2 hold-rate episodes  → supply-shock trajectory coverage
  Phase 2: N/2 random-action episodes → policy-induced (rate-varied) coverage

Usage:
    .venv/Scripts/python src/build_state_db.py
    .venv/Scripts/python src/build_state_db.py --episodes 50 --out data/state_belief_db_smoke.json
    .venv/Scripts/python src/build_state_db.py --episodes 1000 --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import config
from fed_env import DirectLLMAdvisor, FedEnvBase, OllamaBackend

# ------------------------------------------------------------─
# LOGGING
# ------------------------------------------------------------─

def _setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

# ------------------------------------------------------------─
# KEY FORMAT
# ------------------------------------------------------------─

def make_key(pi: float, u: float, rate: float) -> str:
    """State key at 0.1% precision — LLM responses don't differ at finer resolution."""
    return f"{pi:.1f}_{u:.1f}_{rate:.2f}"


def parse_key(key: str) -> tuple[float, float, float]:
    parts = key.split("_")
    return float(parts[0]), float(parts[1]), float(parts[2])


# ------------------------------------------------------------─
# EPISODE SIMULATION
# ------------------------------------------------------------─

def collect_states_from_episode(env: FedEnvBase, rng: np.random.Generator,
                                 hold_rate: bool) -> set[str]:
    """
    Run one episode and return all unique state keys encountered.
    hold_rate=True  → always action 3 (0.00 bps change)
    hold_rate=False → uniform random actions
    """
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    keys: set[str] = set()

    done = False
    while not done:
        pi, u, rate = obs["macro"]
        keys.add(make_key(float(pi), float(u), float(rate)))

        action = 3 if hold_rate else int(rng.integers(0, 7))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return keys


# ------------------------------------------------------------─
# DB I/O
# ------------------------------------------------------------─

def load_db(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"metadata": {}, "states": {}}


def save_db(db: dict, path: str, n_unique: int):
    all_config = {k: v for k, v in vars(config).items() if k.isupper()}
    db["metadata"].update({
        "key_format": "pi_u_rate (0.1% resolution)",
        "belief_state_keys": ["P_normal", "P_supply", "sentiment", "hawkishness", "uncertainty"],
        "n_unique_states": n_unique,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d"),
        "config": all_config,
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(db, f)


# ------------------------------------------------------------─
# MAIN
# ------------------------------------------------------------─

def main():
    parser = argparse.ArgumentParser(description="Build state-keyed offline belief-state DB")
    parser.add_argument("--episodes", type=int,  default=1000,
                        help="Total episodes to simulate (default 1000)")
    parser.add_argument("--model",    type=str,  default=config.DEFAULT_MODEL,
                        help=f"Ollama model (default {config.DEFAULT_MODEL})")
    parser.add_argument("--out",      type=str,  default=config.DEFAULT_STATE_DB_PATH,
                        help=f"Output DB path (default {config.DEFAULT_STATE_DB_PATH})")
    parser.add_argument("--seed",     type=int,  default=config.DEFAULT_SEED,
                        help="Master RNG seed (default 42)")
    parser.add_argument("--resume",   action="store_true", default=True,
                        help="Skip keys already in DB (default True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "state_belief_db.log")
    _setup_logging(log_path)
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("build_state_db.py — state-keyed belief DB builder")
    log.info(f"  episodes={args.episodes}  model={args.model}")
    log.info(f"  out={args.out}  seed={args.seed}  resume={args.resume}")
    log.info("=" * 60)

    # -- Load existing DB --------------------------------------
    db = load_db(args.out) if args.resume else {"metadata": {}, "states": {}}
    existing_keys: set[str] = set(db["states"].keys())
    log.info(f"Existing keys in DB: {len(existing_keys)}")

    # -- Phase 1 & 2: collect pending state keys --------------─
    rng = np.random.default_rng(args.seed)
    env = FedEnvBase(llm_dim=config.LLM_DIM)

    n_hold   = args.episodes // 2
    n_random = args.episodes - n_hold
    pending: set[str] = set()

    log.info(f"Phase 1: {n_hold} hold-rate episodes (supply-shock coverage) …")
    t_phase = time.time()
    for ep in range(n_hold):
        keys = collect_states_from_episode(env, rng, hold_rate=True)
        pending |= keys
        if (ep + 1) % 100 == 0:
            log.info(f"  Phase 1  ep {ep+1:>4}/{n_hold}  unique_states={len(pending)}")
    log.info(f"  Phase 1 done — {len(pending)} unique states  ({time.time()-t_phase:.1f}s)")

    log.info(f"Phase 2: {n_random} random-action episodes (policy-induced coverage) …")
    t_phase = time.time()
    prev = len(pending)
    for ep in range(n_random):
        keys = collect_states_from_episode(env, rng, hold_rate=False)
        pending |= keys
        if (ep + 1) % 100 == 0:
            log.info(f"  Phase 2  ep {ep+1:>4}/{n_random}  unique_states={len(pending)}")
    log.info(f"  Phase 2 done — +{len(pending)-prev} new states  total={len(pending)}  ({time.time()-t_phase:.1f}s)")

    env.close()

    new_keys = sorted(pending - existing_keys)
    log.info(f"Total unique pending states : {len(pending)}")
    log.info(f"Already in DB              : {len(existing_keys)}")
    log.info(f"New keys needing LLM calls : {len(new_keys)}")

    if not new_keys:
        log.info("No new states to process — DB is up to date.")
        return

    # -- LLM calls — one per unique new key ------------------─
    advisor = DirectLLMAdvisor(OllamaBackend(model=args.model))
    checkpoint_every = config.CHECKPOINT_EVERY_KEYS
    n_done = 0
    n_errors = 0
    t_start = time.time()
    n_total = len(new_keys)

    log.info(f"Starting LLM calls — {n_total} keys  (checkpoint every {checkpoint_every}) …")
    log.info("-" * 60)

    for i, key in enumerate(new_keys):
        pi, u, rate = parse_key(key)
        t_key = time.time()
        try:
            _, belief = advisor.get_belief_state(pi, u, rate, "unknown")
            db["states"][key] = belief
            n_done += 1
            b = belief
            log.info(
                f"  [{i+1:>5}/{n_total}] {key:<18} "
                f"P_n={b[0]:.2f} P_s={b[1]:.2f} "
                f"sent={b[2]:+.2f} hawk={b[3]:+.2f} unc={b[4]:.2f}  "
                f"({time.time()-t_key:.1f}s)"
            )
        except Exception as exc:
            n_errors += 1
            log.warning(
                f"  [{i+1:>5}/{n_total}] {key:<18} ERROR: {exc}"
            )

        # Progress summary every 10 keys
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate_s  = elapsed / (i + 1)
            eta_s   = rate_s * (n_total - i - 1)
            err_pct = 100 * n_errors / (i + 1)
            log.info(
                f"  -- progress {i+1}/{n_total} ({100*(i+1)/n_total:.0f}%)  "
                f"done={n_done}  errors={n_errors} ({err_pct:.0f}%)  "
                f"{rate_s:.1f}s/key  ETA={eta_s/60:.1f}min"
            )

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            save_db(db, args.out, len(db["states"]))
            log.info(f"  -- checkpoint saved — {len(db['states'])} total keys in DB")

    # Final save
    save_db(db, args.out, len(db["states"]))
    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info(f"Done.")
    log.info(f"  Total keys in DB : {len(db['states'])}")
    log.info(f"  New              : {n_done}")
    log.info(f"  Errors           : {n_errors}  ({100*n_errors/max(1,n_total):.1f}%)")
    log.info(f"  Elapsed          : {elapsed/60:.1f} min  ({elapsed/max(1,n_done):.1f}s/key avg)")
    log.info(f"  Saved to         : {args.out}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
