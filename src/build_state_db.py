"""
src/build_state_db.py — Build a state-keyed offline belief-state DB.

Keys each unique (pi, u, rate) macro state to a 5-dim LLM belief vector,
enabling StateKeyedLLMWrapper to look up beliefs for any episode trajectory
without live inference at training time.
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
from fed_env import DirectLLMAdvisor, HierarchicalLLMAdvisor, OllamaBackend

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
# GRID ENUMERATION
# ------------------------------------------------------------─

def enumerate_grid_keys(pi_min, pi_max, pi_step,
                         u_min,  u_max,  u_step,
                         rate_min, rate_max, rate_step) -> list[str]:
    """Return all (pi, u, rate) grid keys covering the reachable state space."""
    keys = []
    for pi in np.arange(pi_min, pi_max + pi_step / 2, pi_step):
        for u in np.arange(u_min, u_max + u_step / 2, u_step):
            for rate in np.arange(rate_min, rate_max + rate_step / 2, rate_step):
                keys.append(make_key(round(pi, 4), round(u, 4), round(rate, 4)))
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
    parser = argparse.ArgumentParser(description="Build state-keyed offline belief-state DB via grid search")
    # Grid bounds
    parser.add_argument("--pi-min",   type=float, default=-1.0,
                        help="Min inflation rate (default -1.0)")
    parser.add_argument("--pi-max",   type=float, default=12.0,
                        help="Max inflation rate (default 12.0)")
    parser.add_argument("--pi-step",  type=float, default=0.5,
                        help="Inflation grid step (default 0.5)")
    parser.add_argument("--u-min",    type=float, default=1.5,
                        help="Min unemployment rate (default 1.5)")
    parser.add_argument("--u-max",    type=float, default=11.0,
                        help="Max unemployment rate (default 11.0)")
    parser.add_argument("--u-step",   type=float, default=0.5,
                        help="Unemployment grid step (default 0.5)")
    parser.add_argument("--rate-min", type=float, default=0.0,
                        help="Min fed funds rate (default 0.0)")
    parser.add_argument("--rate-max", type=float, default=15.0,
                        help="Max fed funds rate (default 15.0)")
    parser.add_argument("--rate-step",type=float, default=0.25,
                        help="Fed funds rate grid step (default 0.25)")
    # Other
    parser.add_argument("--advisor", choices=["direct", "hierarchical"], default="direct",
                        help="LLM advisor: 'direct' (1 call/point) or 'hierarchical' (8 calls/point). Default: direct")
    parser.add_argument("--model",    type=str,  default=config.DEFAULT_MODEL,
                        help=f"Ollama model (default {config.DEFAULT_MODEL})")
    parser.add_argument("--out",      type=str,  default=config.DEFAULT_STATE_DB_PATH,
                        help=f"Output DB path (default {config.DEFAULT_STATE_DB_PATH})")
    parser.add_argument("--resume",   action="store_true", default=True,
                        help="Skip keys already in DB (default True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "state_belief_db.log")
    _setup_logging(log_path)
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("build_state_db.py — state-keyed belief DB builder (grid search)")
    log.info(f"  pi=[{args.pi_min},{args.pi_max}] step={args.pi_step}")
    log.info(f"  u=[{args.u_min},{args.u_max}] step={args.u_step}")
    log.info(f"  rate=[{args.rate_min},{args.rate_max}] step={args.rate_step}")
    log.info(f"  model={args.model}  advisor={args.advisor}  out={args.out}  resume={args.resume}")
    log.info("=" * 60)

    # -- Load existing DB --------------------------------------
    db = load_db(args.out) if args.resume else {"metadata": {}, "states": {}}
    existing_keys: set[str] = set(db["states"].keys())

    # -- Enumerate grid ----------------------------------------
    all_keys = enumerate_grid_keys(
        args.pi_min, args.pi_max, args.pi_step,
        args.u_min,  args.u_max,  args.u_step,
        args.rate_min, args.rate_max, args.rate_step,
    )
    new_keys = [k for k in all_keys if k not in existing_keys]
    log.info(
        f"Grid: {len(all_keys)} total points  |  "
        f"{len(existing_keys)} already in DB  |  "
        f"{len(new_keys)} to query"
    )

    if not new_keys:
        log.info("No new states to process — DB is up to date.")
        return

    # -- LLM calls — one per unique new key ------------------─
    if args.advisor == "hierarchical":
        advisor = HierarchicalLLMAdvisor(OllamaBackend(model=args.model))
    else:
        advisor = DirectLLMAdvisor(OllamaBackend(model=args.model))
    log.info(f"Advisor: {args.advisor}  ({'~8 calls/point' if args.advisor == 'hierarchical' else '1 call/point'})")
    checkpoint_every = config.CHECKPOINT_EVERY_KEYS
    n_done = 0
    n_errors = 0
    t_start = time.time()
    n_total = len(new_keys)

    log.info(f"Starting LLM calls — {n_total} keys  (checkpoint every {checkpoint_every}) …")
    log.info("-" * 60)

    for i, key in enumerate(new_keys):
        pi, u, rate = parse_key(key)
        advisor.reset_episode()
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
