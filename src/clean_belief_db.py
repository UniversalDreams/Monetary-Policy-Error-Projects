"""
src/clean_belief_db.py — Normalize P_normal + P_supply to sum to 1.0 in-place.

The LLM occasionally violates the simplex constraint, producing P_n + P_s != 1.0.
This script rescales the two regime probabilities for every entry in the DB.
"""

import argparse
import json
import os
import sys


def normalize_entry(belief: list) -> tuple[list, float]:
    """Normalize belief[0] (P_normal) and belief[1] (P_supply) to sum to 1.
    Returns (normalized_belief, original_sum)."""
    p_n, p_s = belief[0], belief[1]
    total = p_n + p_s
    if total > 0:
        belief = [p_n / total, p_s / total] + belief[2:]
    else:
        belief = [0.5, 0.5] + belief[2:]
    return belief, total


def main():
    parser = argparse.ArgumentParser(description="Normalize P_normal+P_supply to 1.0 in belief state DB")
    parser.add_argument("--db",        type=str, required=True,  help="Path to belief state DB JSON")
    parser.add_argument("--out",       type=str, default=None,   help="Output path (default: overwrite input)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Flag entries where |sum-1| > threshold (default 0.01)")
    parser.add_argument("--dry-run",   action="store_true",      help="Report violations without writing")
    args = parser.parse_args()

    out_path = args.out or args.db

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found: {args.db}")
        sys.exit(1)

    with open(args.db) as f:
        db = json.load(f)

    states = db.get("states", {})
    n_total = len(states)
    n_bad = 0
    n_fixed = 0
    worst_key, worst_dev = None, 0.0

    for key, belief in states.items():
        total = belief[0] + belief[1]
        dev = abs(total - 1.0)
        if dev > worst_dev:
            worst_dev, worst_key = dev, key
        if dev > args.threshold:
            n_bad += 1
            if not args.dry_run:
                states[key], _ = normalize_entry(belief)
                n_fixed += 1

    print(f"DB: {args.db}")
    print(f"Total keys  : {n_total}")
    print(f"Violations (|P_n+P_s - 1| > {args.threshold}): {n_bad}  ({100*n_bad/max(1,n_total):.1f}%)")
    if worst_key:
        b = db["states"][worst_key]
        print(f"Worst key   : {worst_key}  sum={b[0]+b[1]:.4f}  (dev={worst_dev:.4f})")

    if args.dry_run:
        print("Dry run — no changes written.")
        return

    db["states"] = states
    db["metadata"]["cleaned"] = True
    db["metadata"]["clean_threshold"] = args.threshold

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(db, f)

    print(f"Fixed       : {n_fixed} keys")
    print(f"Saved to    : {out_path}")


if __name__ == "__main__":
    main()
