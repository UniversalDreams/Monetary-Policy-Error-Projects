"""
scripts/analyze_run.py — Analyse a training run produced by train_custom.py.

Reads training.csv from each condition (baseline / llm / oracle) in a run
directory and produces:
  - Printed summary table (mean, median, std, p5, p25, p75, p95, min, max)
  - Catastrophic episode rate (reward < CATASTROPHIC_THRESHOLD) by window
  - Rolling-window reward curve (saved as training_curves.png)
  - Benchmark overlay if benchmark.csv is present in the run directory

Usage:
    # by run directory
    .venv/Scripts/python scripts/analyze_run.py runs_custom/paper_run

    # by run name inside default out dir
    .venv/Scripts/python scripts/analyze_run.py --run paper_run

    # choose conditions explicitly
    .venv/Scripts/python scripts/analyze_run.py runs_custom/paper_run --conditions baseline llm

    # set a custom catastrophic threshold
    .venv/Scripts/python scripts/analyze_run.py runs_custom/paper_run --threshold -10000

    # only print, don't save plots
    .venv/Scripts/python scripts/analyze_run.py runs_custom/paper_run --no-plot
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CATASTROPHIC_THRESHOLD = -10_000   # episode reward below this is "catastrophic"
ROLLING_WINDOW        = 100        # episodes for rolling average
CATASTROPHIC_WINDOW   = 500        # episodes per bucket for catastrophic-rate table

# Reference benchmarks from results/paper_run (SB3 RecurrentPPO, 100 seeds, 1.2M steps)
PAPER_BENCHMARKS = {
    "taylor_rule":    {"mean": -648,  "median": -312},
    "sb3_baseline":   {"mean": -701,  "median": -459},
    "sb3_llm":        {"mean": -519,  "median": -280},
    "sb3_oracle":     {"mean": -593,  "median": -335},
}

ALL_CONDITIONS = ["baseline", "llm", "oracle"]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rewards_from_rows(rows: list[dict]) -> np.ndarray:
    return np.array([float(r["ep_reward"]) for r in rows])


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def summary_stats(rewards: np.ndarray) -> dict:
    return {
        "n":      len(rewards),
        "mean":   float(np.mean(rewards)),
        "median": float(np.median(rewards)),
        "std":    float(np.std(rewards)),
        "p5":     float(np.percentile(rewards, 5)),
        "p25":    float(np.percentile(rewards, 25)),
        "p75":    float(np.percentile(rewards, 75)),
        "p95":    float(np.percentile(rewards, 95)),
        "min":    float(np.min(rewards)),
        "max":    float(np.max(rewards)),
    }


def rolling_mean(rewards: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """Unpadded rolling mean — length = len(rewards) - window + 1."""
    cumsum = np.cumsum(rewards)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / window


def catastrophic_rate_by_window(
    rewards: np.ndarray,
    threshold: float = CATASTROPHIC_THRESHOLD,
    window: int = CATASTROPHIC_WINDOW,
) -> list[dict]:
    """
    Returns list of dicts with keys: start_ep, end_ep, n_episodes, n_catastrophic, rate_pct.
    """
    results = []
    n = len(rewards)
    for start in range(0, n, window):
        end   = min(start + window, n)
        chunk = rewards[start:end]
        n_cat = int(np.sum(chunk < threshold))
        results.append({
            "start_ep":      start + 1,
            "end_ep":        end,
            "n_episodes":    len(chunk),
            "n_catastrophic": n_cat,
            "rate_pct":      100.0 * n_cat / len(chunk),
        })
    return results


def recent_improvement(rewards: np.ndarray, window: int = ROLLING_WINDOW) -> dict | None:
    """Compare last N episodes to the N episodes before that."""
    if len(rewards) < 2 * window:
        return None
    recent = rewards[-window:]
    prior  = rewards[-2 * window:-window]
    return {
        "prior_mean":  float(np.mean(prior)),
        "recent_mean": float(np.mean(recent)),
        "delta":       float(np.mean(recent) - np.mean(prior)),
    }


def peak_rolling(rewards: np.ndarray, window: int = ROLLING_WINDOW) -> dict | None:
    """Return the peak rolling-window mean and the episode it was achieved."""
    if len(rewards) < window:
        return None
    rm   = rolling_mean(rewards, window)
    idx  = int(np.argmax(rm))
    return {
        "peak_rolling_mean": float(rm[idx]),
        "at_episode":        idx + window,   # episode index of the window's last step
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def _hr(width=72):
    print("-" * width)

HR = "-" * 72


def print_summary_table(condition_stats: dict[str, dict]) -> None:
    cols = ["mean", "median", "std", "p5", "p25", "p75", "p95", "min", "max"]
    w_name = max(len(k) for k in condition_stats) + 2
    w_col  = 10

    header = f"{'condition':<{w_name}}" + "".join(f"{c:>{w_col}}" for c in cols)
    print(header)
    _hr(len(header))
    for cond, s in condition_stats.items():
        row = f"{cond:<{w_name}}"
        for c in cols:
            row += f"{s[c]:>{w_col}.1f}"
        print(row)


def print_catastrophic_table(
    cat_by_window: list[dict],
    threshold: float,
    condition: str,
) -> None:
    print(f"\nCatastrophic rate (reward < {threshold:,.0f}) -- {condition}")
    print(f"  {'ep range':<16}  {'n_eps':>6}  {'n_cat':>6}  {'rate':>7}")
    _hr(48)
    for b in cat_by_window:
        rng = f"{b['start_ep']}-{b['end_ep']}"
        print(f"  {rng:<16}  {b['n_episodes']:>6}  {b['n_catastrophic']:>6}  {b['rate_pct']:>6.1f}%")


def print_benchmark_comparison(
    condition_stats: dict[str, dict],
    benchmark_csv_rows: list[dict] | None,
) -> None:
    print("\nBenchmark comparison")
    _hr(60)

    # Build reference rows: paper constants + any benchmark.csv conditions
    refs = {}
    refs.update(PAPER_BENCHMARKS)

    if benchmark_csv_rows:
        cond_names = [k for k in benchmark_csv_rows[0] if k != "seed"]
        for cname in cond_names:
            vals = [float(r[cname]) for r in benchmark_csv_rows if r[cname]]
            if vals:
                refs[f"eval:{cname}"] = {
                    "mean":   float(np.mean(vals)),
                    "median": float(np.median(vals)),
                }

    all_rows = {}
    for name, s in refs.items():
        all_rows[name] = s
    for cond, s in condition_stats.items():
        all_rows[f"trained:{cond}"] = {"mean": s["mean"], "median": s["median"]}

    w = max(len(k) for k in all_rows) + 2
    print(f"{'':>{w}}  {'mean':>10}  {'median':>10}")
    _hr(w + 26)
    for name, s in all_rows.items():
        print(f"  {name:<{w}}  {s['mean']:>10.1f}  {s['median']:>10.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(
    condition_rewards: dict[str, np.ndarray],
    run_dir: str,
    threshold: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping plots")
        return

    n_conds = len(condition_rewards)
    fig, axes = plt.subplots(n_conds, 2, figsize=(14, 4 * n_conds), squeeze=False)
    fig.suptitle(f"Training analysis — {os.path.basename(run_dir)}", fontsize=13)

    colors = ["steelblue", "darkorange", "seagreen"]

    for row_i, (cond, rewards) in enumerate(condition_rewards.items()):
        color = colors[row_i % len(colors)]
        eps   = np.arange(1, len(rewards) + 1)

        # ── Left: raw episode rewards + rolling mean ──────────────────────
        ax = axes[row_i][0]
        ax.scatter(eps, rewards, s=2, alpha=0.3, color=color, label="episode reward")
        if len(rewards) >= ROLLING_WINDOW:
            rm    = rolling_mean(rewards)
            rm_ep = np.arange(ROLLING_WINDOW, len(rewards) + 1)
            ax.plot(rm_ep, rm, color="black", lw=1.5, label=f"rolling-{ROLLING_WINDOW}")
        ax.axhline(threshold, color="red", ls="--", lw=0.8, label=f"catastrophic ({threshold:,.0f})")
        ax.set_title(f"{cond} — reward curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode reward")
        ax.legend(fontsize=8)

        # ── Right: catastrophic rate per window ───────────────────────────
        ax2 = axes[row_i][1]
        cat = catastrophic_rate_by_window(rewards, threshold)
        midpoints = [(b["start_ep"] + b["end_ep"]) / 2 for b in cat]
        rates     = [b["rate_pct"] for b in cat]
        ax2.bar(midpoints, rates, width=CATASTROPHIC_WINDOW * 0.8,
                color=color, alpha=0.7, edgecolor="white")
        ax2.set_title(f"{cond} — catastrophic rate per {CATASTROPHIC_WINDOW} eps")
        ax2.set_xlabel("Episode (window midpoint)")
        ax2.set_ylabel("% episodes < threshold")
        ax2.set_ylim(0, max(rates) * 1.2 + 1)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "analysis_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse a train_custom.py run directory")
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Path to run directory (e.g. runs_custom/paper_run)")
    parser.add_argument("--run",        type=str, default=None,
                        help="Run name inside runs_custom/ (alternative to positional arg)")
    parser.add_argument("--out",        type=str, default="runs_custom",
                        help="Base output dir when using --run (default: runs_custom)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=ALL_CONDITIONS,
                        help="Conditions to analyse (default: all present)")
    parser.add_argument("--threshold",  type=float, default=CATASTROPHIC_THRESHOLD,
                        help=f"Catastrophic episode threshold (default: {CATASTROPHIC_THRESHOLD})")
    parser.add_argument("--window",     type=int,   default=ROLLING_WINDOW,
                        help=f"Rolling-mean window (default: {ROLLING_WINDOW})")
    parser.add_argument("--cat-window", type=int,   default=CATASTROPHIC_WINDOW,
                        help=f"Catastrophic-rate bucket size (default: {CATASTROPHIC_WINDOW})")
    parser.add_argument("--no-plot",    action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    # Resolve run directory
    if args.run_dir:
        run_dir = args.run_dir
    elif args.run:
        run_dir = args.run if os.path.isdir(args.run) else os.path.join(args.out, args.run)
    else:
        parser.error("Provide a run_dir positional argument or --run <name>")

    if not os.path.isdir(run_dir):
        sys.exit(f"ERROR: run directory not found: {run_dir}")

    # Load metadata if present
    meta_path = os.path.join(run_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # Resolve which conditions to analyse
    requested = args.conditions or ALL_CONDITIONS
    conditions_to_run = []
    for cond in requested:
        csv_path = os.path.join(run_dir, cond, "training.csv")
        if os.path.exists(csv_path):
            conditions_to_run.append((cond, csv_path))
        else:
            print(f"[skip] {cond}: training.csv not found")

    if not conditions_to_run:
        sys.exit("No training CSVs found — nothing to analyse.")

    # Load benchmark CSV if present
    bench_csv_path = os.path.join(run_dir, "benchmark.csv")
    bench_rows = None
    if os.path.exists(bench_csv_path):
        bench_rows = load_csv(bench_csv_path)

    # ── Per-condition analysis ─────────────────────────────────────────────
    condition_rewards: dict[str, np.ndarray] = {}
    condition_stats:   dict[str, dict]       = {}

    print(f"\n{'='*72}")
    print(f"  Run: {run_dir}")
    if meta:
        print(f"  policy={meta.get('policy','?')}  seed={meta.get('seed','?')}")
    print(f"  threshold={args.threshold:,.0f}  rolling_window={args.window}")
    print(f"{'='*72}")

    for cond, csv_path in conditions_to_run:
        rows    = load_csv(csv_path)
        rewards = rewards_from_rows(rows)
        condition_rewards[cond] = rewards

        s = summary_stats(rewards)
        condition_stats[cond] = s

        peak  = peak_rolling(rewards, args.window)
        impro = recent_improvement(rewards, args.window)

        print(f"\n{HR}")
        print(f"  Condition: {cond}   ({len(rewards)} episodes)")
        print(HR)

        print(f"\n  Summary statistics:")
        print(f"    mean={s['mean']:>10.1f}   median={s['median']:>10.1f}   std={s['std']:>10.1f}")
        print(f"    p5  ={s['p5']:>10.1f}   p25   ={s['p25']:>10.1f}")
        print(f"    p75 ={s['p75']:>10.1f}   p95   ={s['p95']:>10.1f}")
        print(f"    min ={s['min']:>10.1f}   max   ={s['max']:>10.1f}")

        if peak:
            print(f"\n  Peak rolling-{args.window}: {peak['peak_rolling_mean']:.1f}  "
                  f"(at ep {peak['at_episode']})")

        if impro:
            sign = "+" if impro["delta"] >= 0 else ""
            print(f"  Last {args.window} vs prior {args.window}: "
                  f"{impro['prior_mean']:.1f} -> {impro['recent_mean']:.1f}  "
                  f"({sign}{impro['delta']:.1f})")

        cat = catastrophic_rate_by_window(rewards, args.threshold, args.cat_window)
        print_catastrophic_table(cat, args.threshold, cond)

    # ── Cross-condition summary ────────────────────────────────────────────
    if len(condition_stats) > 1 or bench_rows:
        print(f"\n{'='*72}")
        print("  Summary table -- all conditions")
        print(f"{'='*72}\n")
        print_summary_table(condition_stats)

    # ── Benchmark comparison ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print_benchmark_comparison(condition_stats, bench_rows)

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plot:
        make_plots(condition_rewards, run_dir, args.threshold)

    print()


if __name__ == "__main__":
    main()
