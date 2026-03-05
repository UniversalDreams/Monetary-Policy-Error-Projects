"""
src/plot_belief_db.py — Heatmaps of the state-keyed belief DB.

Shows each of the 5 belief dimensions as a 2D heatmap over (pi, u),
either averaged across all rate values or sliced at a specific rate.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


DIMS = ["P_normal", "P_supply", "sentiment", "hawkishness", "uncertainty"]

# Diverging colormap for signed dims, sequential for [0,1] dims
DIM_CMAPS = {
    "P_normal":    ("Blues",    0.0,  1.0),
    "P_supply":    ("Reds",     0.0,  1.0),
    "sentiment":   ("RdYlGn",  -1.0,  1.0),
    "hawkishness": ("RdYlGn",  -1.0,  1.0),
    "uncertainty": ("YlOrRd",   0.0,  1.0),
}


def load_db(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_grid(states: dict, fix_rate=None):
    """
    Returns (pi_vals, u_vals, grid) where grid has shape (n_dims, n_u, n_pi).
    If fix_rate is given, use only that rate slice; otherwise average over all rates.
    """
    pis   = sorted(set(float(k.split("_")[0]) for k in states))
    us    = sorted(set(float(k.split("_")[1]) for k in states))
    rates = sorted(set(float(k.split("_")[2]) for k in states))

    if fix_rate is not None:
        # Find closest available rate
        closest = min(rates, key=lambda r: abs(r - fix_rate))
        if abs(closest - fix_rate) > 1e-6:
            print(f"Note: rate {fix_rate} not in DB — using closest: {closest}")
        use_rates = [closest]
    else:
        use_rates = rates

    pi_idx = {v: i for i, v in enumerate(pis)}
    u_idx  = {v: i for i, v in enumerate(us)}

    # accumulator: shape (n_dims, n_u, n_pi)
    acc   = np.zeros((len(DIMS), len(us), len(pis)))
    count = np.zeros((len(us), len(pis)))

    for key, belief in states.items():
        parts = key.split("_")
        pi, u, rate = float(parts[0]), float(parts[1]), float(parts[2])
        if rate not in use_rates:
            continue
        i = u_idx[u]
        j = pi_idx[pi]
        acc[:, i, j] += belief
        count[i, j]  += 1

    # Avoid divide-by-zero for missing cells
    mask = count > 0
    grid = np.full_like(acc, np.nan)
    for d in range(len(DIMS)):
        grid[d][mask] = acc[d][mask] / count[mask]

    return pis, us, grid


def plot_heatmaps(pis, us, grid, fix_rate, out_path):
    n_dims = len(DIMS)
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4.5))

    rate_label = f"rate={fix_rate:.2f}%" if fix_rate is not None else "avg over all rates"
    fig.suptitle(f"Belief State DB Heatmaps  ({rate_label})", fontsize=13, fontweight="bold", y=1.01)

    for ax, dim, (cmap, vmin, vmax) in zip(axes, DIMS, DIM_CMAPS.values()):
        d_idx = DIMS.index(dim)
        data = grid[d_idx]  # shape (n_u, n_pi) — rows=u, cols=pi

        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_xticks(range(len(pis)))
        ax.set_xticklabels([f"{p:.0f}" for p in pis], fontsize=7, rotation=45)
        ax.set_yticks(range(len(us)))
        ax.set_yticklabels([f"{u:.0f}" for u in us], fontsize=7)
        ax.set_xlabel("Inflation π (%)", fontsize=9)
        ax.set_ylabel("Unemployment u (%)", fontsize=9)
        ax.set_title(dim, fontsize=10, fontweight="bold")

        # Annotate cells with value
        for i in range(len(us)):
            for j in range(len(pis)):
                v = data[i, j]
                if not np.isnan(v):
                    text_color = "white" if abs(v) > 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color=text_color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Heatmap of state-keyed belief DB")
    parser.add_argument("--db",       type=str, required=True,  help="Path to belief state DB JSON")
    parser.add_argument("--fix-rate", type=float, default=None, help="Show slice at this fed funds rate (default: average over all rates)")
    parser.add_argument("--out",      type=str, default=None,   help="Output image path (default: alongside DB)")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found: {args.db}")
        sys.exit(1)

    db = load_db(args.db)
    states = db.get("states", {})
    print(f"Loaded {len(states)} states from {args.db}")

    pis, us, grid = build_grid(states, fix_rate=args.fix_rate)
    print(f"Grid: {len(pis)} pi values × {len(us)} u values")

    if args.out:
        out_path = args.out
    else:
        suffix = f"_rate{args.fix_rate:.0f}" if args.fix_rate is not None else "_avg"
        base = os.path.splitext(args.db)[0]
        out_path = base + f"_heatmap{suffix}.png"

    plot_heatmaps(pis, us, grid, args.fix_rate, out_path)


if __name__ == "__main__":
    main()
