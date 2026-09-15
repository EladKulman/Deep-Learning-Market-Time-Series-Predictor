#!/usr/bin/env python3
"""Rank and plot permutation-importance runs produced by scripts/permutation_importance.py.

    python scripts/summarize_permutation_importance.py models/permutation-importance/run1 [run2 ...] \
        --output-dir docs/assets/permutation --top 20

Writes into --output-dir:
  permutation_importance_ranking.md   ranked table (one section per run)
  <run>_heatmap.png                   features x forecast-origin dates, delta WQL, diverging around 0
  <run>_by_horizon.png                mean delta WQL per horizon for the top features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Output directories of permutation_importance.py")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/permutation"))
    parser.add_argument("--top", type=int, default=20, help="Features shown in the plots")
    return parser.parse_args()


def load_run(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_window = pd.read_csv(run / "importance_by_window.csv", parse_dates=["forecast_origin"])
    summary = pd.read_csv(run / "importance_summary.csv")
    return by_window, summary


def markdown_table(summary: pd.DataFrame, horizons: list[str]) -> str:
    lines = ["| Rank | Feature | Mean dWQL | Relative | Std | Windows > 0 | " + " | ".join(h.upper() for h in horizons) + " |",
             "|---:|---|---:|---:|---:|---:|" + "---:|" * len(horizons)]
    for rank, row in enumerate(summary.itertuples(index=False), start=1):
        per_h = " | ".join(f"{getattr(row, f'mean_delta_{h}'):+.5f}" for h in horizons)
        star = " (known-future)" if int(getattr(row, "known_future", 0)) else ""
        lines.append(f"| {rank} | `{row.feature}`{star} | {row.mean_delta_wql:+.5f} | {row.mean_relative_delta:+.2%} | "
                     f"{row.std_delta_wql:.5f} | {row.share_windows_positive:.0%} | {per_h} |")
    return "\n".join(lines)


def heatmap(by_window: pd.DataFrame, order: list[str], path: Path, title: str) -> None:
    pivot = (by_window.groupby(["feature", "forecast_origin"])["delta_wql"].mean().unstack("forecast_origin")
             .reindex(order))
    values = pivot.to_numpy()
    limit = float(np.nanmax(np.abs(values))) or 1e-6
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * pivot.shape[1] + 3), max(4, 0.32 * len(order) + 1.5)))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    ax.set_yticks(range(len(order)), order, fontsize=8)
    dates = [d.strftime("%Y-%m-%d") for d in pivot.columns]
    step = max(1, len(dates) // 12)
    ax.set_xticks(range(0, len(dates), step), dates[::step], rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Forecast origin (last context session)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="Change in WQL when the feature is swapped in from another window")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_horizon_plot(summary: pd.DataFrame, horizons: list[str], top: int, path: Path, title: str) -> None:
    head = summary.head(top)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(horizons) + 1)
    for row in head.itertuples(index=False):
        ax.plot(x, [getattr(row, f"mean_delta_{h}") for h in horizons], marker="o", linewidth=1.2, label=row.feature)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(x, [f"h={i}" for i in x])
    ax.set_ylabel("Mean change in WQL")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sections = ["# Permutation importance (swap between validation windows)", "",
                "Positive values: the forecast got worse when the feature was replaced by the same feature from "
                "another window, i.e. the model was using it. Relative = mean of dWQL / baseline WQL per window.", ""]
    for run in args.runs:
        by_window, summary = load_run(run)
        horizons = [c.replace("mean_delta_", "") for c in summary.columns if c.startswith("mean_delta_h")]
        order = summary["feature"].tolist()[: args.top]
        name = run.name
        heatmap(by_window[by_window["feature"].isin(order)], order, args.output_dir / f"{name}_heatmap.png",
                f"{name}: permutation importance over time (top {len(order)})")
        by_horizon_plot(summary, horizons, min(args.top, 10), args.output_dir / f"{name}_by_horizon.png",
                        f"{name}: importance by forecast horizon (top {min(args.top, 10)})")
        n_windows = by_window["window"].nunique()
        n_repeats = by_window["repeat"].nunique()
        sections += [f"## {name}", "", f"{n_windows} windows x {n_repeats} repeats; baseline mean WQL "
                     f"{by_window.drop_duplicates('window')['baseline_wql'].mean():.6f}.", "",
                     markdown_table(summary, horizons), "",
                     f"![heatmap]({name}_heatmap.png)", "", f"![by horizon]({name}_by_horizon.png)", ""]
    (args.output_dir / "permutation_importance_ranking.md").write_text("\n".join(sections))
    print(f"Wrote {args.output_dir}/permutation_importance_ranking.md and {2 * len(args.runs)} figures")


if __name__ == "__main__":
    main()
