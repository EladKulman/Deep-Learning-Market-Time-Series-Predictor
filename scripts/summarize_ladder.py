#!/usr/bin/env python3
"""Summarize a pretrained-ladder run into docs-ready tables and a chart.

    python scripts/summarize_ladder.py --root models/pretrained-ladder

Writes <root>/ladder_summary.md and <root>/ladder_wql.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RUNG_ORDER = [
    "gaussian", "r0_target", "r1_qqq", "r2_qqq_calendar", "r3_control", "r3_control_nofuture",
    "r4_plus_uncertainty", "r5_plus_fed", "r6_plus_gdelt_fed_recession", "r7_all_external",
]
REFERENCE = "r3_control"


def ordered(values) -> list[str]:
    present = set(values)
    return [r for r in RUNG_ORDER if r in present] + sorted(present - set(RUNG_ORDER))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models/pretrained-ladder"))
    args = parser.parse_args()

    metrics = pd.read_csv(args.root / "ladder_metrics.csv")
    overall = metrics[metrics["horizon"].astype(str) == "overall"].copy()
    folds = list(dict.fromkeys(overall["fold"]))
    rungs = ordered(overall["rung"])

    wql = overall.pivot_table(index="rung", columns="fold", values="weighted_quantile_loss").reindex(rungs)[folds]
    wql["mean"] = wql.mean(axis=1)
    delta = (wql[folds].sub(wql.loc[REFERENCE, folds], axis=1) / wql.loc[REFERENCE, folds] * 100) if REFERENCE in wql.index else None
    coverage = overall.pivot_table(index="rung", columns="fold", values="q10_q90_coverage").reindex(rungs)[folds]
    direction = overall.pivot_table(index="rung", columns="fold", values="directional_accuracy").reindex(rungs)[folds]

    candidates = wql.drop(index=[r for r in ("gaussian",) if r in wql.index])
    best = candidates["mean"].idxmin()
    per_h = metrics[(metrics["rung"].isin([best, REFERENCE, "r0_target"])) & (metrics["horizon"].astype(str) != "overall")].copy()
    per_h["horizon"] = per_h["horizon"].astype(int)
    per_h_table = per_h.pivot_table(index="horizon", columns="rung", values="weighted_quantile_loss", aggfunc="mean")

    lines = ["# Zero-shot covariate ladder (pretrained Chronos-2, no fine-tuning)", "",
             f"Folds: {', '.join(folds)}. Lower WQL is better. Reference rung: `{REFERENCE}`.", "",
             "## Weighted quantile loss by rung and fold", "", wql.round(5).to_markdown(), ""]
    if delta is not None:
        lines += ["## Percent change in WQL versus the control rung (negative = better)", "", delta.round(2).to_markdown(), ""]
    lines += ["## 10%-90% interval coverage (target 0.80)", "", coverage.round(3).to_markdown(), "",
              "## Directional accuracy (compare with the up-day base rate, not 50%)", "", direction.round(3).to_markdown(), "",
              f"## Per-horizon WQL, averaged over folds: best rung `{best}` versus control and target-only", "",
              per_h_table.round(5).to_markdown(), ""]
    if "gaussian" in wql.index:
        ups = overall[overall["rung"] == "gaussian"].set_index("fold")["directional_accuracy"].reindex(folds)
        lines += ["Up-day base rate per fold (Gaussian zero-forecast direction): " + ", ".join(f"{f} {v:.1%}" for f, v in ups.items()), ""]
    (args.root / "ladder_summary.md").write_text("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped chart")
    else:
        plot_rungs = [r for r in rungs if r != "gaussian"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for fold in folds:
            ax.plot(range(len(plot_rungs)), wql.loc[plot_rungs, fold], marker="o", label=fold)
            if "gaussian" in wql.index:
                ax.axhline(wql.loc["gaussian", fold], linestyle=":", linewidth=0.8, color=ax.lines[-1].get_color())
        ax.set_xticks(range(len(plot_rungs)))
        ax.set_xticklabels(plot_rungs, rotation=35, ha="right")
        ax.set_ylabel("Weighted quantile loss")
        ax.set_title("Pretrained Chronos-2: covariate ladder by validation fold (dotted = Gaussian baseline)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.root / "ladder_wql.png", dpi=150)
    print((args.root / "ladder_summary.md").read_text())


if __name__ == "__main__":
    main()
