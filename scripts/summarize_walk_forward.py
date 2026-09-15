#!/usr/bin/env python3
"""Summarize walk-forward ladder runs (fine-tuned and pretrained arms) into report tables.

Reads every ``<rung>-fold<year>-seed<S>/validation_predictions.csv`` and
``<rung>-fold<year>-pretrained/validation_predictions.csv`` under the given roots. Each file
stacks the models evaluated on identical windows (``base_pretrained``, ``fine_tuned``,
``zero_gaussian``).

Statistics use per-window pinball loss normalized by the fold-level mean absolute return, so
the mean over windows equals the aggregate weighted quantile loss (WQL) and paired bootstrap
intervals over windows are on the same scale as the headline numbers. Seeds are averaged per
window before pairing, so the intervals reflect window-to-window variation of the seed-mean.

    python scripts/summarize_walk_forward.py models/walk-forward-896510 models/walk-forward-pretrained-896501 \
        --output-dir docs/results/walk_forward
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

CELL = re.compile(r"^(?P<rung>.+)-fold(?P<fold>[^-]+)-(?P<arm>pretrained|seed\d+)$")
RUNG_ORDER = ["r0_target", "r1_qqq", "r2_qqq_calendar", "r3_qqq_calendar_market", "r4_plus_uncertainty",
              "r5_plus_fed", "r6_plus_gdelt_fed_recession", "r7_all_external"]


def per_window_loss(frame: pd.DataFrame) -> pd.Series:
    """Mean pinball loss per window over all quantiles and horizons."""
    qcols = [c for c in frame.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]
    out = {}
    for window, g in frame.groupby("window"):
        losses = []
        for c in qcols:
            q = float(c[1:]); e = g["actual"] - g[c]
            losses.append(np.maximum(q * e, (q - 1) * e).to_numpy())
        out[window] = float(np.mean(losses))
    return pd.Series(out).sort_index()


def load_runs(roots: list[Path]) -> pd.DataFrame:
    rows = []
    for root in roots:
        for path in sorted(root.glob("*/validation_predictions.csv")):
            match = CELL.match(path.parent.name)
            if not match:
                continue
            frame = pd.read_csv(path)
            fold, rung, arm = match["fold"], match["rung"], match["arm"]
            denominator = frame.loc[frame["model"] == frame["model"].iloc[0], "actual"].abs().mean()
            for model, g in frame.groupby("model"):
                if arm == "pretrained" and model == "fine_tuned":
                    continue
                losses = per_window_loss(g)
                overall = g.copy()
                cov = ((overall["actual"] >= overall["q0.1"]) & (overall["actual"] <= overall["q0.9"])).mean() if "q0.1" in overall else np.nan
                cov99 = ((overall["actual"] >= overall["q0.01"]) & (overall["actual"] <= overall["q0.99"])).mean() if "q0.01" in overall else np.nan
                direction = ((overall["prediction"] >= 0) == (overall["actual"] >= 0)).mean()
                for window, loss in losses.items():
                    rows.append({"root": str(root), "rung": rung, "fold": fold, "arm": arm, "model": model,
                                 "window": int(window), "wql_contrib": 2 * loss / denominator,
                                 "coverage_10_90": cov, "coverage_01_99": cov99, "direction": direction,
                                 "up_rate": (overall["actual"] >= 0).mean()})
    if not rows:
        raise SystemExit("No walk-forward cells found under the given roots")
    return pd.DataFrame(rows)


def seed_mean_windows(df: pd.DataFrame, rung: str, fold: str, model: str) -> pd.Series | None:
    sub = df[(df.rung == rung) & (df.fold == fold) & (df.model == model)]
    if sub.empty:
        return None
    # average over seeds (and over duplicate pretrained evaluations) per window
    return sub.groupby("window")["wql_contrib"].mean().sort_index()


def paired(a: pd.Series | None, b: pd.Series | None, rng, draws: int = 5000):
    if a is None or b is None:
        return np.nan, np.nan, np.nan, np.nan
    d = (a - b).dropna().to_numpy()
    if len(d) == 0:
        return np.nan, np.nan, np.nan, np.nan
    boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(draws)]
    return d.mean(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), (d < 0).mean()


def fmt(x, digits=4):
    return "" if pd.isna(x) else f"{x:+.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/walk_forward"))
    parser.add_argument("--control", default="r3_qqq_calendar_market")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    df = load_runs(args.roots)
    df.to_csv(args.output_dir / "window_losses.csv", index=False)
    folds = sorted(df.fold.unique())
    rungs = [r for r in RUNG_ORDER if r in set(df.rung)] + sorted(set(df.rung) - set(RUNG_ORDER))
    seeds = sorted({a for a in df.arm.unique() if a.startswith("seed")})

    # --- headline WQL tables ---
    def table(model):
        t = pd.DataFrame(index=rungs, columns=folds, dtype=float)
        for r in rungs:
            for f in folds:
                s = seed_mean_windows(df, r, f, model)
                t.loc[r, f] = s.mean() if s is not None else np.nan
        t["mean"] = t[folds].mean(axis=1)
        return t
    wql_ft, wql_pt = table("fine_tuned"), table("base_pretrained")
    gauss = pd.Series({f: seed_mean_windows(df, rungs[0], f, "zero_gaussian").mean() for f in folds if seed_mean_windows(df, rungs[0], f, "zero_gaussian") is not None})

    # --- fine-tuning gain per rung/fold (fine_tuned - base_pretrained), pooled over folds too ---
    gain_rows, marg_rows, ctrl_rows = [], [], []
    for r in rungs:
        pooled_d = []
        for f in folds:
            ft, pt = seed_mean_windows(df, r, f, "fine_tuned"), seed_mean_windows(df, r, f, "base_pretrained")
            m, lo, hi, share = paired(ft, pt, rng)
            gain_rows.append({"rung": r, "fold": f, "delta_ft_minus_pt": m, "ci_lo": lo, "ci_hi": hi, "share_windows_better": share})
            if ft is not None and pt is not None:
                pooled_d.append((ft - pt).dropna().to_numpy())
        if pooled_d:
            d = np.concatenate(pooled_d); boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
            gain_rows.append({"rung": r, "fold": "pooled", "delta_ft_minus_pt": d.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5), "share_windows_better": (d < 0).mean()})
    for i, r in enumerate(rungs):
        for f in folds + ["pooled"]:
            if i > 0:
                if f == "pooled":
                    ds = [(seed_mean_windows(df, r, ff, "fine_tuned") - seed_mean_windows(df, rungs[i - 1], ff, "fine_tuned")).dropna().to_numpy()
                          for ff in folds if seed_mean_windows(df, r, ff, "fine_tuned") is not None and seed_mean_windows(df, rungs[i - 1], ff, "fine_tuned") is not None]
                    if not ds:
                        continue
                    d = np.concatenate(ds); boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
                    marg_rows.append({"rung": r, "previous": rungs[i - 1], "fold": f, "marginal_delta": d.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5), "share_windows_better": (d < 0).mean()})
                else:
                    m, lo, hi, share = paired(seed_mean_windows(df, r, f, "fine_tuned"), seed_mean_windows(df, rungs[i - 1], f, "fine_tuned"), rng)
                    marg_rows.append({"rung": r, "previous": rungs[i - 1], "fold": f, "marginal_delta": m, "ci_lo": lo, "ci_hi": hi, "share_windows_better": share})
            if r != args.control and args.control in rungs:
                if f == "pooled":
                    ds = [(seed_mean_windows(df, r, ff, "fine_tuned") - seed_mean_windows(df, args.control, ff, "fine_tuned")).dropna().to_numpy()
                          for ff in folds if seed_mean_windows(df, r, ff, "fine_tuned") is not None and seed_mean_windows(df, args.control, ff, "fine_tuned") is not None]
                    if not ds:
                        continue
                    d = np.concatenate(ds); boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
                    ctrl_rows.append({"rung": r, "fold": f, "delta_vs_control": d.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5), "share_windows_better": (d < 0).mean()})
                else:
                    m, lo, hi, share = paired(seed_mean_windows(df, r, f, "fine_tuned"), seed_mean_windows(df, args.control, f, "fine_tuned"), rng)
                    ctrl_rows.append({"rung": r, "fold": f, "delta_vs_control": m, "ci_lo": lo, "ci_hi": hi, "share_windows_better": share})
    gains, marg, ctrl = pd.DataFrame(gain_rows), pd.DataFrame(marg_rows), pd.DataFrame(ctrl_rows)
    gains.to_csv(args.output_dir / "finetune_gain.csv", index=False)
    marg.to_csv(args.output_dir / "marginal_group_effect.csv", index=False)
    ctrl.to_csv(args.output_dir / "delta_vs_control.csv", index=False)
    wql_ft.to_csv(args.output_dir / "wql_fine_tuned.csv"); wql_pt.to_csv(args.output_dir / "wql_pretrained.csv")

    # --- coverage / direction ---
    cov = df.groupby(["rung", "fold", "model"]).agg(coverage_10_90=("coverage_10_90", "mean"), coverage_01_99=("coverage_01_99", "mean"),
                                                   direction=("direction", "mean"), up_rate=("up_rate", "mean")).reset_index()
    cov.to_csv(args.output_dir / "coverage_direction.csv", index=False)

    # --- markdown ---
    def md(t: pd.DataFrame, digits=4):
        return t.round(digits).to_markdown()
    lines = [f"# Walk-forward covariate ladder\n", f"Folds: {', '.join(folds)}. Seeds present: {', '.join(seeds) or 'none'}. Control rung: `{args.control}`. Lower WQL is better.\n",
             "## Fine-tuned WQL (mean over seeds)\n", md(wql_ft), "\n## Pretrained WQL on the same windows\n", md(wql_pt),
             "\n## Zero-mean Gaussian baseline WQL by fold\n", md(gauss.to_frame("gaussian").T),
             "\n## Fine-tuning gain: fine_tuned minus pretrained (negative = fine-tuning helped)\n"]
    g = gains.copy(); g["cell"] = g.apply(lambda r: f"{fmt(r.delta_ft_minus_pt)} [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}]", axis=1)
    lines.append(md(g.pivot(index="rung", columns="fold", values="cell").reindex(rungs)[folds + ["pooled"]] if "pooled" in set(g.fold) else g.pivot(index="rung", columns="fold", values="cell").reindex(rungs)))
    if not marg.empty:
        lines.append("\n## Marginal effect of each added group on the fine-tuned model (rung minus previous rung)\n")
        m2 = marg.copy(); m2["cell"] = m2.apply(lambda r: f"{fmt(r.marginal_delta)} [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}]", axis=1)
        lines.append(md(m2.pivot(index="rung", columns="fold", values="cell").reindex([r for r in rungs if r in set(m2.rung)])))
    if not ctrl.empty:
        lines.append(f"\n## Fine-tuned delta versus the control rung\n")
        c2 = ctrl.copy(); c2["cell"] = c2.apply(lambda r: f"{fmt(r.delta_vs_control)} [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}]", axis=1)
        lines.append(md(c2.pivot(index="rung", columns="fold", values="cell").reindex([r for r in rungs if r in set(c2.rung)])))
    lines.append("\n## 10-90 coverage (target 0.80) and direction with the up-day base rate\n")
    c = cov[cov.model.isin(["fine_tuned", "base_pretrained"])].copy()
    lines.append(md(c.pivot_table(index=["rung", "model"], columns="fold", values="coverage_10_90").reindex(rungs, level=0), 3))
    lines.append("\nUp-day base rate per fold: " + ", ".join(f"{f} {cov[cov.fold == f].up_rate.iloc[0]:.1%}" for f in folds) + "\n")
    lines.append(md(c.pivot_table(index=["rung", "model"], columns="fold", values="direction").reindex(rungs, level=0), 3))
    (args.output_dir / "walk_forward_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(folds), figsize=(4.2 * len(folds), 4), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, f in zip(axes, folds):
            x = np.arange(len(rungs))
            ax.plot(x, wql_pt[f].to_numpy(), marker="o", label="pretrained", color="#7a8594")
            ax.plot(x, wql_ft[f].to_numpy(), marker="o", label="fine-tuned", color="#0e6b5e")
            if f in gauss:
                ax.axhline(gauss[f], ls=":", color="#9a4a3c", label="zero-mean Gaussian")
            ax.set_xticks(x); ax.set_xticklabels([r.split("_", 1)[0] for r in rungs], rotation=0)
            ax.set_title(f"fold {f}"); ax.grid(alpha=.3)
        axes[0].set_ylabel("WQL"); axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.output_dir / "walk_forward_wql.png", dpi=150)
    except Exception as exc:  # matplotlib optional
        print("figure skipped:", exc)
    print((args.output_dir / "walk_forward_summary.md").read_text())


if __name__ == "__main__":
    main()
