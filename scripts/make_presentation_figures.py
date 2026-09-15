#!/usr/bin/env python3
"""Presentation figures from the saved results under docs/results/.

    python scripts/make_presentation_figures.py            # writes docs/assets/presentation/*.png

Figures
  A  pretrained_importance.png   what the pretrained model does with each input (permutation test)
  B  finetuning_effect.png       importance before/after fine-tuning + coverage before/after
  C  ladder.png                  loss by feature group, pretrained vs fine-tuned, and fine-tuning gain
  D  direction_vs_base_rate.png  directional accuracy against the always-up base rate (backup slide)
  E  screen_vs_corrected.png     the earlier single-seed topic screen vs the corrected multi-seed result
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "assets" / "presentation"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical slots from the dataviz reference palette (light surface).
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})
NEWS = ("gdelt_", "epu_", "emu_", "fomc_", "sec_", "ndx_earnings")
CALENDAR = ("is_", "days_to")
PRETTY = {
    "vxn_log_chg": "VXN change", "overnight_gap": "Overnight gap", "volume_z_20d": "Volume z-score",
    "d_dfii10_bp": "Real yield change", "tlt_log_ret": "TLT return", "hyg_log_ret": "HYG return",
    "gdelt_fed_avg_tone": "GDELT Fed tone", "gdelt_fed_news_share": "GDELT Fed share",
    "days_to_fomc": "Days to FOMC", "is_fomc_day": "FOMC day", "epu_log": "EPU", "emu_log": "EMU",
    "fomc_net_hawkish_ewma": "Fed tone (EWMA)", "dist_200dma": "Distance to 200-day avg",
    "vix_term_ratio": "VIX term ratio", "d_dgs2_bp": "2y yield change", "mom_21d": "21-day momentum",
    "parkinson_vol_1d": "Range volatility", "cpi_yoy": "CPI yoy", "gdelt_recession_avg_tone": "GDELT recession tone",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


def group_of(name: str) -> str:
    if name.startswith(NEWS):
        return "news"
    if name.startswith(CALENDAR):
        return "calendar"
    return "market"


def ci_from_windows(path: Path, seed: int = 0) -> pd.DataFrame:
    w = pd.read_csv(path)
    rng = np.random.default_rng(seed)
    rows = []
    for f, g in w.groupby("feature"):
        d = g.groupby("window")["delta_wql"].mean().to_numpy()
        boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(3000)]
        rows.append((f, d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
    return pd.DataFrame(rows, columns=["feature", "mean", "lo", "hi"]).set_index("feature")


# ---------------------------------------------------------------- A
def figure_a() -> None:
    ci = ci_from_windows(RES / "permutation_pretrained" / "importance_by_window.csv").sort_values("mean")
    ci = ci * 100  # percent of loss units are small; show x100 for readability? keep WQL units x1000
    ci = ci / 100 * 1000  # -> "loss change x1000"
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(ci))
    colors = []
    for f, r in ci.iterrows():
        sig = r.lo > 0 or r.hi < 0
        base = RED if r["mean"] < 0 else BLUE
        colors.append(base if sig else "#c9c9c4")
    ax.barh(y, ci["mean"], color=colors, height=0.72)
    ax.errorbar(ci["mean"], y, xerr=[ci["mean"] - ci.lo, ci.hi - ci["mean"]], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
    ax.set_yticks(y)
    labels = []
    for f in ci.index:
        tag = {"news": "  [news]", "calendar": "  [calendar]", "market": ""}[group_of(f)]
        labels.append(pretty(f) + tag)
    ax.set_yticklabels(labels, fontsize=9)
    for t, f in zip(ax.get_yticklabels(), ci.index):
        if group_of(f) == "news":
            t.set_color(ORANGE)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("Change in loss when the input is swapped for another week's (x1000)")
    ax.set_title("Pretrained Chronos-2: which inputs does it actually use?\n42 inputs, 50 test weeks (Jul 2025 - Jun 2026), 3 repeats, 95% intervals\n"
                 "red = misleading (removing it helps), blue = useful, grey = no effect; orange labels = news inputs",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "A_pretrained_importance.png")
    plt.close(fig)


# ---------------------------------------------------------------- B
def figure_b() -> None:
    cell = "r7_all_external-fold2025-seed42-bw4"
    pre = ci_from_windows(RES / "permutation_finetuned" / "runs" / f"{cell}-pretrained" / "importance_by_window.csv")
    ft = ci_from_windows(RES / "permutation_finetuned" / "runs" / f"{cell}-finetuned" / "importance_by_window.csv", seed=1)
    order = pre["mean"].sort_values().index.tolist()
    keep = order[:6] + [f for f in order if group_of(f) == "news"][:6] + order[-3:]
    keep = list(dict.fromkeys(keep))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(keep))
    ax1.barh(y + 0.18, pre.loc[keep, "mean"] * 1000, height=0.34, color="#9a9a94", label="pretrained")
    ax1.barh(y - 0.18, ft.loc[keep, "mean"] * 1000, height=0.34, color=BLUE, label="fine-tuned (LoRA, 1,000 steps)")
    ax1.set_yticks(y); ax1.set_yticklabels([pretty(f) for f in keep], fontsize=9)
    for t, f in zip(ax1.get_yticklabels(), keep):
        if group_of(f) == "news":
            t.set_color(ORANGE)
    ax1.axvline(0, color=INK, linewidth=1)
    ax1.set_xlabel("Change in loss when input removed (x1000)")
    ax1.legend(frameon=False, loc="upper left", fontsize=9)
    corr_all = np.corrcoef(pre["mean"], ft.reindex(pre.index)["mean"])[0, 1]
    ax1.set_title(f"Fine-tuning does not change how inputs are used\nAll-external model, 2025, same test weeks; profile correlation over 42 inputs = {corr_all:.2f}", loc="left", fontsize=12)

    cov = pd.read_csv(RES / "walk_forward" / "coverage_direction.csv")
    cov = cov[cov.model.isin(["base_pretrained", "fine_tuned"])].groupby(["fold", "model"])["coverage_10_90"].mean().unstack()
    x = np.arange(len(cov.index))
    ax2.bar(x - 0.18, cov["base_pretrained"], width=0.34, color="#9a9a94", label="pretrained")
    ax2.bar(x + 0.18, cov["fine_tuned"], width=0.34, color=BLUE, label="fine-tuned")
    ax2.axhline(0.80, color=INK, linewidth=1, linestyle="--")
    ax2.text(len(x) - 0.5, 0.803, "target 80%", ha="right", va="bottom", fontsize=9, color=INK)
    ax2.set_xticks(x); ax2.set_xticklabels(cov.index)
    ax2.set_ylim(0.70, 0.85); ax2.set_ylabel("Share of outcomes inside the 10-90% band")
    ax2.set_title("What fine-tuning does change: calibration\nmean over all feature sets, 3 seeds", loc="left", fontsize=12)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    for i, (p, f) in enumerate(zip(cov["base_pretrained"], cov["fine_tuned"])):
        ax2.text(i - 0.18, p + 0.003, f"{p:.0%}", ha="center", fontsize=8.5, color=MUTED)
        ax2.text(i + 0.18, f + 0.003, f"{f:.0%}", ha="center", fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "B_finetuning_effect.png")
    plt.close(fig)


# ---------------------------------------------------------------- C
def figure_c() -> None:
    ft = pd.read_csv(RES / "walk_forward" / "wql_fine_tuned.csv", index_col=0)
    pt = pd.read_csv(RES / "walk_forward" / "wql_pretrained.csv", index_col=0)
    gain = pd.read_csv(RES / "walk_forward" / "finetune_gain.csv")
    rungs = ["r0_target", "r1_qqq", "r2_qqq_calendar", "r3_qqq_calendar_market", "r4_plus_uncertainty",
             "r5_plus_fed", "r6_plus_gdelt_fed_recession", "r7_all_external-bw4"]
    names = ["target only", "+ QQQ\ntransforms", "+ calendar", "+ market\n& rates", "+ EPU/EMU", "+ Fed tone", "+ GDELT\nFed/recession", "+ everything\n(42 inputs)"]
    pt_row = pt.copy(); pt_row.loc["r7_all_external-bw4"] = pt.loc["r7_all_external"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(rungs))
    ax1.plot(x, pt_row.loc[rungs, "mean"], marker="o", color="#9a9a94", linewidth=2, markersize=7, label="pretrained (no training)")
    ax1.plot(x, ft.loc[rungs, "mean"], marker="o", color=BLUE, linewidth=2, markersize=7, label="fine-tuned (3 seeds)")
    gauss_mean = float(pd.read_csv(RES / "walk_forward" / "window_losses.csv").query("model == 'zero_gaussian'").groupby("fold").wql_contrib.mean().mean())
    ax1.axhline(gauss_mean, color=ORANGE, linewidth=1.5, linestyle=":", label="zero-mean Gaussian baseline")
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Forecast loss (WQL), mean over 2022-2025")
    ax1.set_title("Adding feature groups: the loss barely moves\n4 validation years x 50 weeks each", loc="left", fontsize=12)
    ax1.legend(frameon=False, fontsize=9)
    lo, hi = min(ft.loc[rungs, "mean"].min(), pt_row.loc[rungs, "mean"].min()) - 0.003, max(ft.loc[rungs, "mean"].max(), pt_row.loc[rungs, "mean"].max(), gauss_mean) + 0.003
    ax1.set_ylim(lo, hi)

    g = gain[gain.fold == "pooled"].set_index("rung").loc[rungs]
    ax2.errorbar(g["delta_ft_minus_pt"] * 1000, x, xerr=[(g["delta_ft_minus_pt"] - g.ci_lo) * 1000, (g.ci_hi - g["delta_ft_minus_pt"]) * 1000],
                 fmt="o", color=BLUE, ecolor=MUTED, capsize=3, markersize=7)
    ax2.axvline(0, color=INK, linewidth=1)
    ax2.set_yticks(x); ax2.set_yticklabels([n.replace("\n", " ") for n in names], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Fine-tuned minus pretrained loss (x1000), 95% interval\nnegative = fine-tuning helped")
    ax2.set_title("Fine-tuning gain grows with inputs, but never clears zero\npooled over 4 years, 3 seeds", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "C_ladder.png")
    plt.close(fig)


# ---------------------------------------------------------------- D
def figure_d() -> None:
    cov = pd.read_csv(RES / "walk_forward" / "coverage_direction.csv")
    base = cov.groupby("fold")["up_rate"].first()
    models = cov[cov.model.isin(["base_pretrained", "fine_tuned"])].groupby(["fold", "model"])["direction"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(base.index))
    ax.bar(x - 0.27, base, width=0.26, color="#c9c9c4", label='"always up" (no model)')
    ax.bar(x, models["base_pretrained"], width=0.26, color="#9a9a94", label="pretrained")
    ax.bar(x + 0.27, models["fine_tuned"], width=0.26, color=BLUE, label="fine-tuned")
    ax.axhline(0.5, color=INK, linewidth=1, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(base.index)
    ax.set_ylim(0.35, 0.65); ax.set_ylabel("Share of days with the correct sign")
    ax.set_title("Direction: every model matches the up-day base rate\n(57% in a good year looks like skill; it is the base rate)", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for i, b in enumerate(base):
        ax.text(i - 0.27, b + 0.005, f"{b:.0%}", ha="center", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "D_direction_vs_base_rate.png")
    plt.close(fig)


# ---------------------------------------------------------------- E
def figure_e() -> None:
    screen = {"Federal Reserve": 1.28, "Recession": 1.24, "Semiconductors": 1.14, "AI": 1.13, "Inflation": 1.05, "All sources": 1.02, "Big-tech earnings": 0.96}
    marg = pd.read_csv(RES / "walk_forward" / "marginal_group_effect.csv")
    m = marg[(marg.rung == "r6_plus_gdelt_fed_recession") & (marg.fold == "pooled")].iloc[0]
    ctrl_mean = pd.read_csv(RES / "walk_forward" / "wql_fine_tuned.csv", index_col=0).loc["r5_plus_fed", "mean"]
    corrected = -m.marginal_delta / ctrl_mean * 100
    err = [[(m.ci_hi - m.marginal_delta) / ctrl_mean * 100], [(m.marginal_delta - m.ci_lo) / ctrl_mean * 100]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(screen))
    ax1.barh(y, list(screen.values()), color="#9a9a94", height=0.7)
    ax1.set_yticks(y); ax1.set_yticklabels(list(screen.keys()))
    ax1.invert_yaxis()
    ax1.set_xlabel("Loss reduction vs one seed-42 control run (%)")
    ax1.set_title("First screen: every topic 'helps' by the same ~1%\n1 seed, 1 year, 1 window per training step", loc="left", fontsize=12)
    for i, v in enumerate(screen.values()):
        ax1.text(v + 0.02, i, f"{v:.2f}%", va="center", fontsize=9, color=INK)
    ax1.set_xlim(0, 1.6)
    ax2.errorbar([corrected], [0], xerr=err, fmt="o", color=BLUE, ecolor=MUTED, capsize=4, markersize=8)
    ax2.axvline(0, color=INK, linewidth=1)
    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-1, 1); ax2.set_yticks([0]); ax2.set_yticklabels(["GDELT Fed +\nrecession"])
    ax2.set_xlabel("Loss reduction vs control (%), 95% interval")
    ax2.set_title("Corrected: 8 windows per step, 3 seeds, 4 years\nthe effect is zero", loc="left", fontsize=12)
    ax2.text(corrected, 0.25, f"{corrected:+.2f}%", ha="center", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "E_screen_vs_corrected.png")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (figure_a, figure_b, figure_c, figure_d, figure_e):
        fn()
        print("wrote", fn.__name__)
    print("figures in", OUT)
