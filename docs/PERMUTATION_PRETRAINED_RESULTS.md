# Permutation importance on the pretrained model: which inputs does Chronos-2 actually use?

Run 2026-09-15 on a Mac (MPS), inference only. Artifacts in
`docs/results/permutation_pretrained/` (`importance_by_window.csv`, `importance_summary.csv`,
`importance_ci.csv`, `summary/` with the ranking table, heatmap and per-horizon chart).
Regenerate with `scripts/permutation_importance.py` and `scripts/summarize_permutation_importance.py`.

## Method

Swap permutation, no retraining. For each of the 42 features in the all-external
configuration and each of the 50 validation windows in Elad's 2025-07 to 2026-06 sample, the
feature's context series (and its future block for known-future flags) is replaced by the
same feature taken from a different, randomly chosen window, the window is re-forecast, and
the change in weighted quantile loss is recorded. Three repeats with different donor
assignments. Importance is the mean loss increase when the feature's real values are removed:
positive means the model was using the feature helpfully, negative means the feature was
misleading it. Intervals are 5,000-draw paired bootstraps over windows.

Baseline: pretrained Chronos-2, revision `29ec3766`, context 512, horizon 5, the same 42
features and windows as the fine-tuning screen's `all_external` setup. Baseline WQL 0.6888.

## Result

Only three features have an effect that survives the paired test.

| Feature | Loss change when removed | 95% interval | Windows hurt | Read |
|---|---:|---|---:|---|
| gdelt_fed_avg_tone | +0.00175 | [+0.0002, +0.0036] | 64% | the one input the pretrained model uses helpfully |
| volume_z_20d | -0.00129 | [-0.0026, -0.0001] | 38% | mildly misleading |
| vxn_log_chg | -0.01380 | [-0.0273, -0.0012] | 38% | **actively misleading: removing it improves the loss by 2% of its value** |

Every other feature, including EPU, EMU, all FOMC tone columns, the calendar flags, rates,
TLT, HYG and the other ten GDELT columns, has a mean effect below 0.0015 in either direction
with an interval that includes zero.

The heatmap (`summary/permutation-pretrained_heatmap.png`) shows the VXN effect is not
uniform: it is concentrated in a minority of windows with large negative deltas, i.e. weeks
where the lagged volatility move was large and the model over-reacted to it.

## Findings

1. **Given the VXN change, the pretrained model is punished in volatility-shock weeks.** In the current table
   `vxn_log_chg` is lagged one session (Elad's conservative CBOE settlement rule), so on day T
   it carries the previous session's volatility move. Removing it improves the loss on
   average, which is the strongest single effect in the run; finding 5 below shows the effect
   lives in five volatility-shock weeks, so the mechanism is open. The same-session variant (`vxn_log_chg_t0`, added today as the `market_t0`
   group) was the direct test of whether the lag is to blame; the follow-up section below
   shows it is not.
2. **Fed news tone shows a trace.** GDELT's Fed-topic tone is the only input whose removal
   raises the loss at the 95% level (0.25% of the loss). The zero-shot ladder saw a similar
   hint from a *different* variable, the FOMC-statement classifier tone, so the two do not
   corroborate each other directly; with 42 tests this is a lead to check, not a finding
   (see finding 6).
3. **Most inputs are inert zero-shot.** Thirty-nine of forty-two features are
   indistinguishable from noise. This is consistent with the ladder: the pretrained model's
   in-context use of these covariates is weak, and any larger effect has to come from
   fine-tuning.
4. **Per-horizon** (`summary/permutation-pretrained_by_horizon.png`): the VXN effect is
   strongest at horizons 2 to 4 and near zero at horizon 5; the Fed-tone effect is concentrated
   at horizons 1 and 2.
5. **The VXN effect is five weeks, not fifty.** Its median across windows is -0.002 against a
   mean of -0.013; the five worst windows (origins 2025-07-28, 2025-08-18, 2025-10-07,
   2025-12-17, 2026-02-17) carry 94% of it, and all are weeks where realised volatility jumped
   two to three times above the trailing level. In calm weeks the effect is +0.0015 (nothing).
   The mechanism is therefore not established: the data fit "a calm VXN context narrows the
   bands, and an unforeseeable shock then punishes them" as well as "the model misreads VXN";
   a random donor series acts as a regulariser either way.
6. **Multiplicity.** Forty-two features tested at 95% yield about two false positives by
   chance, and three cleared the bar. The Fed-tone result (+0.00175) is the size of effect that
   multiplicity produces and should be reported as a lead, not a finding.

## Follow-up: is it the lag? (same day)

Two more runs on the 21-feature control alone, same windows and repeats
(`docs/results/permutation_control21/`, `docs/results/permutation_control21_t0/`): one with
the lagged VXN columns as in the table, one with the same-session `vxn_log_chg_t0` and
`vix_term_ratio_t0` swapped in.

| Feature | Lagged control (WQL 0.6943) | Same-session control (WQL 0.6924) |
|---|---:|---:|
| vxn_log_chg / vxn_log_chg_t0 | -0.0134 [-0.0264, -0.0017] | -0.0141 [-0.0279, -0.0017] |
| overnight_gap | -0.0073 [-0.0133, -0.0022] | -0.0070 [-0.0124, -0.0025] |
| d_dfii10_bp | -0.0046 [-0.0075, -0.0017] | -0.0016 [-0.0037, +0.0005] |
| tlt_log_ret | -0.0029 [-0.0056, -0.0001] | not significant |
| volume_z_20d | -0.0025 [-0.0047, -0.0006] | not significant |
| days_to_fomc | +0.0011 [-0.0009, +0.0034] | +0.0017 [-0.0001, +0.0037] |

(Negative: removing the feature improves the loss. Intervals are 95% paired bootstraps.)

**The lag is not the cause.** The same-session VXN change misleads the pretrained model
exactly as much as the lagged one. What the two runs show instead is that, zero-shot,
Chronos-2 is misled by every fast-moving market covariate in the control: the volatility
change, the overnight gap, the real-yield change, and to a lesser degree TLT and volume. The
model reads a large move in a covariate as a signal about the target's next values, and for
these series that mapping is wrong more often than right. The only inputs it uses helpfully
are the slow calendar structure, days-to-FOMC above all, and (in the 42-feature run) Fed news
tone.

This frames the fine-tuning question: either the pretrained model reads fast market
covariates wrongly, or their presence narrows its bands ahead of shocks it cannot foresee. In
both readings fine-tuning's job is to change how these inputs are used, not to add
information. The walk-forward ladder with the corrected batch size
will show whether 500 LoRA steps on a few thousand sessions are enough to do so, and the
permutation tool run on those checkpoints will show whether the sign of the VXN effect flips.

## Caveats

- One validation window (2025-07 to 2026-06) and one model state (pretrained). The same tool
  should be run on a fine-tuned checkpoint from the corrected-batch walk-forward ladder and
  across folds before the ranking is treated as general.
- Swap permutation removes the feature's information but keeps a realistic series shape;
  effects of correlated features (e.g. the six GDELT tones) are shared and can be
  understated individually.
- 50 windows give intervals about ±0.002 wide for well-behaved features; the VXN interval is
  wide because its effect is driven by a few high-volatility weeks.
