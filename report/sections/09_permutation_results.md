# 8  Input perturbation and predictive reliance

The swap test replaces one feature's context with the same feature from another evaluation window and re-forecasts without retraining. For known-future calendar inputs it also replaces the future calendar block. Three donor assignments are averaged. The loss difference is scrambled minus original: a positive value indicates that the original alignment was helpful under this intervention; a negative value indicates that the swap improved the score.

This is an offline diagnostic. Donor windows can be later than the recipient origin, so the perturbed series is not a feasible historical trading input. Its history retains temporal structure but loses its original relationship to the target and other covariates. It is not literal feature removal, a causal intervention on the market, or a direct measure of internal semantic understanding.

::: figure ../figures/permutation.png
Selected features of the 42-covariate pretrained model in the original evaluation interval. Means and exploratory 95% window-bootstrap intervals use locally normalized window WQL, as in the saved swap experiment. These values are not directly interchangeable with the fold-normalized ladder deltas.
:::

Swapping VXN change lowers mean locally normalized WQL by 0.0134 in the control. Its median effect is much smaller (-0.00234), and five windows account for approximately 94% of the signed total. In the 42-feature setup, the mean effect is -0.0138. A same-session VXN variant has a similar effect, which weakens the hypothesis that a one-session lag alone explains the result. The perturbation may alter forecast dispersion ahead of shocks; the evidence does not establish that the model reads volatility backwards.

## 8.1  News effects and adaptation

The 42-feature pretrained test has one small positive news result: GDELT Fed average tone, +0.00175 WQL, with an unadjusted interval excluding zero. Forty-two feature tests create a multiple-comparison problem. This is also a different variable from the FOMC-statement classifier used in another ladder arm, so those two findings do not independently confirm the same signal. Correlated features can share information and make individual perturbations appear weak.
