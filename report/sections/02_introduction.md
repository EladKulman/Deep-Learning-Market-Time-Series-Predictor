# 1  Introduction

The project asks whether broad news narratives supply information that is not already present in market prices and scheduled events. We use QQQ, an exchange-traded fund tracking the Nasdaq-100, as a single, consistently observed target. The objective is to evaluate daily return distributions and source contributions, rather than to attribute an index move to one headline.

The research questions are: **RQ1**, do news-derived features add value beyond a fixed market and calendar control? **RQ2**, does LoRA adaptation improve the same model supplied with the same features? **RQ3**, do input perturbations reveal a change in predictive reliance after adaptation? An accurate return distribution can represent uncertainty even when its median offers little directional discrimination.

Our contribution is an empirical study with explicit information-availability rules, a first-round source screen, a larger chronological evaluation, and diagnostic analysis of the saved forecasts. We use an existing foundation model and adaptation method. The report develops the experiments in our project presentation, with the distinction between promising initial rankings and evidence that survives broader evaluation made explicit.
