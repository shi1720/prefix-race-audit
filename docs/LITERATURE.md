# Literature and originality boundaries

Search date: 12 October 2026. Searches covered first-response cancellation, censored algorithm portfolios, competing risks, randomized truncation, multivariate allocation, cascading feedback and direct policy evaluation. This record supports scoped claims, not an exhaustive proof that no related result exists.

Closest established primitives:

- Dean and Barroso (2013), The Tail at Scale. Hedging and cancellation are established system mechanisms. https://doi.org/10.1145/2408776.2408794
- Tsiatis (1975), A Nonidentifiability Aspect of the Problem of Competing Risks. Missing alternatives are not generally identified from first events. https://doi.org/10.1073/pnas.72.1.20
- Gagliolo and Schmidhuber (2006), Learning Dynamic Algorithm Portfolios. Censored parallel runs and learned runtime distributions. https://doi.org/10.1007/s10472-006-9036-z
- Hutter et al. (2014), Algorithm Runtime Prediction: Methods & Evaluation. Censored runtime modeling. https://doi.org/10.1016/j.artint.2013.10.003
- Horvitz and Thompson (1952), unequal-probability unbiased estimation. https://doi.org/10.1080/01621459.1952.10483446
- Robins, Rotnitzky and Zhao (1994), missing-data inverse-probability methodology. https://doi.org/10.1080/01621459.1994.10476818
- Lyne et al. (2015), Russian roulette estimates. https://doi.org/10.1214/15-STS523
- Beatson and Adams (2019), randomized telescoping, unbiased truncation and adaptive allocation. https://proceedings.mlr.press/v97/beatson19a.html
- Cui, Lee, Lingjiong Zhu and Yunfan Zhu (2021), optimal randomized truncation, monotone allocation algorithms. https://doi.org/10.1016/j.orl.2021.05.004
- Bethel (1989), multivariate sample allocation. https://www150.statcan.gc.ca/n1/pub/12-001-x/1989001/article/14578-eng.pdf
- Kveton et al. (2015), Cascading Bandits. Prefix feedback for ranking, a different selection mechanism and target. https://proceedings.mlr.press/v37/kveton15.html
- Guerdan et al. (2024), Predictive Performance Comparison of Training Interventions in the Presence of Selective Labels. Direct contrasts exploit policy agreement. https://proceedings.mlr.press/v235/guerdan24a.html

The manuscript credits the inverse-inclusion identity, square-root allocation, pooling and convex minimax primitives. Its specific claim concerns the sharp worst-case audit variance compatible with winner logs and an exact separation between identical first-event laws with different hidden order structures. It restricts optimality to rank-only designs and the specified estimator. Response suppression is defined structurally; physical outages are not identified without an additional invariance argument.

The older manuscripts supplied for overlap screening concern different targets, including coherent replay coupling, ranking, cache effects and intervention contracts. Their broad causal vocabulary is not treated as a contribution here. This paper uses independent code, experiments and estimands.
