# Prefix Race Audit

Reference artifact for **What Winner Logs Cannot Tell You: Counterfactual Audits of First-Response Systems**.

A randomized completion prefix supports counterfactual comparisons of response eligibility in parallel inference. The theoretical contribution distinguishes what winner logs justify from what hidden completion orders would justify. This repository includes proofs, executable audit and allocation code, fitted predictions, latency traces, raw results, and four publication figures.

## Reproduce

Python 3.9 was used. No paid model services or external dataset download is required; the three datasets ship with scikit-learn. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run.py
python scripts/structural.py
python scripts/twins.py
python scripts/figures.py
```

The experiment script retrains 180 classifiers and overwrites the saved results. Numerical results can vary slightly across BLAS implementations. The recorded environment is in `results/environment.json`. The timing harness uses controlled sleeps and is not a provider-latency benchmark.

With [Tectonic](https://tectonic-typesetting.github.io/) installed, build the manuscript:

```sh
cd paper
tectonic manuscript.tex
tectonic named.tex
```

The anonymous and named versions have identical research content; the named version includes the author and public artifact URL. Official class files are distributed unmodified.

## Files

- `raceaudit/core.py`: observed-prefix records, inverse-inclusion estimates, full-table replay and variance coefficients.
- `raceaudit/design.py`: weighted monotone allocation and primal/dual optimization bounds.
- `raceaudit/runtime.py`: asyncio collector releasing the winner before the background audit finishes.
- `tests/test_research.py`: exact rational moment tests, exhaustive replay, independently formulated primal optimization and event-gated runtime test.
- `data/`: 30 prediction/split files and 90 engineered latency trace files.
- `results/experiment.json`: all 1,800 design evaluations, including failures.
- `results/monte_carlo.json` and `mc_*.npz`: 500 audit repetitions for each of 45 configurations.
- `docs/PROTOCOL.md`: primary protocol and separately recorded later theoretical extension.
- `docs/LITERATURE.md`: closest literature and scope of originality.
- `paper/`: manuscript, bibliography, table and official formatting classes.
- `figures/`: vector PDF and raster PNG versions of four figures.

## Interpretation

The target is response suppression while holding endpoint mechanisms fixed. It is not a causal estimate of physical removal when resource contention or endpoint behavior changes. Real classifier predictions are paired with engineered completion times, not production LLM traffic. Square-root allocation, inverse-inclusion weighting and pooled optimization primitives are established methods and are credited in the paper.

The main new results are the sharp winner-information minimax envelope and the identical-winner-log separation from the exchangeable-order allocation. The square-root loss ratio is an exact theorem, checked by small-case enumeration, rather than a large-scale model experiment. Pilot gains are modest and can reverse sharply with sparse data. Normal confidence intervals can undercover; only the stated conservative simultaneous bound is a finite-sample coverage guarantee.

Verification recorded eight passing unit tests, 5,376 rational moment cases, 32,896 replay cases, 100 independent optimization comparisons, and 120 controlled races with 4,920 policy checks. The largest randomized optimization verification gap was 6.51e-8.

## Rights and attribution

Original software is MIT licensed. Original manuscript text and generated results/figures are CC BY 4.0; see LICENSE. Third-party classes and bundled dataset origins retain their respective rights. See `docs/DATA.md`. This is a research manuscript, not an assertion of acceptance or publication.
