# ECON472 — DML nuisance structure Monte Carlo

Python code for the partially linear model experiment in the research proposal.

## Install

```
pip install numpy pandas scikit-learn matplotlib joblib doubleml
```
Tested with Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, DoubleML 0.11.4.

## Files

| File | What it does |
|---|---|
| `config.py` | All seeds, grid points, replication counts and learner hyperparameters |
| `dgp.py` | DGP A (smooth additive) and DGP B (interactive/composite); true `m0`, `g0`, `l0 = θ0·m0 + g0`; seed streams |
| `check_dgp.py` | Unit tests, signal strength, how much an additive/linear model can represent, oracle DML check |
| `learners.py` | Linear, additive cubic spline + RidgeCV, small ReLU net (tuned inside training folds) |
| `run_simulation.py` | `run_one()` = one replication via `DoubleMLPLR`; nuisance L2 errors on a test sample; remainder proxy; `run_grid()` |
| `summarize.py` | Grid-point table: bias, RMSE, coverage ± MC SE, SE/sd, CI length, nuisance errors, `R_nuis` |
| `pilot.py` | Pilot driver over the 8 core grid points |
| `figures.py` | Fig 1 coverage, Fig 2 nuisance errors, Fig 3 coverage vs `R_nuis` |

## Run order

```
python check_dgp.py          # ~10 s, must print "all unit tests passed"
python pilot.py 5 --quick    # smoke test, n=500, p=10 only
python pilot.py              # R = 50 pilot over all 8 grid points
python figures.py            # figures from outputs/pilot_summary.csv
```

## Design notes

* **Common random numbers.** At a given (DGP, n, p, replication) all learners see the
  same simulated sample and the same 5-fold split, so learner comparisons are paired.
* **No leakage.** Scaling, spline knots, Ridge penalty, NN tuning and early stopping are
  all inside the estimator, which DoubleML refits on each training fold.
* **Two error measures.** `err_m`, `err_l` use an independent test sample (N = 20,000,
  fixed seed) averaged over the 5 fold models — this is eq. (12). `err_*_oof` are the
  out-of-fold errors on the estimation sample, which are what actually enter the score.
* **`second_order`** is the exact remainder in θ̂ − θ0 for the partialling-out
  estimator, mean[a(b − θ0 a)] / mean[D̃²] with a = m0 − m̂, b = l0 − l̂. Divided by
  sd(θ̂) it is the bias in standard-error units that the remainder creates; compare it
  with the `R_nuis` upper-bound proxy.
* **NN penalty grid.** sklearn scales the L2 penalty by 1/n, so α ≤ 0.01 is effectively
  unregularised and the network overfits the weak-signal DGP B. The grid is
  α ∈ {1, 3, 10, 30, 100}, chosen by 3-fold CV inside each training fold.
