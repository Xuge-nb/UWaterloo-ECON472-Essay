"""
config.py -- single source of truth for the Monte Carlo design.

Every random seed, hyperparameter grid and grid-point definition lives here
(Section 7 of the proposal), so that results can be reproduced exactly.
"""

THETA0 = 0.5          # true target parameter
Q_RELEVANT = 4        # number of covariates the true functions depend on
RHO = 0.5             # Sigma_jk = RHO ** |j - k|

DGPS = ["A", "B"]     # A = smooth additive, B = low-effective-dim interactive/composite
N_GRID = [500, 2000]
P_GRID = [10, 50]
N_FOLDS = 5

# Replications
R_UNIT = 5
R_PILOT = 50
R_MAIN = 300

# Independent test sample used to approximate true L2 nuisance errors
N_TEST = 20_000
TEST_SEED = 20261016   # fixed, so every replication is scored on the same test points

# Master seed for the simulation; per-replication seeds are spawned from it
MASTER_SEED = 472

# ---------------------------------------------------------------- learners
SPLINE = dict(
    n_knots=6,            # interior knots per covariate (quantile-placed)
    degree=3,             # cubic
    ridge_alphas=[0.01, 0.1, 1.0, 10.0, 100.0],  # chosen by RidgeCV inside training folds
)

NEURAL_NET = dict(
    hidden_layer_sizes=(32, 32),
    max_iter=500,
    validation_fraction=0.2,   # early-stopping hold-out, taken from the training folds
    n_iter_no_change=25,
    # Very small pre-specified grid, tuned by inner 3-fold CV within each training fold.
    # NOTE: sklearn scales the L2 penalty as alpha / (2 * n_train) * ||W||^2, so values
    # like 1e-4 are effectively unregularised. With R2(D|X) ~ 0.09 in DGP B the network
    # then fits noise and does worse than a constant. A preliminary check (16 Oct 2026)
    # selected alpha in [3, 30], larger for p = 50, hence this grid.
    param_grid={
        "alpha": [1.0, 3.0, 10.0, 30.0, 100.0],
        "learning_rate_init": [1e-3],        # fixed; add values here to tune it too
    },
    default_alpha=10.0,        # used when tune=False
    inner_cv=3,
    tune=True,
)
