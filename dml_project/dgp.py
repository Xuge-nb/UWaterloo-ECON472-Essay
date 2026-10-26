"""
dgp.py -- data generating processes for the partially linear model (Section 5).

    X ~ N(0, Sigma),  Sigma_jk = 0.5^|j-k|
    D = m0(X) + V,    V ~ N(0, 1)
    Y = theta0 * D + g0(X) + U,   U ~ N(0, 1)

Only X_1..X_4 enter m0 and g0; raising p adds irrelevant covariates only.
The true nuisance functions m0, g0 and l0 = E[Y|X] = theta0*m0 + g0 are exposed
as plain functions of X so they can be evaluated on an independent test set.
"""
from dataclasses import dataclass

import numpy as np

from config import THETA0, RHO, Q_RELEVANT


# --------------------------------------------------------------------------
# True nuisance functions.  Columns are 0-indexed: X[:, 0] is X_1 in the paper.
# --------------------------------------------------------------------------
def m0_A(X):
    """DGP A, eq. (8): smooth additive."""
    return (0.5 * np.sin(X[:, 0]) + 0.25 * (X[:, 1] ** 2 - 1)
            + 0.25 * X[:, 2] + 0.20 * np.tanh(X[:, 3]))


def g0_A(X):
    """DGP A, eq. (9): smooth additive."""
    return (np.sin(X[:, 0]) + 0.50 * (X[:, 1] ** 2 - 1)
            + 0.50 * X[:, 2] + 0.30 * np.tanh(X[:, 3]))


def m0_B(X):
    """DGP B, eq. (10): interaction X1*X2 and composite single index X3+X4."""
    return 0.5 * np.tanh(X[:, 0] * X[:, 1]) + 0.25 * np.tanh(X[:, 2] + X[:, 3])


def g0_B(X):
    """DGP B, eq. (11)."""
    return np.sin(X[:, 0] * X[:, 1]) + 0.50 * np.tanh(X[:, 2] + X[:, 3]) ** 2


NUISANCE = {
    "A": (m0_A, g0_A),
    "B": (m0_B, g0_B),
}


def true_nuisance(dgp, X, theta0=THETA0):
    """Return dict with the true m0(X), g0(X) and l0(X) = theta0*m0(X) + g0(X)."""
    m_fun, g_fun = NUISANCE[dgp]
    m = m_fun(X)
    g = g_fun(X)
    return {"m0": m, "g0": g, "l0": theta0 * m + g}


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def toeplitz_cov(p, rho=RHO):
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def draw_X(n, p, rng, rho=RHO):
    """X ~ N(0, Sigma) via Cholesky; Sigma is AR(1)-Toeplitz."""
    if p < Q_RELEVANT:
        raise ValueError(f"p must be at least {Q_RELEVANT}")
    L = np.linalg.cholesky(toeplitz_cov(p, rho))
    return rng.standard_normal((n, p)) @ L.T


@dataclass
class SimData:
    X: np.ndarray
    D: np.ndarray
    Y: np.ndarray
    m0: np.ndarray   # true m0(X_i)
    g0: np.ndarray   # true g0(X_i)
    l0: np.ndarray   # true l0(X_i) = E[Y | X_i]
    V: np.ndarray    # treatment noise (kept for diagnostics)
    U: np.ndarray    # outcome noise
    dgp: str
    theta0: float


def simulate(dgp, n, p, rng, theta0=THETA0):
    """
    Draw one sample of size n from DGP 'A' or 'B' with ambient dimension p.

    rng : np.random.Generator (or an int seed).
    """
    if dgp not in NUISANCE:
        raise ValueError(f"dgp must be one of {list(NUISANCE)}")
    rng = np.random.default_rng(rng)
    X = draw_X(n, p, rng)
    V = rng.standard_normal(n)
    U = rng.standard_normal(n)
    tn = true_nuisance(dgp, X, theta0)
    D = tn["m0"] + V
    Y = theta0 * D + tn["g0"] + U
    return SimData(X=X, D=D, Y=Y, m0=tn["m0"], g0=tn["g0"], l0=tn["l0"],
                   V=V, U=U, dgp=dgp, theta0=theta0)


def make_test_X(n_test, p, seed):
    """Independent covariate sample for approximating L2(P_X) nuisance errors."""
    return draw_X(n_test, p, np.random.default_rng(seed))


def replication_seeds(master_seed, n_reps, *keys):
    """
    Reproducible, statistically independent seeds for each replication.
    `keys` (e.g. dgp, n, p as ints) make different grid points use different
    streams, so adding a grid point never changes the draws of another one.
    """
    ss = np.random.SeedSequence([master_seed, *keys])
    return [int(c.generate_state(1)[0]) for c in ss.spawn(n_reps)]
