"""
check_dgp.py -- unit tests and signal-strength checks for the two DGPs.

Run:  python check_dgp.py

Part 1  correctness checks (assertions; the script stops if any fail)
Part 2  signal strength: how much of Var(D) and Var(Y) the nuisances explain
Part 3  structural match: how much of m0 / l0 a linear or an additive model
        can represent at all (population projections), and the implied
        floor on the DML second-order remainder for a learner stuck at that
        approximation error
Part 4  oracle DML (true l0, m0): bias, SE and coverage with no first-stage
        error, i.e. the best any learner could do
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from config import THETA0, Q_RELEVANT, N_GRID
from dgp import simulate, toeplitz_cov, true_nuisance, replication_seeds

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


# ============================================================ Part 1: tests
def run_unit_tests():
    # reproducibility
    a = simulate("A", 200, 10, 123)
    b = simulate("A", 200, 10, 123)
    assert np.array_equal(a.Y, b.Y) and np.array_equal(a.X, b.X), "seed not reproducible"

    for dgp in ["A", "B"]:
        for p in [10, 50]:
            s = simulate(dgp, 300, p, 1)
            assert s.X.shape == (300, p) and s.D.shape == (300,) and s.Y.shape == (300,)
            # model identities
            assert np.allclose(s.D, s.m0 + s.V)
            assert np.allclose(s.Y, THETA0 * s.D + s.g0 + s.U)
            assert np.allclose(s.l0, THETA0 * s.m0 + s.g0)
            assert np.allclose(s.Y - s.l0, THETA0 * s.V + s.U)   # Y - E[Y|X]

        # irrelevant covariates really are irrelevant
        s = simulate(dgp, 100, 50, 7)
        X2 = s.X.copy()
        X2[:, Q_RELEVANT:] = np.random.default_rng(0).standard_normal((100, 50 - Q_RELEVANT))
        tn = true_nuisance(dgp, X2)
        assert np.allclose(tn["m0"], s.m0) and np.allclose(tn["g0"], s.g0)

    # covariance structure and mean-zero centring of X2^2 - 1
    big = simulate("A", 400_000, 10, 99)
    S_hat = np.cov(big.X, rowvar=False)
    assert np.max(np.abs(S_hat - toeplitz_cov(10))) < 0.01, "Sigma mismatch"
    assert abs(np.mean(big.X[:, 1] ** 2 - 1)) < 0.01
    assert abs(np.mean(big.V)) < 0.01 and abs(np.var(big.V) - 1) < 0.01

    # different grid points get different seed streams; same key -> same seeds
    s1 = replication_seeds(472, 5, 0, 500, 10)
    s2 = replication_seeds(472, 5, 0, 500, 10)
    s3 = replication_seeds(472, 5, 1, 500, 10)
    assert s1 == s2 and s1 != s3 and len(set(s1)) == 5
    print("Part 1: all unit tests passed.\n")


# ============================================================ Part 2 + 3
def r2_of_projection(model, Xtr, ftr, Xte, fte):
    """Share of Var(f) captured by the best fit in `model`'s class (noise-free target)."""
    model.fit(Xtr, ftr)
    resid = fte - model.predict(Xte)
    return 1 - resid.var() / fte.var(), np.sqrt(np.mean(resid ** 2))


def signal_and_structure(N=200_000, p=10, seed=2026):
    rows = []
    for dgp in ["A", "B"]:
        s = simulate(dgp, N, p, seed)
        half = N // 2
        Xr = s.X[:, :Q_RELEVANT]          # give the projections the oracle variable set
        row = {
            "DGP": dgp,
            "Var(m0)": s.m0.var(), "Var(g0)": s.g0.var(), "Var(l0)": s.l0.var(),
            "Var(D)": s.D.var(), "Var(Y)": s.Y.var(),
            "R2(D|X)": s.m0.var() / s.D.var(),
            "R2(Y|X)": s.l0.var() / s.Y.var(),
            "E[m0]": s.m0.mean(), "E[g0]": s.g0.mean(),
            "Corr(m0,g0)": np.corrcoef(s.m0, s.g0)[0, 1],
        }
        for name, f in [("m0", s.m0), ("l0", s.l0)]:
            lin = LinearRegression()
            add = make_pipeline(
                SplineTransformer(n_knots=12, degree=3, knots="quantile", extrapolation="linear"),
                RidgeCV(alphas=[1e-6, 1e-4, 1e-2]))
            r2_lin, err_lin = r2_of_projection(lin, Xr[:half], f[:half], Xr[half:], f[half:])
            r2_add, err_add = r2_of_projection(add, Xr[:half], f[:half], Xr[half:], f[half:])
            row[f"linR2({name})"] = r2_lin
            row[f"addR2({name})"] = r2_add
            row[f"linErr({name})"] = err_lin
            row[f"addErr({name})"] = err_add
        rows.append(row)
    df = pd.DataFrame(rows).set_index("DGP")

    # Floor on the remainder proxy  sqrt(n) * e_m * (e_m + e_l)  for a learner
    # whose L2 error cannot go below its population approximation error.
    for cls in ["lin", "add"]:
        em, el = df[f"{cls}Err(m0)"], df[f"{cls}Err(l0)"]
        for n in N_GRID:
            df[f"R_floor_{cls}_n{n}"] = np.sqrt(n) * em * (em + el)
    return df


# ============================================================ Part 4: oracle
def oracle_dml(dgp, n, p, R=1000, master_seed=99):
    """Partialling-out estimator with the TRUE l0, m0 (no first-stage error)."""
    est, se, cover = [], [], []
    for seed in replication_seeds(master_seed, R, ord(dgp), n, p):
        s = simulate(dgp, n, p, seed)
        Dt, Yt = s.D - s.m0, s.Y - s.l0
        th = np.sum(Dt * Yt) / np.sum(Dt ** 2)
        J = np.mean(Dt ** 2)
        psi = (Yt - th * Dt) * Dt
        sd = np.sqrt(np.mean(psi ** 2) / J ** 2 / n)
        est.append(th); se.append(sd)
        cover.append(abs(th - THETA0) <= 1.959964 * sd)
    est, se = np.array(est), np.array(se)
    return {"DGP": dgp, "n": n, "p": p, "bias": est.mean() - THETA0,
            "sd(theta_hat)": est.std(ddof=1), "mean SE": se.mean(),
            "coverage": np.mean(cover)}


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.precision", 4)
    run_unit_tests()

    sig = signal_and_structure()
    sig.to_csv(f"{OUT}/dgp_signal_structure.csv")
    print("Part 2: signal strength (N = 200,000, p = 10)")
    print(sig[["Var(m0)", "Var(g0)", "Var(l0)", "R2(D|X)", "R2(Y|X)",
               "E[m0]", "E[g0]", "Corr(m0,g0)"]].T, "\n")
    print("Part 3: share of nuisance variance a model class can represent "
          "(population projection on X1..X4), and its L2 approximation error")
    print(sig[[c for c in sig if c.startswith(("lin", "add"))]].T, "\n")
    print("Implied floor on sqrt(n)*e_m*(e_m+e_l) from approximation error alone")
    print(sig[[c for c in sig if c.startswith("R_floor")]].T, "\n")

    orc = pd.DataFrame([oracle_dml(d, n, 10) for d in ["A", "B"] for n in N_GRID])
    orc.to_csv(f"{OUT}/oracle_check.csv", index=False)
    print("Part 4: oracle DML with true nuisances (R = 1000, p = 10)")
    print(orc.to_string(index=False))
