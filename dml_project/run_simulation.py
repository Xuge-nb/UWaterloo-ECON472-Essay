"""
run_simulation.py -- one replication at one grid point, and a grid driver.

    run_one(dgp, n, p, learner, seed, X_test)  ->  dict of results

For each replication it records
  * theta_hat, SE, 95% CI, coverage indicator, CI length       (DoubleML PLR)
  * true L2 nuisance errors ||m_hat - m0||, ||l_hat - l0||
        - on an independent test sample (eq. 12; average over the 5 fold models)
        - out-of-fold on the estimation sample (what actually enters the score)
  * remainder proxy  R_nuis = sqrt(n) * e_m * (e_m + e_l)      (eq. 12)
  * the exact second-order term of theta_hat - theta0, see `second_order_term`

Learner "oracle" plugs in the true l0, m0 (no first stage) as a benchmark.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

import doubleml as dml

from config import THETA0, N_FOLDS, N_TEST, TEST_SEED, MASTER_SEED
from dgp import simulate, true_nuisance, make_test_X, replication_seeds
from learners import get_learner

Z975 = 1.959963984540054
DGP_KEY = {"A": 0, "B": 1}


def l2(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def second_order_term(m_hat, l_hat, s, theta0=THETA0):
    """
    Exact decomposition for the partialling-out estimator.  With
    a = m0 - m_hat, b = l0 - l_hat, D~ = a + V and Y~ - theta0*D~ = b - theta0*a + U:

        theta_hat - theta0 = [ mean(V*U)                  (oracle, first order)
                             + mean(a*U) + mean(V*(b - theta0*a))  (mean-zero cross terms)
                             + mean(a*(b - theta0*a)) ]   (second-order remainder)
                             / mean(D~^2)

    Returns the last piece's contribution to theta_hat - theta0.
    """
    a = s.m0 - m_hat
    b = s.l0 - l_hat
    Dt = s.D - m_hat
    return float(np.mean(a * (b - theta0 * a)) / np.mean(Dt ** 2))


def oracle_fit(s):
    Dt, Yt = s.D - s.m0, s.Y - s.l0
    th = np.sum(Dt * Yt) / np.sum(Dt ** 2)
    psi = (Yt - th * Dt) * Dt
    se = np.sqrt(np.mean(psi ** 2) / np.mean(Dt ** 2) ** 2 / len(Dt))
    return th, se


def run_one(dgp, n, p, learner, seed, X_test=None, learner_kwargs=None):
    t0 = time.perf_counter()
    s = simulate(dgp, n, p, seed)
    out = {"dgp": dgp, "n": n, "p": p, "learner": learner, "seed": seed}

    if learner == "oracle":
        th, se = oracle_fit(s)
        out.update(theta=th, se=se)
    else:
        kw = dict(learner_kwargs or {})
        if learner == "nn":
            kw.setdefault("random_state", seed % (2 ** 31))
        ml_l = get_learner(learner, **kw)
        ml_m = get_learner(learner, **kw)

        data = dml.DoubleMLData.from_arrays(s.X, s.Y, s.D)
        model = dml.DoubleMLPLR(data, ml_l, ml_m, n_folds=N_FOLDS,
                                score="partialling out", draw_sample_splitting=False)
        # explicit, seeded splits -> fully reproducible and reusable for ablations
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed % (2 ** 31))
        model.set_sample_splitting([(tr, te) for tr, te in folds.split(s.X)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(store_predictions=True, store_models=True)

        th, se = float(model.coef[0]), float(model.se[0])
        out.update(theta=th, se=se)
        l_hat = model.predictions["ml_l"][:, 0, 0]     # out-of-fold predictions
        m_hat = model.predictions["ml_m"][:, 0, 0]

        # out-of-fold errors on the estimation sample
        out["err_l_oof"] = l2(l_hat, s.l0)
        out["err_m_oof"] = l2(m_hat, s.m0)
        out["second_order"] = second_order_term(m_hat, l_hat, s)

        # independent-test-sample errors, one per fold model, pooled in mean square
        if X_test is not None:
            tn = true_nuisance(dgp, X_test)
            fold_models = {k: model.models[k]["d"][0] for k in ("ml_l", "ml_m")}
            out["err_l"] = float(np.sqrt(np.mean(
                [np.mean((f.predict(X_test) - tn["l0"]) ** 2) for f in fold_models["ml_l"]])))
            out["err_m"] = float(np.sqrt(np.mean(
                [np.mean((f.predict(X_test) - tn["m0"]) ** 2) for f in fold_models["ml_m"]])))
            out["R_nuis"] = np.sqrt(n) * out["err_m"] * (out["err_m"] + out["err_l"])

    out["ci_lo"], out["ci_hi"] = out["theta"] - Z975 * out["se"], out["theta"] + Z975 * out["se"]
    out["cover"] = int(out["ci_lo"] <= THETA0 <= out["ci_hi"])
    out["ci_len"] = out["ci_hi"] - out["ci_lo"]
    out["seconds"] = time.perf_counter() - t0
    return out


def run_grid(dgps, ns, ps, learners, R, n_jobs=-1, out_csv=None,
             master_seed=MASTER_SEED, verbose=5):
    """
    Run R replications at every (dgp, n, p, learner) combination.

    All learners at the same (dgp, n, p, r) see the *same* simulated sample
    (common random numbers), so learner comparisons are paired.
    """
    tasks = []
    for dgp in dgps:
        for p in ps:
            X_test = make_test_X(N_TEST, p, TEST_SEED)
            for n in ns:
                seeds = replication_seeds(master_seed, R, DGP_KEY[dgp], n, p)
                for learner in learners:
                    tasks += [(dgp, n, p, learner, sd, X_test) for sd in seeds]
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(run_one)(*t) for t in tasks)
    df = pd.DataFrame(res)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
