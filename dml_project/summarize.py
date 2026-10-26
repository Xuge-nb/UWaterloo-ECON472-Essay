"""
summarize.py -- aggregate replication-level results to grid-point tables.

Per (dgp, n, p, learner):
  bias, sd, RMSE of theta_hat; coverage with its Monte Carlo SE;
  mean SE and SE/sd ratio (is the variance estimate honest?); mean CI length;
  median test-sample nuisance errors; mean remainder proxy R_nuis;
  the exact second-order term in units of sd(theta_hat).
"""
import numpy as np
import pandas as pd

from config import THETA0

KEYS = ["dgp", "n", "p", "learner"]
LEARNER_ORDER = ["oracle", "linear", "spline", "nn"]


def summarize(df):
    def agg(g):
        R = len(g)
        sd = g["theta"].std(ddof=1)
        cov = g["cover"].mean()
        row = {
            "R": R,
            "bias": g["theta"].mean() - THETA0,
            "sd": sd,
            "rmse": np.sqrt(np.mean((g["theta"] - THETA0) ** 2)),
            "coverage": cov,
            "cov_mcse": np.sqrt(cov * (1 - cov) / R),
            "mean_se": g["se"].mean(),
            "se/sd": g["se"].mean() / sd,
            "ci_len": g["ci_len"].mean(),
            "sec/rep": g["seconds"].mean(),
        }
        if "err_m" in g and g["err_m"].notna().any():
            row.update({
                "err_m": g["err_m"].median(),
                "err_l": g["err_l"].median(),
                "R_nuis": g["R_nuis"].mean(),
                "2nd_order/sd": g["second_order"].mean() / sd,
            })
        return row

    out = pd.DataFrame([dict(zip(KEYS, k), **agg(g)) for k, g in df.groupby(KEYS)])
    out["learner"] = pd.Categorical(out["learner"], LEARNER_ORDER, ordered=True)
    return out.sort_values(KEYS).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot_raw.csv"
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summarize(pd.read_csv(path)).round(3).to_string(index=False))
