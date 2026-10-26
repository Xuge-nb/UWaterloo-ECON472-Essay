"""
pilot.py -- small pilot run over the 8 core grid points (Section 5.3 / week 5).

    python pilot.py            # R = config.R_PILOT (50)
    python pilot.py 20         # custom R
    python pilot.py 5 --quick  # unit-test size, n = 500 and p = 10 only

Writes outputs/pilot_raw.csv (one row per replication x learner) and
outputs/pilot_summary.csv, and prints the summary table.
"""
import os
import sys
import time
import warnings

# one BLAS thread per worker; parallelism is across replications
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")   # silence MLP warnings in workers
warnings.filterwarnings("ignore")

import pandas as pd

from config import DGPS, N_GRID, P_GRID, R_PILOT
from run_simulation import run_grid
from summarize import summarize

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else R_PILOT
    quick = "--quick" in sys.argv
    ns, ps = ([500], [10]) if quick else (N_GRID, P_GRID)
    learners = ["oracle", "linear", "spline", "nn"]

    t0 = time.time()
    raw = run_grid(DGPS, ns, ps, learners, R, n_jobs=-1, out_csv="outputs/pilot_raw.csv")
    summ = summarize(raw)
    summ.to_csv("outputs/pilot_summary.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summ.round(3).to_string(index=False))
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
