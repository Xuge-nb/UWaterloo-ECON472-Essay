"""
figures.py -- core figures (Section 5.5) from a summary CSV.

    python figures.py                           # uses outputs/pilot_summary.csv
    python figures.py outputs/main_summary.csv  # later, for the main run

Fig 1  coverage by learner across DGPs and (n, p), with Monte Carlo 95% bands
Fig 2  test-sample nuisance errors ||m_hat - m0||, ||l_hat - l0||
Fig 3  coverage against the remainder proxy R_nuis  (hypothesis H1)
"""
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed learner -> style mapping; colour follows the learner in every figure.
# Marker shapes double-encode identity for greyscale printing.
STYLE = {
    "oracle": dict(color="#6b6a64", marker="D", label="Oracle (true nuisances)"),
    "linear": dict(color="#2a78d6", marker="o", label="Linear"),
    "spline": dict(color="#eb6834", marker="s", label="Additive spline"),
    "nn":     dict(color="#1baf7a", marker="^", label="ReLU network"),
}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def grid_labels(df):
    cells = df[["n", "p"]].drop_duplicates().sort_values(["n", "p"])
    return [(r.n, r.p, f"n={r.n}\np={r.p}") for r in cells.itertuples()]


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def fig_coverage(s, out):
    cells = grid_labels(s)
    learners = [l for l in STYLE if l in set(s.learner)]
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, dgp in zip(axes, ["A", "B"]):
        for j, lrn in enumerate(learners):
            st = STYLE[lrn]
            x, y, e = [], [], []
            for i, (n, p, _) in enumerate(cells):
                r = s[(s.dgp == dgp) & (s.n == n) & (s.p == p) & (s.learner == lrn)]
                if len(r):
                    x.append(i + (j - (len(learners) - 1) / 2) * width)
                    y.append(r.coverage.iloc[0]); e.append(1.96 * r.cov_mcse.iloc[0])
            y, e = np.array(y), np.array(e)
            yerr = np.vstack([np.minimum(e, y), np.minimum(e, 1 - y)])   # clip to [0, 1]
            ax.errorbar(x, y, yerr=yerr, fmt=st["marker"], color=st["color"], ms=5,
                        elinewidth=1, capsize=0, label=st["label"])
        ax.axhline(0.95, color=INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(cells)), [c[2] for c in cells])
        ax.set_title({"A": "DGP A: smooth additive", "B": "DGP B: interactive/composite"}[dgp],
                     loc="left", fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Coverage of 95% CI")
    axes[0].set_ylim(-0.03, 1.03)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(learners), fontsize=8,
               bbox_to_anchor=(0.5, -0.1))
    fig.text(0.01, -0.16, "Bars: Monte Carlo 95% interval for the coverage rate. "
             "Dashed line: nominal 0.95.", fontsize=7.5, color=MUTED)
    _save(fig, out, "fig1_coverage")


def fig_nuisance(s, out):
    s = s[s.learner != "oracle"]
    cells = grid_labels(s)
    learners = [l for l in STYLE if l in set(s.learner)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2), sharex=True)
    for row, dgp in enumerate(["A", "B"]):
        for col, (var, lab) in enumerate([("err_m", r"$\|\hat m - m_0\|_2$"),
                                          ("err_l", r"$\|\hat\ell - \ell_0\|_2$")]):
            ax = axes[row, col]
            for lrn in learners:
                st = STYLE[lrn]
                d = s[(s.dgp == dgp) & (s.learner == lrn)]
                y = [d[(d.n == n) & (d.p == p)][var].iloc[0] for n, p, _ in cells]
                ax.plot(range(len(cells)), y, marker=st["marker"], color=st["color"],
                        lw=1.5, ms=5, label=st["label"])
            ax.set_ylim(bottom=0)
            ax.set_title(f"DGP {dgp}: {lab}", loc="left", fontsize=9, color=INK)
            ax.set_xticks(range(len(cells)), [c[2] for c in cells])
    axes[0, 0].set_ylabel("Median test-sample L2 error")
    axes[1, 0].set_ylabel("Median test-sample L2 error")
    axes[0, 1].legend(fontsize=8)
    _save(fig, out, "fig2_nuisance_error")


def fig_coverage_vs_remainder(s, out):
    s = s[s.learner != "oracle"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for lrn in [l for l in STYLE if l in set(s.learner)]:
        st = STYLE[lrn]
        for dgp, face in [("A", st["color"]), ("B", "white")]:
            d = s[(s.learner == lrn) & (s.dgp == dgp)]
            ax.scatter(d.R_nuis, d.coverage, marker=st["marker"], s=40,
                       facecolor=face, edgecolor=st["color"], linewidth=1.5,
                       label=f"{st['label']}, DGP {dgp}")
    ax.axhline(0.95, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"Remainder proxy $\hat R_{nuis}=\sqrt{n}\,\|\hat m-m_0\|"
                  r"(\|\hat m-m_0\|+\|\hat\ell-\ell_0\|)$  (log scale)")
    ax.set_ylabel("Coverage of 95% CI")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, out, "fig3_coverage_vs_remainder")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot_summary.csv"
    out = os.path.join(os.path.dirname(path) or ".", "figures")
    os.makedirs(out, exist_ok=True)
    s = pd.read_csv(path)
    fig_coverage(s, out)
    fig_nuisance(s, out)
    fig_coverage_vs_remainder(s, out)
    print(f"Figures written to {out}/")
