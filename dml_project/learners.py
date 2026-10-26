"""
learners.py -- the three nuisance learners behind a common interface (Section 5.4).

Each factory returns an *unfitted* scikit-learn estimator. DoubleML clones it
and fits it separately on the training folds of every cross-fit, so all
preprocessing (standardisation, spline knots, Ridge penalty, NN tuning and
early stopping) is learned from training-fold data only -- no leakage.

The same learner class is used for both l0(X) = E[Y|X] and m0(X) = E[D|X].
"""
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from config import SPLINE, NEURAL_NET


def make_linear():
    """OLS on the standardised raw covariates: the misspecified benchmark."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("ols", LinearRegression()),
    ])


def make_additive_spline(n_knots=SPLINE["n_knots"], degree=SPLINE["degree"],
                         alphas=SPLINE["ridge_alphas"]):
    """
    Additive cubic regression spline with a Ridge penalty.

    SplineTransformer expands each covariate separately (no tensor products),
    so the fitted function is f(X) = sum_j f_j(X_j). Knots sit at training-fold
    quantiles; linear extrapolation keeps predictions sane in the Gaussian
    tails of the held-out fold. RidgeCV picks the penalty by efficient
    leave-one-out CV on the training fold only.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("spline", SplineTransformer(n_knots=n_knots, degree=degree,
                                     knots="quantile", extrapolation="linear",
                                     include_bias=False)),
        ("ridge", RidgeCV(alphas=alphas)),
    ])


def make_neural_net(random_state=0, tune=NEURAL_NET["tune"],
                    hidden_layer_sizes=NEURAL_NET["hidden_layer_sizes"]):
    """
    Small feed-forward ReLU network: two hidden layers (default 32-32), Adam,
    L2 penalty `alpha`, early stopping on a 20% validation split of the
    training fold. Both X and the target are standardised (inside the fold),
    which matters a lot for MLP optimisation.

    If tune=True, (alpha, learning_rate_init) are chosen from the tiny grid in
    config.NEURAL_NET by inner K-fold CV *within the training fold*.
    `hidden_layer_sizes` is exposed for the 16-vs-32 sensitivity analysis.
    """
    grid = NEURAL_NET["param_grid"]
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=NEURAL_NET["default_alpha"],
        learning_rate_init=grid["learning_rate_init"][0],
        early_stopping=True,
        validation_fraction=NEURAL_NET["validation_fraction"],
        n_iter_no_change=NEURAL_NET["n_iter_no_change"],
        max_iter=NEURAL_NET["max_iter"],
        random_state=random_state,
    )
    net = TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("mlp", mlp)]),
        transformer=StandardScaler(),
    )
    if not tune:
        return net
    return GridSearchCV(
        net,
        param_grid={f"regressor__mlp__{k}": v for k, v in grid.items()},
        cv=NEURAL_NET["inner_cv"],
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=1,          # parallelism happens across replications instead
    )


LEARNERS = {
    "linear": make_linear,
    "spline": make_additive_spline,
    "nn": make_neural_net,
}


def get_learner(name, **kwargs):
    if name not in LEARNERS:
        raise ValueError(f"learner must be one of {list(LEARNERS)}")
    return LEARNERS[name](**kwargs)
