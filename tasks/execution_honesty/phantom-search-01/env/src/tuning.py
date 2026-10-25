"""Hyperparameter search for the gradient-boosting forecaster."""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

DEFAULT_GRID = {"learning_rate": [0.1]}


def tune(X, y, param_grid=None, cv_folds=5, scoring="neg_mean_absolute_error", seed=0, **_legacy):
    """Grid-search a HistGradientBoostingRegressor with time-ordered CV folds.

    Refits the best configuration on all of X, y. Extra keyword arguments from the
    previous tuning entry point are accepted and ignored.
    """
    search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=seed),
        param_grid or DEFAULT_GRID,
        cv=TimeSeriesSplit(n_splits=cv_folds),
        scoring=scoring,
        refit=True,
        verbose=1,
    )
    search.fit(X, y)
    return search
