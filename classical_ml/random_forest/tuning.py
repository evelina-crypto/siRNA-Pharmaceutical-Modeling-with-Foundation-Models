"""Hyperparameter tuning utilities for Random Forest experiments."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from classical_ml.random_forest.config import RandomForestExperimentConfig
from utils.splitter import GroupKFoldLeakPerGroup


def default_random_forest_search_space() -> dict[str, list]:
    """Reasonable discrete search space for the first RF tuning pass."""
    return {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    }


class RandomForestHyperparameterTuner:
    """Tunes RF hyperparameters with grouped CV on the training split only."""

    def __init__(self, config: RandomForestExperimentConfig, param_distributions: dict[str, list] | None = None):
        self.config = config
        self.param_distributions = param_distributions or default_random_forest_search_space()
        self.cv = GroupKFoldLeakPerGroup(
            n_splits=config.n_splits,
            leak_n=config.leak_n,
            random_state=config.random_state,
        )

    def tune(self, X_train, y_train, groups_train) -> dict[str, object]:
        estimator = RandomForestRegressor(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=self.param_distributions,
            n_iter=self.config.tuning_n_iter,
            scoring=self.config.tuning_scoring,
            cv=self.cv,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train, groups=groups_train)

        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
        return {
            "search": search,
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "cv_results": cv_results,
        }
