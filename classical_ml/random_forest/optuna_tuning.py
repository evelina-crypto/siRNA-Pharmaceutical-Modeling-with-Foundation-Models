"""Optuna-based hyperparameter tuning utilities for Random Forest experiments."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from classical_ml.random_forest.config import RandomForestExperimentConfig
from classical_ml.random_forest.metrics import compute_regression_metrics
from utils.splitter import GroupKFoldLeakPerGroup


def default_random_forest_optuna_space(trial) -> dict[str, object]:
    """Default Optuna search space for the first RF tuning pass."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30, 40]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
    }


class RandomForestOptunaTuner:
    """Tunes RF hyperparameters with Optuna using grouped inner CV only on outer-train data."""

    def __init__(self, config: RandomForestExperimentConfig, search_space_fn=None):
        self.config = config
        self.search_space_fn = search_space_fn or default_random_forest_optuna_space
        self.cv = GroupKFoldLeakPerGroup(
            n_splits=config.n_splits,
            leak_n=config.leak_n,
            random_state=config.random_state,
        )

    def tune(self, X_train, y_train, groups_train) -> dict[str, object]:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna is not installed. Install it with `pip install optuna` "
                "before using RandomForestOptunaTuner."
            ) from exc

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train, dtype=float)
        groups_train = np.asarray(groups_train)

        def objective(trial) -> float:
            params = self.search_space_fn(trial)
            model = RandomForestRegressor(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                **params,
            )

            train_maes = []
            validation_maes = []
            train_pearsons = []
            validation_pearsons = []

            for train_idx, validation_idx in self.cv.split(X_train, y_train, groups_train):
                X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
                X_fold_val, y_fold_val = X_train[validation_idx], y_train[validation_idx]

                model.fit(X_fold_train, y_fold_train)

                train_pred = model.predict(X_fold_train)
                validation_pred = model.predict(X_fold_val)

                train_metrics = compute_regression_metrics(y_fold_train, train_pred)
                validation_metrics = compute_regression_metrics(y_fold_val, validation_pred)

                train_maes.append(train_metrics["mae"])
                validation_maes.append(validation_metrics["mae"])
                train_pearsons.append(train_metrics["pearson"])
                validation_pearsons.append(validation_metrics["pearson"])

            mean_train_mae = float(np.mean(train_maes))
            mean_validation_mae = float(np.mean(validation_maes))
            mean_train_pearson = float(np.nanmean(train_pearsons))
            mean_validation_pearson = float(np.nanmean(validation_pearsons))

            trial.set_user_attr("mean_train_mae", mean_train_mae)
            trial.set_user_attr("mean_validation_mae", mean_validation_mae)
            trial.set_user_attr("mean_train_pearson", mean_train_pearson)
            trial.set_user_attr("mean_validation_pearson", mean_validation_pearson)
            trial.set_user_attr("train_validation_mae_gap", mean_validation_mae - mean_train_mae)

            return mean_validation_mae

        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=self.config.optuna_n_trials)

        trials_df = self._build_trials_dataframe(study)
        best_estimator = RandomForestRegressor(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            **study.best_params,
        )
        best_estimator.fit(X_train, y_train)

        return {
            "study": study,
            "best_estimator": best_estimator,
            "best_params": study.best_params,
            "best_score": float(study.best_value),
            "trials_df": trials_df,
        }

    def _build_trials_dataframe(self, study) -> pd.DataFrame:
        records = []
        for trial in study.trials:
            records.append({
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value if trial.value is not None else math.nan,
                "mean_validation_mae": trial.user_attrs.get("mean_validation_mae", math.nan),
                "mean_train_mae": trial.user_attrs.get("mean_train_mae", math.nan),
                "train_validation_mae_gap": trial.user_attrs.get("train_validation_mae_gap", math.nan),
                "mean_validation_pearson": trial.user_attrs.get("mean_validation_pearson", math.nan),
                "mean_train_pearson": trial.user_attrs.get("mean_train_pearson", math.nan),
                **trial.params,
            })

        trials_df = pd.DataFrame(records)
        if not trials_df.empty and "value" in trials_df.columns:
            trials_df = trials_df.sort_values("value", ascending=True).reset_index(drop=True)
            trials_df.insert(0, "rank_validation_mae", np.arange(1, len(trials_df) + 1))
        return trials_df
