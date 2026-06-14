"""End-to-end outer-holdout experiment workflow for Random Forest."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from classical_ml.random_forest.config import RandomForestExperimentConfig
from classical_ml.random_forest.cross_validation import RandomForestCrossValidator
from classical_ml.random_forest.metrics import compute_regression_metrics
from classical_ml.random_forest.model import RandomForestTrainer
from classical_ml.random_forest.splitting import RandomForestHoldoutSplitter
from classical_ml.random_forest.tuning import RandomForestHyperparameterTuner


class RandomForestNestedExperiment:
    """Tunes on train/validation data and evaluates once on held-out test data."""

    def __init__(self, config: RandomForestExperimentConfig, param_distributions: dict[str, list] | None = None):
        self.config = config
        self.holdout_splitter = RandomForestHoldoutSplitter(config)
        self.cross_validator = RandomForestCrossValidator(config)
        self.tuner = RandomForestHyperparameterTuner(config, param_distributions=param_distributions)

    def run(self, X, y, groups) -> dict[str, object]:
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)

        train_idx, test_idx = self.holdout_splitter.split(X, y, groups)
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]

        tuning_results = self.tuner.tune(X_train, y_train, groups_train)
        tuned_config = self._config_with_best_params(tuning_results["best_params"])

        inner_cv_results = RandomForestCrossValidator(tuned_config).run_cv(X_train, y_train, groups_train)

        final_trainer = RandomForestTrainer(tuned_config)
        final_trainer.fit(X_train, y_train)
        y_test_pred = final_trainer.predict(X_test)
        test_metrics = compute_regression_metrics(y_test, y_test_pred)

        test_predictions = pd.DataFrame({
            "row_index": test_idx,
            "group": groups_test,
            "y_true": y_test,
            "y_pred": y_test_pred,
        })
        test_predictions["residual"] = test_predictions["y_true"] - test_predictions["y_pred"]
        test_predictions["abs_error"] = test_predictions["residual"].abs()

        split_summary = pd.DataFrame([{
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_groups": int(len(np.unique(groups_train))),
            "n_test_groups": int(len(np.unique(groups_test))),
        }])

        return {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "split_summary": split_summary,
            "tuning_results": tuning_results,
            "inner_cv_results": inner_cv_results,
            "test_metrics": pd.DataFrame([test_metrics]),
            "test_predictions": test_predictions,
            "best_params": tuning_results["best_params"],
        }

    def _config_with_best_params(self, best_params: dict[str, object]) -> RandomForestExperimentConfig:
        config_values = asdict(self.config)
        config_values.update(best_params)
        return RandomForestExperimentConfig(**config_values)
