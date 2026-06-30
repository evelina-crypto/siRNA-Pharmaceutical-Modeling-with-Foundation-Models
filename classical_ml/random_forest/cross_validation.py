"""Cross-validation orchestration for Random Forest experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from classical_ml.random_forest.config import RandomForestExperimentConfig
from classical_ml.random_forest.metrics import compute_regression_metrics
from classical_ml.random_forest.model import RandomForestTrainer
from utils.splitter import GroupKFoldLeakPerGroup


class RandomForestCrossValidator:
    """Runs fold-wise training and evaluation using the project splitter."""

    def __init__(self, config: RandomForestExperimentConfig):
        self.config = config
        self.splitter = GroupKFoldLeakPerGroup(
            n_splits=config.n_splits,
            leak_n=config.leak_n,
            random_state=config.random_state,
        )

    def run_cv(self, X, y, groups) -> dict[str, object]:
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)

        fold_rows = []
        prediction_rows = []

        for fold_idx, (train_idx, test_idx) in enumerate(
                self.splitter.split(X, y, groups),
                start=1,
        ):
            trainer = RandomForestTrainer(self.config)
            trainer.fit(X[train_idx], y[train_idx])
            y_pred = trainer.predict(X[test_idx])

            metrics = compute_regression_metrics(y[test_idx], y_pred)
            fold_rows.append({
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_groups": int(len(np.unique(groups[train_idx]))),
                "n_test_groups": int(len(np.unique(groups[test_idx]))),
                **metrics,
            })

            fold_prediction_df = pd.DataFrame({
                "fold": fold_idx,
                "row_index": test_idx,
                "group": groups[test_idx],
                "y_true": y[test_idx],
                "y_pred": y_pred,
            })
            prediction_rows.append(fold_prediction_df)

        fold_results = pd.DataFrame(fold_rows)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        summary = fold_results.drop(columns=["fold"]).mean(numeric_only=True).to_frame().T
        summary.insert(0, "fold", "mean")

        return {
            "fold_results": fold_results,
            "summary": summary,
            "predictions": predictions,
        }
