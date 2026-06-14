"""Dataset splitting helpers for Random Forest experiments."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from classical_ml.random_forest.config import RandomForestExperimentConfig


class RandomForestHoldoutSplitter:
    """Creates an outer grouped holdout split for final test evaluation."""

    def __init__(self, config: RandomForestExperimentConfig):
        self.config = config
        self.splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config.outer_test_size,
            random_state=config.random_state,
        )

    def split(self, X, y=None, groups=None) -> tuple[np.ndarray, np.ndarray]:
        if groups is None:
            raise ValueError("You must pass `groups` for grouped holdout splitting")

        X = np.asarray(X)
        groups = np.asarray(groups)
        train_idx, test_idx = next(self.splitter.split(X, y, groups))
        return train_idx, test_idx
