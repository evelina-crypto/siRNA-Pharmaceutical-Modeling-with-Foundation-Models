"""Model wrapper for Random Forest regression."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from classical_ml.random_forest.config import RandomForestExperimentConfig


class RandomForestTrainer:
    """Thin wrapper around sklearn's RandomForestRegressor."""

    def __init__(self, config: RandomForestExperimentConfig):
        self.config = config
        self.model = self.build_model()

    def build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

    def fit(self, X_train, y_train) -> "RandomForestTrainer":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        return self.model.predict(X_test)
