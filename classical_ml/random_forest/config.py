"""Configuration objects for Random Forest experiments."""

from dataclasses import dataclass


@dataclass(slots=True)
class RandomForestExperimentConfig:
    """Single source of truth for preprocessing, CV, and RF parameters."""

    target_column: str = "Inhibition"
    target_len: int = 25
    strict_cleaning: bool = True
    add_mrna: bool = True
    fetch_missing_mrna: bool = True
    use_normalized_conditions: bool = False

    n_splits: int = 3
    leak_n: int = 0
    outer_test_size: float = 0.33
    random_state: int = 42

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float | int | None = "sqrt"
    n_jobs: int = -1

    tuning_n_iter: int = 20
    tuning_scoring: str = "neg_mean_absolute_error"
    optuna_n_trials: int = 20
