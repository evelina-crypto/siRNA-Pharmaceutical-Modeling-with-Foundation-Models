import optuna
import numpy as np
from run_crew import run_sequence_cnn_cv

def objective(trial):
    params = {
        "channels": (
            trial.suggest_categorical("c0", [32, 64, 128]),
            trial.suggest_categorical("c1", [64, 128, 256]),
            trial.suggest_categorical("c2", [128, 256, 512])
        ),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "emb_dim": trial.suggest_categorical("emb_dim", [64, 128, 256]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [64, 128, 256])
    }

    metrics = run_sequence_cnn_cv(
        n_splits=2,
        channels=params["channels"],
        dropout=params["dropout"],
        lr=params["lr"],
        emb_dim=params["emb_dim"],
        fusion_hidden=params["fusion_hidden"],
        epochs=10,
        tag=f"trial_{trial.number}"
    )
    return np.mean([m["test_spearman"] for m in metrics])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best params:", study.best_params)