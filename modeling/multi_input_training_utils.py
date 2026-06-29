"""multi_input_training_utils.py

Comments : Multi-input variants of IndexedTensorDataset / train_model / evaluate_model
           for the CrewSiRNAModel when use_experimental=True (sequence + experimental
           conditions as two separate tensors). Evaluation reports flat Spearman
           (all samples) and weighted Spearman per patent group.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error


class IndexedMultiTensorDataset(torch.utils.data.Dataset):
    """Carries two feature tensors, targets and patent labels."""

    def __init__(self, X_seq_tensor, X_exp_tensor, y_tensor, patent_indices):
        self.X_seq = X_seq_tensor
        self.X_exp = X_exp_tensor
        self.y = y_tensor
        self.patent_indices = patent_indices

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_exp[idx], self.y[idx], self.patent_indices[idx]


def compute_weighted_spearman(predictions, actuals, groups, min_n=5):
    """Weighted Spearman: sum(spearman_group * n_group) / total_n.

    Only groups with at least min_n samples contribute.
    Returns NaN if no group meets the threshold.
    """
    unique_groups = np.unique(groups)
    weighted_sum = 0.0
    total_n = 0

    for group in unique_groups:
        mask = groups == group
        n = mask.sum()
        if n < min_n:
            continue
        g_preds = predictions[mask]
        g_actuals = actuals[mask]
        if np.std(g_preds) == 0 or np.std(g_actuals) == 0:
            continue
        r, _ = spearmanr(g_preds, g_actuals)
        if np.isnan(r):
            continue
        weighted_sum += r * n
        total_n += n

    return weighted_sum / total_n if total_n > 0 else np.nan


def train_model_multi(model, train_loader, val_loader, criterion, optimizer,
                      epochs=1000, device='cpu', patience=15):
    from modeling.training_utils import EarlyStopping

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_seq, X_exp, y, _ in train_loader:
            X_seq, X_exp, y = X_seq.to(device), X_exp.to(device), y.to(device)
            y = y.squeeze(-1)
            optimizer.zero_grad()
            y_pred = model(X_seq, X_exp).flatten()
            loss = criterion(y_pred, y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_seq, X_exp, y, _ in val_loader:
                X_seq, X_exp, y = X_seq.to(device), X_exp.to(device), y.to(device)
                y = y.squeeze(-1)
                y_pred = model(X_seq, X_exp).flatten()
                loss = criterion(y_pred, y.float())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return model, history


def evaluate_model_multi(scaler_y, model, test_loader, device='cpu'):
    model.eval()
    predictions, actuals, patent_names = [], [], []
    test_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X_seq, X_exp, y, patent_id in test_loader:
            X_seq, X_exp, y = X_seq.to(device), X_exp.to(device), y.to(device)
            y = y.squeeze(-1)
            y_pred = model(X_seq, X_exp).flatten()
            loss = criterion(y_pred, y.float())
            test_loss += loss.item()

            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            patent_names.extend(patent_id)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()

    test_loss /= len(test_loader)

    if (np.std(predictions) == 0 or np.std(actuals) == 0
            or np.any(np.isnan(predictions)) or np.any(np.isnan(actuals))):
        test_spearman = np.nan
        test_corr = np.nan
        test_pearson = np.nan
    else:
        test_spearman, _ = spearmanr(predictions, actuals)
        test_corr = np.corrcoef(predictions, actuals)[0, 1]
        test_pearson = pearsonr(predictions, actuals)[0]

    test_mse = mean_squared_error(actuals, predictions)

    test_spearman_weighted_patent = compute_weighted_spearman(
        predictions, actuals, np.array(patent_names), min_n=5
    )

    metrics = {
        'test_loss':                     test_loss,
        'test_correlation':              test_corr,
        'test_mse':                      test_mse,
        'test_spearman':                 test_spearman,
        'test_pearson':                  test_pearson,
        'test_spearman_weighted_patent': test_spearman_weighted_patent,
    }
    return metrics, predictions, actuals