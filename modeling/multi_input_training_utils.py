"""multi_input_training_utils.py

Comments : Multi-input variants of IndexedTensorDataset / train_model / evaluate_model
           for the CrewSiRNAModel when use_experimental=True (sequence + experimental
           conditions as two separate tensors).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader


class IndexedMultiTensorDataset(torch.utils.data.Dataset):
    """Same idea as IndexedTensorDataset but carries two feature tensors."""

    def __init__(self, X_seq_tensor, X_exp_tensor, y_tensor, indices):
        self.X_seq = X_seq_tensor
        self.X_exp = X_exp_tensor
        self.y = y_tensor
        self.indices = indices

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_exp[idx], self.y[idx], self.indices[idx]


def train_model_multi(model, train_loader, val_loader, criterion, optimizer,
                       epochs=1000, device='cpu', patience=15):
    from modeling.training_utils import EarlyStopping  # reuse existing early stopping

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
    predictions, actuals, sample_names = [], [], []
    test_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X_seq, X_exp, y, sample_id in test_loader:
            X_seq, X_exp, y = X_seq.to(device), X_exp.to(device), y.to(device)
            y = y.squeeze(-1)
            y_pred = model(X_seq, X_exp).flatten()
            loss = criterion(y_pred, y.float())
            test_loss += loss.item()

            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            sample_names.extend(sample_id)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()

    test_loss /= len(test_loader)

    if (np.std(predictions) == 0 or np.std(actuals) == 0
            or np.any(np.isnan(predictions)) or np.any(np.isnan(actuals))):
        test_corr = np.nan
    else:
        test_corr = np.corrcoef(predictions, actuals)[0, 1]

    test_mse = mean_squared_error(actuals, predictions)

    metrics = {'test_loss': test_loss, 'test_correlation': test_corr, 'test_mse': test_mse}
    return metrics, predictions, actuals, sample_names