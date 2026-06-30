"""training_utils.py

Comments : Project-agnostic training utility methods: seeding, datasets, early stopping, train/eval loops.
"""

import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader


def set_global_seed(seed):
    """Set global seed for deterministic results across all random operations"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # determinism for numpy operations
    np.random.RandomState(seed)
    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_validation_loader(train_dataset, val_split=0.2, batch_size=32,
                             generator=None, collate_fn=None):
    """Create validation loader from training data."""
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=generator
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "generator": generator,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(train_subset, **loader_kwargs)
    val_loader = DataLoader(val_subset, **loader_kwargs)
    return train_loader, val_loader


class IndexedTensorDataset(torch.utils.data.Dataset):
    """Preserves sample IDs with features and targets"""

    def __init__(self, X_tensor, y_tensor, indices):
        self.X = X_tensor
        self.y = y_tensor
        self.indices = indices  # Sample IDs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())

        elif val_loss < self.best_loss - self.min_delta:
            # Improvement: update best loss and reset counter
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

        else:
            # No improvement: increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1000, device='cpu', patience=15):
    """Training function for PyTorch model with early stopping"""
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            y = y.squeeze(-1)
            optimizer.zero_grad()
            y_pred = model(X).flatten()
            loss = criterion(y_pred, y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                y = y.squeeze(-1)
                y_pred = model(X).flatten()
                loss = criterion(y_pred, y.float())
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        history['val_loss'].append(val_loss)

        # printing the training progress
        # print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return model, history


def evaluate_model(scaler_y, model, test_loader, device='cpu'):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    actuals = []
    sample_names = []  # Store sample names
    test_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X, y, sample_id in test_loader:
            X, y = X.to(device), y.to(device)
            y = y.squeeze(-1)
            y_pred = model(X).flatten()
            loss = criterion(y_pred, y.float())
            test_loss += loss.item()

            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            sample_names.extend(sample_id)  # Collect sample names

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # inverse transformation: backscaling the y-targetcolumn for both predictions and actual values
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Metrics
    test_loss = test_loss / len(test_loader)

    # handle edge cases for correlation
    if (
            np.std(predictions) == 0 or
            np.std(actuals) == 0 or
            np.any(np.isnan(predictions)) or
            np.any(np.isnan(actuals))
    ):
        test_corr = np.nan
    else:
        test_corr = np.corrcoef(predictions, actuals)[0, 1]

    test_mse = mean_squared_error(actuals, predictions)

    metrics = {
        'test_loss': test_loss,
        'test_correlation': test_corr,
        'test_mse': test_mse
    }

    return metrics, predictions, actuals, sample_names
