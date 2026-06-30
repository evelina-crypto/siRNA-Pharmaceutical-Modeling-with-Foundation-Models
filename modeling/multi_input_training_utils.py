"""multi_input_training_utils.py

Comments : Multi-input variants of IndexedTensorDataset / train_model / evaluate_model
           for the CrewSiRNAModel when use_experimental=True (sequence + experimental
           conditions as two separate tensors).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


class IndexedMultiTensorDataset(torch.utils.data.Dataset):
    """Same idea as IndexedTensorDataset but carries two feature tensors."""

    def __init__(self, X_seq_tensor, X_exp_tensor, y_tensor, indices,
                 X_mrna_tensor=None):
        self.X_seq = X_seq_tensor
        self.X_exp = X_exp_tensor
        self.X_mrna = X_mrna_tensor
        self.y = y_tensor
        self.indices = indices

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.X_mrna is not None:
            return (
                self.X_seq[idx], self.X_exp[idx], self.X_mrna[idx],
                self.y[idx], self.indices[idx],
            )
        return self.X_seq[idx], self.X_exp[idx], self.y[idx], self.indices[idx]


def collate_runtime_slices(samples, width=100):
    """Collate three raw RNA slices into a fixed ``(B, 3, 4, width)`` tensor."""
    from utils.fm_utils import seq_to_one_hot

    X_seq, X_exp, slice_triplets, y, sample_ids = zip(*samples)
    batch_size = len(samples)
    one_hot = torch.zeros(batch_size, 3, 4, width, dtype=torch.float32)
    lengths = torch.ones(batch_size, 3, dtype=torch.long)
    present_mask = torch.zeros(batch_size, 3, dtype=torch.bool)

    for row, triplet in enumerate(slice_triplets):
        for column, sequence in enumerate(triplet):
            if not isinstance(sequence, str) or not sequence:
                continue
            encoded = seq_to_one_hot(sequence)
            sequence_length = encoded.shape[1]
            if sequence_length > width:
                raise ValueError(
                    f"Runtime slice length {sequence_length} exceeds fixed width {width}"
                )
            one_hot[row, column, :, :sequence_length] = torch.from_numpy(encoded)
            lengths[row, column] = sequence_length
            present_mask[row, column] = True

    packed_slices = (one_hot, lengths, present_mask)
    return (
        torch.stack(X_seq), torch.stack(X_exp), packed_slices,
        torch.stack(y), list(sample_ids),
    )


def _unpack_batch(batch, device):
    if len(batch) == 5:
        X_seq, X_exp, X_mrna, y, sample_id = batch
        if isinstance(X_mrna, (tuple, list)):
            X_mrna = tuple(value.to(device) for value in X_mrna)
        else:
            X_mrna = X_mrna.to(device)
    else:
        X_seq, X_exp, y, sample_id = batch
        X_mrna = None
    return X_seq.to(device), X_exp.to(device), X_mrna, y.to(device), sample_id


def train_model_multi(model, train_loader, val_loader, criterion, optimizer,
                       epochs=1000, device='cpu', patience=15,
                       gradient_accumulation=1, mixed_precision=False,
                       max_grad_norm=None):
    from modeling.training_utils import EarlyStopping

    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1")
    use_amp = mixed_precision and str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            X_seq, X_exp, X_mrna, y, _ = _unpack_batch(batch, device)
            y = y.squeeze(-1)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp,
            ):
                y_pred = model(X_seq, X_exp, X_mrna).flatten()
                loss = criterion(y_pred, y.float())
                remainder = len(train_loader) % gradient_accumulation
                is_final_partial = remainder and step > len(train_loader) - remainder
                accumulation_divisor = remainder if is_final_partial else gradient_accumulation
                backward_loss = loss / accumulation_divisor

            scaler.scale(backward_loss).backward()
            should_step = step % gradient_accumulation == 0 or step == len(train_loader)
            if should_step:
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X_seq, X_exp, X_mrna, y, _ = _unpack_batch(batch, device)
                y = y.squeeze(-1)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp,
                ):
                    y_pred = model(X_seq, X_exp, X_mrna).flatten()
                    loss = criterion(y_pred, y.float())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return model, history


def evaluate_model_multi(scaler_y, model, test_loader, device='cpu',
                         mixed_precision=False):
    model.eval()
    predictions, actuals, sample_names = [], [], []
    test_loss = 0.0
    criterion = nn.MSELoss()
    use_amp = mixed_precision and str(device).startswith("cuda")

    with torch.no_grad():
        for batch in test_loader:
            X_seq, X_exp, X_mrna, y, sample_id = _unpack_batch(batch, device)
            y = y.squeeze(-1)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp,
            ):
                y_pred = model(X_seq, X_exp, X_mrna).flatten()
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
        test_spearman = np.nan
    else:
        test_corr = np.corrcoef(predictions, actuals)[0, 1]
        test_spearman = spearmanr(predictions, actuals).statistic

    test_mse = mean_squared_error(actuals, predictions)

    metrics = {
        'test_loss': test_loss,
        'test_correlation': test_corr,  # Pearson correlation
        'test_spearman': test_spearman,
        'test_mse': test_mse,
    }
    return metrics, predictions, actuals, sample_names