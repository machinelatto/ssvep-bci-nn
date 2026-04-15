import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    recall_score,
    accuracy_score,
)
import torch
from tqdm import tqdm
import copy
# import tqdm.notebook

from benchmark_dataset import (
    bandpass_filter as _bandpass_filter,
    filter_signals_subbands as _filter_signals_subbands,
    load_data_from_users as _load_data_from_users,
)


class EarlyStopping:
    """Early stopping to avoid overfitting during training.
    
    Monitors a validation metric and stops training if no improvement is seen
    for a specified number of epochs (patience).
    
    Args:
        monitor (str): Metric to monitor. Options: 'val_loss', 'val_accuracy'. 
                      Default: 'val_loss'
        patience (int): Number of epochs with no improvement after which training 
                       will be stopped. Default: 10
        verbose (bool): If True, prints messages when early stopping is triggered 
                       or best model is saved. Default: True
        delta (float): Minimum change in monitored value to qualify as improvement. 
                      Default: 0.0
    """
    
    def __init__(self, monitor='val_loss', patience=10, verbose=True, delta=0.0):
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0
        
        # Determine if we're maximizing or minimizing the metric
        if monitor == 'val_loss':
            self.is_maximize = False
        else:  # val_accuracy
            self.is_maximize = True
    
    def __call__(self, model, current_value, epoch):
        """Check if training should stop.
        
        Args:
            model (nn.Module): PyTorch model to potentially save
            current_value (float): Current value of the monitored metric
            epoch (int): Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_value
            self._save_checkpoint(model, epoch)
        else:
            if self._is_improvement(current_value):
                self.best_score = current_value
                self.counter = 0
                self._save_checkpoint(model, epoch)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"EarlyStopping: No improvement for {self.patience} epochs. Stopping training.")
        
        return self.early_stop
    
    def _is_improvement(self, current_value):
        """Check if current value is an improvement over best score."""
        if self.is_maximize:
            return current_value > (self.best_score + self.delta)
        else:
            return current_value < (self.best_score - self.delta)
    
    def _save_checkpoint(self, model, epoch):
        """Save model checkpoint."""
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.best_epoch = epoch
        if self.verbose:
            metric_name = self.monitor.replace('_', ' ').title()
            print(f"EarlyStopping: {metric_name} improved. Saving model at epoch {epoch + 1}.")
    
    def load_best_model(self, model):
        """Load the best model state into the given model.
        
        Args:
            model (nn.Module): PyTorch model to load weights into
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"EarlyStopping: Loaded best model from epoch {self.best_epoch + 1}.")
        return model
    
    def reset(self):
        """Reset early stopping counters for a new training session."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot learning curves for training and validation loss and accuracy.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_accuracies (list): List of training accuracies.
        val_accuracies (list): List of validation accuracies.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves: Training and Validation")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curves: Training and Validation")
    plt.tight_layout()
    plt.show()


def filter_signals_subbands(eeg_signals, subban_no, sampling_rate):
    """Compatibility wrapper for benchmark_dataset.filter_signals_subbands."""
    return _filter_signals_subbands(eeg_signals, subban_no, sampling_rate)


def get_windows(eeg_matrix, window_size, include_last=False):
    """Extract sliding windows from the EEG matrix.

    Args:
        eeg_matrix (np.ndarray): Input EEG matrix. Supports both:
            - Standard BCI format: shape (channels, timepoints)
            - Time-first format: shape (timepoints, channels)
        window_size (int): Size of each window (in time samples).
        include_last (bool, optional): Whether to include the last window if it's smaller than window_size. Defaults to False.

    Returns:
        tuple: (list of windows, num_windows)
            - If input is (channels, timepoints): each window has shape (channels, window_size)
            - If input is (timepoints, channels): each window has shape (window_size, channels)
    """
    # Determine which axis contains time samples
    # We check if the first dimension is likely time (typically larger) or channels (typically smaller, e.g., 9-64)
    # For BCI data: channels are usually 8-64, timepoints are typically 100-1000+
    if eeg_matrix.shape[0] > eeg_matrix.shape[1]:
        # Likely (timepoints, channels) - time is first dimension
        time_axis = 0
        total_samples = eeg_matrix.shape[0]
    else:
        # Likely (channels, timepoints) - time is second dimension
        time_axis = 1
        total_samples = eeg_matrix.shape[1]

    num_windows = total_samples // window_size
    windows = []

    if time_axis == 0:
        # (timepoints, channels) format
        for i in range(0, total_samples, window_size):
            window = eeg_matrix[i : i + window_size]
            windows.append(window)
    else:
        # (channels, timepoints) format
        for i in range(0, total_samples, window_size):
            window = eeg_matrix[:, i : i + window_size]
            windows.append(window)

    if not include_last and total_samples % window_size != 0:
        windows.pop()

    return windows, num_windows


def evaluate(model, test_loader):
    model.eval()
    model_device = next(model.parameters()).device
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(model_device)
            labels = labels.to(model_device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    # print(f"Test set Accuracy: {accuracy:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # classes = np.unique(np.concatenate((all_labels, all_preds)))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    # plt.show()
    return accuracy, recall, f1, cm


def bandpass_filter(
    dados, taxa_amostragem, freq_corte_low, freq_corte_high, ordem_filtro
):
    """Compatibility wrapper for benchmark_dataset.bandpass_filter."""
    return _bandpass_filter(
        dados,
        taxa_amostragem,
        freq_corte_low,
        freq_corte_high,
        ordem_filtro,
    )


def car_filter(eeg_matrix, reference_channels=None, target_channels=None):
    """Apply CAR to a 2D EEG matrix with shape (channels, samples).

    Args:
        eeg_matrix (np.ndarray): EEG array in (channels, samples) layout.
        reference_channels (array-like, optional): Channels used to compute the
            CAR reference. If None, all channels are used.
        target_channels (array-like, optional): Channels where CAR is applied.
            If None, CAR is applied to all channels.

    Returns:
        np.ndarray: CAR-referenced EEG matrix with same shape as input.
    """
    eeg_matrix = np.asarray(eeg_matrix)
    if eeg_matrix.ndim != 2:
        raise ValueError(
            f"car_filter expects a 2D array (channels, samples), got shape {eeg_matrix.shape}"
        )

    num_channels = eeg_matrix.shape[0]

    if reference_channels is None:
        reference_channels = np.arange(num_channels)
    reference_channels = np.asarray(reference_channels, dtype=int)

    if target_channels is None:
        target_channels = np.arange(num_channels)
    target_channels = np.asarray(target_channels, dtype=int)

    if reference_channels.size == 0:
        raise ValueError("reference_channels cannot be empty")

    reference = np.mean(eeg_matrix[reference_channels, :], axis=0, keepdims=False)
    out = eeg_matrix.copy()
    out[target_channels, :] = eeg_matrix[target_channels, :] - reference[np.newaxis, :]
    return out


def load_data_from_users(
    users,
    visual_delay=160,
    dataset_path="C:/Users/machi/Documents/Mestrado/repos/data/benchmark/",
    filter_bandpass=False,
    apply_car=False,
    car_reference_channels=None,
    car_target_channels=None,
    sample_rate=250,
    freq_cut_low=6,
    freq_cut_high=70,
    filter_order=10,
):
    """Compatibility wrapper for benchmark_dataset.load_data_from_users.

    Optional preprocessing order:
    1) Bandpass (if filter_bandpass=True)
    2) CAR per (frequency, trial) slice (if apply_car=True)
    """
    data = _load_data_from_users(
        users=users,
        visual_delay=visual_delay,
        dataset_path=dataset_path,
        filter_bandpass=filter_bandpass,
        sample_rate=sample_rate,
        freq_cut_low=freq_cut_low,
        freq_cut_high=freq_cut_high,
        filter_order=filter_order,
    )

    if not apply_car:
        return data

    data_car = []
    for user_data in data:
        user_data_car = user_data.copy()
        _, _, num_freqs, num_trials = user_data_car.shape

        for freq_idx in range(num_freqs):
            for trial_idx in range(num_trials):
                user_data_car[:, :, freq_idx, trial_idx] = car_filter(
                    user_data_car[:, :, freq_idx, trial_idx],
                    reference_channels=car_reference_channels,
                    target_channels=car_target_channels,
                )

        data_car.append(user_data_car)

    return data_car


