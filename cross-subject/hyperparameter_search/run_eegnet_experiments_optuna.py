"""
Run EEGNet cross-subject experiments with Optuna hyperparameter tuning.
This script uses Optuna to automatically find optimal hyperparameters for each user.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import copy
import scipy.io
from braindecode.models import EEGNet
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from cross_subject_utils import (
    evaluate,
    load_data_from_users,
)


def extract_trials_trial_major(data, occipital_electrodes, indices, tamanho_da_janela, frequencias):
    """Build trial tensor in deterministic session-major then frequency order.

    Returns:
        x: np.ndarray, shape (num_trials, num_channels, num_timepoints)
        labels: list[float], frequency label per trial
    """
    x = []
    labels = []
    for session in range(data.shape[3]):
        for freq in indices:
            eeg_trial = np.ascontiguousarray(
                data[occipital_electrodes, :tamanho_da_janela, freq, session]
            )
            x.append(eeg_trial)
            labels.append(frequencias[freq])
    return np.array(x), labels


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=100,
    device=0,
    save_path="best_model.pth",
    trial=None,
):
    """
    Training function with optional Optuna trial for pruning.
    
    Args:
        trial: Optional Optuna Trial object for pruning
    """
    best_val_accuracy = -float("inf")
    model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(num_epochs)):
        # Training Phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # eval train
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total
        avg_train_loss = running_loss / len(train_loader)

        # eval validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # val accuracy
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Save if best vall acc
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # Optuna pruning
        if trial is not None:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    model.load_state_dict(best_model)
    return model, best_val_accuracy


def objective_per_user(
    trial: Trial,
    X_train,
    Y_train,
    tamanho_da_janela,
    frequencias_desejadas,
    device,
    seed,
    sample_rate=250,
    num_epochs=50,
):
    """
    Optuna objective function for hyperparameter tuning per user.
    
    Suggests hyperparameters and returns validation accuracy.
    """
    # Suggest hyperparameters
    F1 = trial.suggest_categorical("F1", [4, 8, 16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Create dataset and loaders (fixed 80/20 split for tuning)
    dataset = TensorDataset(X_train, Y_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create model with F1 parameter
    model = EEGNet(
        n_chans=9,
        n_outputs=len(frequencias_desejadas),
        n_times=tamanho_da_janela,
        kernel_length=(sample_rate // 2),
        F1=F1,
    )
    model = model.to(device)

    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    # Train and get best validation accuracy
    try:
        _, best_val_accuracy = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            device=device,
            save_path="/tmp/trial_model.pth",
            trial=trial,
        )
        return best_val_accuracy
    except optuna.TrialPruned:
        raise


def tune_hyperparameters_per_user(
    X_train,
    Y_train,
    tamanho_da_janela,
    frequencias_desejadas,
    device,
    seed,
    sample_rate=250,
    n_trials=20,
    tuning_epochs=25,
):
    """
    Run Optuna hyperparameter tuning for a specific user.
    
    Args:
        n_trials: Number of Optuna trials to run
        tuning_epochs: Number of epochs for each trial
    
    Returns:
        best_params: Dictionary of best hyperparameters
        study: Optuna study object with all trials
    """
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective_per_user(
            trial,
            X_train,
            Y_train,
            tamanho_da_janela,
            frequencias_desejadas,
            device,
            seed,
            sample_rate=sample_rate,
            num_epochs=tuning_epochs,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (accuracy): {study.best_value:.4f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    return study.best_params, study


# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Load frequency and phase information
frequencias_e_fases = scipy.io.loadmat(
    "/home/mateuschinelatto/Experiments/data/benchmark/Freq_Phase.mat"
)
frequencias = frequencias_e_fases["freqs"]
frequencias = np.round(frequencias, 2).ravel()
fases = frequencias_e_fases["phases"]

# Preprocessing parameters
filter_order = 10
freq_cut_high = 50
freq_cut_low = 6
sample_rate = 250
delay = 160

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 36))  # 35 users (full dataset)
frequencias_desejadas = frequencias[:]  # All 40 frequencies
indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

print("Users of interest:", users)
print("Frequencies of interest:", frequencias_desejadas)
print("Indices of frequencies of interest:", indices)

# Load all data
print("\nLoading data from all users...")
all_data = load_data_from_users(
    dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
    users=users,
    visual_delay=delay,
    filter_bandpass=True,
    sample_rate=sample_rate,
    freq_cut_low=freq_cut_low,
    freq_cut_high=freq_cut_high,
    filter_order=filter_order,
)

# Time window sizes in seconds
tamanho_da_janela_seg = [1.0]

tuning_epochs = 25  # Fewer epochs for hyperparameter tuning
n_tuning_trials = 20  # Number of Optuna trials per user

for tamanho_da_janela_seg_val in tamanho_da_janela_seg:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg_val * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg_val} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"EEGNet_optuna_tuning/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg_val}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    tuning_results = []

    # Leave-one-user-out cross-validation
    for test_user_idx, test_user in enumerate(users):
        print(f"\n{'#'*80}")
        print(f"Processing User {test_user} ({test_user_idx + 1}/{len(users)})")
        print(f"{'#'*80}")
        train_users = [u for u in users if u != test_user]

        x_train = []
        labels_train = []

        # Train data (excluding test user)
        for u in train_users:
            data = all_data[u - 1]
            user_x, user_labels = extract_trials_trial_major(
                data,
                occipital_electrodes,
                indices,
                tamanho_da_janela,
                frequencias,
            )
            x_train.extend(user_x)
            labels_train.extend(user_labels)
        x_train = np.array(x_train)

        # Label mapping
        mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))}
        labels_train = torch.tensor([mapeamento[rotulo.item()] for rotulo in labels_train])

        # Convert to tensors
        X_train = torch.from_numpy(x_train.copy()).float().to(device)
        Y_train = labels_train.to(torch.long).to(device)

        print(f"X_train: {X_train.shape}")
        print(f"Y_train: {Y_train.shape}")

        # ============================================================================
        # HYPERPARAMETER TUNING with Optuna (per user, excluding test user)
        # ============================================================================
        print(f"\nStarting hyperparameter tuning for user {test_user}...")
        best_params, study = tune_hyperparameters_per_user(
            X_train,
            Y_train,
            tamanho_da_janela,
            frequencias_desejadas,
            device,
            seed,
            sample_rate=sample_rate,
            n_trials=n_tuning_trials,
            tuning_epochs=tuning_epochs,
        )

        # Store tuning results
        tuning_results.append({
            "usuario": test_user,
            "best_F1": best_params.get("F1"),
            "best_learning_rate": best_params.get("learning_rate"),
            "best_batch_size": best_params.get("batch_size"),
            "best_optimizer": best_params.get("optimizer"),
            "best_weight_decay": best_params.get("weight_decay"),
            "best_tuning_accuracy": study.best_value,
            "n_trials": len(study.trials),
        })

        print(
            f"User {test_user} Finished: Best Accuracy={study.best_value:.4f}"
        )

        # Save tuning results
        df_tuning = pd.DataFrame(tuning_results)
        df_tuning.to_csv(exp_dir.joinpath("tuning_results.csv"), index=False)

        print("-" * 80)

print("\n" + "="*100)
print("All hyperparameter tuning completed!")
