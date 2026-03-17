"""
Run DNN+CCA cross-subject experiments with Optuna hyperparameter tuning.
Uses CCA optimization per frequency and subband-specific filtering.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import copy
import scipy.io
import torch.nn.functional as F
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from cross_subject_utils import (
    evaluate,
    get_windows,
    load_data_from_users,
    filter_signals_subbands,
)
from cca import CCA, reference_matrix


class SSVEPDNN(nn.Module):
    """SSVEP Deep Neural Network with subband processing."""
    def __init__(self, num_classes=40, channels=9, samples=250, subbands=3):
        super(SSVEPDNN, self).__init__()
        # [batch, subbands, channels, time]
        # Subband combination layer
        self.subband_combination = nn.Conv2d(
            subbands, 1, kernel_size=(1, 1), bias=False
        )
        # Channel combination layer
        self.channel_combination = nn.Conv2d(1, 120, kernel_size=(channels, 1))
        # First dropout
        self.drop1 = nn.Dropout(0.1)
        # Third layer - Time convolution
        self.third_conv = nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2))
        # Second dropout
        self.drop2 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        # 4th conv - FIR filtering
        self.fourth_conv = nn.Conv2d(120, 120, kernel_size=(1, 10), padding="same")
        self.drop3 = nn.Dropout(0.95)

        # Fully connected layer - Classifier
        self.fc = nn.Linear(120 * (samples // 2), num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.subband_combination.weight.fill_(1.0)

    def forward(self, x):
        # x shape: [batch, subbands, channels, time]
        x = self.subband_combination(x)  # [batch, 1, channels, time]
        x = self.channel_combination(x)  # [batch, 120, 1, time]
        x = self.drop1(x)
        x = self.third_conv(x)  # [batch, 120, 1, time/2]
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fourth_conv(x)  # [batch, 120, 1, time/2]
        x = self.drop3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # [batch, num_classes]
        output = F.softmax(x, dim=1)
        return output


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
    """Train the model with early stopping based on validation accuracy."""
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

        # Save if best val acc
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)

        # Print progress (less verbose for scripts)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        # Optuna pruning
        if trial is not None:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    model.load_state_dict(best_model)
    return model, best_val_accuracy


def objective_per_user(
    trial: Trial,
    X_train,
    Y_train,
    tamanho_da_janela,
    device,
    seed,
    num_epochs=25,
):
    """
    Optuna objective function for hyperparameter tuning per user.
    
    Suggests hyperparameters and returns validation accuracy.
    """
    # Suggest hyperparameters
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

    # Create model
    model = SSVEPDNN(
        num_classes=40,
        channels=40,
        samples=tamanho_da_janela,
        subbands=3,
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
            save_path="/tmp/trial_model_dnncca.pth",
            trial=trial,
        )
        return best_val_accuracy
    except optuna.TrialPruned:
        raise


def tune_hyperparameters_per_user(
    X_train,
    Y_train,
    tamanho_da_janela,
    device,
    seed,
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
            device,
            seed,
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

print(f"Using device: {device}")

# Load frequency and phase information
freq_phase_path = "/home/mateuschinelatto/Experiments/data/benchmark/Freq_Phase.mat"
freq_phase = scipy.io.loadmat(freq_phase_path)
frequencias = np.round(freq_phase["freqs"], 2).ravel()
fases = freq_phase["phases"]

# Preprocessing parameters
sample_rate = 250
delay = 160

# CCA parameters
num_harmonica = 3
inform_fase = 0

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 36))  # 35 users for cross-subject
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
    filter_bandpass=False,
    sample_rate=sample_rate,
)

# Time window sizes in seconds
tamanho_da_janela_seg_list = [1.0]

# Training parameters
tuning_epochs = 25  # Fewer epochs for hyperparameter tuning
n_tuning_trials = 20  # Number of Optuna trials per user

# Experiment loop for each time window
for tamanho_da_janela_seg in tamanho_da_janela_seg_list:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"CCA_dnn_optuna/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    tuning_results = []

    # Leave-one-user-out cross-validation
    for test_user_idx, test_user in enumerate(users):
        print(f"\n{'#'*80}")
        print(f"Processing User {test_user} ({test_user_idx + 1}/{len(users)})")
        print(f"{'#'*80}")
        train_users = [u for u in users if u != test_user]
        print(f"Train Users: {len(train_users)} users")

        # Concatenate training data from all train_users
        train_data = np.concatenate(
            [all_data[users.index(u)] for u in train_users], axis=-1
        )  # shape: (channels, samples, freqs, trials)

        num_canais, _, num_freqs, num_trials_train = train_data.shape

        # Prepare reference matrices for all frequencies
        Y_train = np.zeros(
            (num_harmonica * 2, tamanho_da_janela * num_trials_train, len(indices))
        )
        # Prepare CCA projection matrix
        X_train = np.zeros(
            (len(occipital_electrodes), tamanho_da_janela * num_trials_train, len(indices))
        )

        # Prepare training tensors
        X_train_windows = np.zeros(
            (num_trials_train * len(indices), len(occipital_electrodes), tamanho_da_janela)
        )

        # Prepare labels
        labels_train = []

        for k in range(len(indices)):
            # Generate reference signals for this frequency
            y_train = reference_matrix(
                num_harmonica,
                inform_fase,
                num_trials_train,
                frequencias[indices[k]],
                fases,
                tamanho_da_janela,
            )
            Y_train[:, :, k] = y_train

            # Extract training data for this frequency
            eeg_matrix_train_windows = train_data[
                occipital_electrodes, :tamanho_da_janela, indices[k], :
            ] # shape: (num_channels, num_timepoints, num_trials)
            # Flatten as trial-major blocks: [trial0 all time][trial1 all time]...
            # so ordering matches reference_matrix tiling across sessions.
            eeg_matrix_train = eeg_matrix_train_windows.transpose(0, 2, 1).reshape(
                len(occipital_electrodes), -1
            )
            X_train[:, :, k] = eeg_matrix_train

            # Add to window tensors (for later CCA projections and training)
            X_train_windows[k * num_trials_train : (k + 1) * num_trials_train] = eeg_matrix_train_windows.transpose(2, 0, 1)

            # Add labels
            labels_train.extend([frequencias[indices[k]]] * num_trials_train)

        # CCA optimization (across all training data)
        Combinadores_Y = []
        Combinadores_X = []
        correlacoes_max = []
        for k in range(len(indices)):
            Wx, Wy, corr = CCA(X_train[:, :, k], Y_train[:, :, k])
            Combinadores_Y.append(Wy)
            Combinadores_X.append(Wx)
            correlacoes_max.append(corr)
        Combinadores_X = np.column_stack(Combinadores_X)
        Combinadores_Y = np.column_stack(Combinadores_Y)

        # Split into windows
        X_treino_janelas = []
        Y_treino_janelas = []

        for k in range(len(indices)):
            X_v, numero_janelas_treino = get_windows(
                X_train[:, :, k], tamanho_da_janela, include_last=False
            )
            Y_v, _ = get_windows(Y_train[:, :, k], tamanho_da_janela, include_last=False)

            X_treino_janelas.append(X_v)
            Y_treino_janelas.append(Y_v)

        # Build training tensor with CCA projections
        rotulos_treinamento = []
        tensor_treinamento = np.zeros(
            [len(indices) * numero_janelas_treino, len(indices), tamanho_da_janela]
        )
        cont = 0

        for m in range(len(indices)):
            for j in range(numero_janelas_treino):
                janela_x = X_treino_janelas[m][j]  # shape: (num_channels, num_timepoints)
                rotulos_treinamento.append(frequencias[indices[m]])
                cont_1 = 0
                for w in range(len(indices)):
                    Wx = Combinadores_X[:, w]
                    janela_x = janela_x - np.mean(janela_x, axis=1, keepdims=True)
                    projecao_x = np.dot(Wx, janela_x)  # (num_channels,) @ (num_channels, num_timepoints) = (num_timepoints,)
                    tensor_treinamento[cont, cont_1, :] = projecao_x
                    cont_1 += 1
                cont += 1

        # Map labels to indices
        mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))}
        rotulos_treinamento = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in rotulos_treinamento
            ]
        )

        # Apply subband filtering to tensors
        tensor_treinamento = filter_signals_subbands(
            tensor_treinamento, subban_no=3, sampling_rate=250
        )

        X_treino = torch.tensor(tensor_treinamento, dtype=torch.float32).to(device)
        Y_treino = torch.tensor(rotulos_treinamento, dtype=torch.long).to(device)

        print(f"X_train: {X_treino.shape}")
        print(f"Y_train: {Y_treino.shape}")

        # ============================================================================
        # HYPERPARAMETER TUNING with Optuna (per user, excluding test user)
        # ============================================================================
        print(f"\nStarting hyperparameter tuning for user {test_user}...")
        best_params, study = tune_hyperparameters_per_user(
            X_treino,
            Y_treino,
            tamanho_da_janela,
            device,
            seed,
            n_trials=n_tuning_trials,
            tuning_epochs=tuning_epochs,
        )

        # Store tuning results
        tuning_results.append({
            "usuario": test_user,
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

    print(f"\nHyperparameter tuning completed for window size {tamanho_da_janela_seg} s.")
    print(f"Results saved to {exp_dir}")

print("\n" + "="*100)
print("All hyperparameter tuning completed!")
