"""
Run EEGNet+CCA cross-subject experiments (no CCA training) with all time windows.
CCA is applied per-trial without training on aggregated data.
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
from braindecode.models import EEGNet

from cross_subject_utils import (
    plot_learning_curves,
    evaluate,
    load_data_from_users,
)
from cca import CCA, reference_matrix


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=100,
    device=0,
    save_path="best_model.pth",
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
        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )
    
    model.load_state_dict(best_model)
    return model


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
filter_order = 10
freq_cut_high = 70
freq_cut_low = 6
delay = 160

# CCA parameters
num_harmonica = 5
inform_fase = 0

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 36))  # 10 users for cross-subject
users_to_run = users.copy()  # Ex.: [1, 5, 10]
frequencias_desejadas = frequencias[:]  # 8 frequencies
indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

# Optional CAR configuration on loaded data
apply_car = False
car_reference_channels = occipital_electrodes
car_target_channels = None

print("Users of interest:", users)
print("Users to run:", users_to_run)
print("Frequencies of interest:", frequencias_desejadas)
print("Indices of frequencies of interest:", indices)

# Load all data
print("\nLoading data from all users...")
all_data = load_data_from_users(
    dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
    users=users,
    visual_delay=delay,
    filter_bandpass=True,
    apply_car=apply_car,
    car_reference_channels=car_reference_channels,
    car_target_channels=car_target_channels,
    sample_rate=sample_rate,
    freq_cut_low=freq_cut_low,
    freq_cut_high=freq_cut_high,
    filter_order=filter_order,
)

# Time window sizes in seconds
tamanho_da_janela_seg_list = [0.6, 0.8, 1.0]

# Training parameters
epochs = 1000

# Experiment loop for each time window
for tamanho_da_janela_seg in tamanho_da_janela_seg_list:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"CCA_no_train_eegnet_full_dataset/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metricas_usuarios = []

    # Leave-one-user-out cross-validation
    for test_user in users_to_run:
        # if test_user != 2 and tamanho_da_janela_seg == 0.4:
        #     continue
        print(f"\nProcessing User {test_user}")
        train_users = [u for u in users if u != test_user]
        print(f"Train Users: {train_users}")

        # Concatenate training data from all train_users
        train_data = np.concatenate(
            [all_data[users.index(u)] for u in train_users], axis=-1
        )  # shape: (channels, samples, freqs, trials)
        test_data = all_data[users.index(test_user)]

        num_canais, _, num_freqs, num_trials_train = train_data.shape
        num_trials_test = test_data.shape[-1]

        # Prepare reference matrices for all frequencies (single window, no concatenation)
        Y_train = np.zeros(
            (tamanho_da_janela, num_harmonica * 2, len(indices))
        )
        Y_test = np.zeros(
            (tamanho_da_janela, num_harmonica * 2, len(indices))
        )
        for k in indices:
            y_train = reference_matrix(
                num_harmonica,
                inform_fase,
                1,
                frequencias[k],
                fases,
                tamanho_da_janela,
            )
            Y_train[:, :, k] = y_train
            y_test = reference_matrix(
                num_harmonica,
                inform_fase,
                1,
                frequencias[k],
                fases,
                tamanho_da_janela,
            )
            Y_test[:, :, k] = y_test

        # Build tensors with per-trial CCA application (no CCA training phase)
        rotulos_treinamento = []
        tensor_treinamento = np.zeros(
            [len(indices) * num_trials_train, len(indices), tamanho_da_janela]
        )
        rotulos_teste = []
        tensor_teste = np.zeros(
            [len(indices) * num_trials_test, len(indices), tamanho_da_janela]
        )

        for k in range(len(indices)):
            for session in range(num_trials_train):
                eeg_matrix_train = train_data[
                    occipital_electrodes, :tamanho_da_janela, indices[k], session
                ]
                eeg_matrix_train = np.transpose(eeg_matrix_train)
                rotulos_treinamento.append(k)
                for freq in range(len(indices)):
                    Wx, Wy, corr = CCA(eeg_matrix_train, Y_train[:, :, freq])
                    tensor_treinamento[k * num_trials_train + session, freq, :] = np.dot(Wx, eeg_matrix_train.T)

            for session in range(num_trials_test):
                eeg_matrix_test = test_data[
                    occipital_electrodes, :tamanho_da_janela, indices[k], session
                ]
                eeg_matrix_test = np.transpose(eeg_matrix_test)
                rotulos_teste.append(k)
                for freq in range(len(indices)):
                    Wx, Wy, corr = CCA(eeg_matrix_test, Y_test[:, :, freq])
                    tensor_teste[k * num_trials_test + session, freq, :] = np.dot(Wx, eeg_matrix_test.T)

        # Convert to tensors
        X_treino = torch.tensor(tensor_treinamento, dtype=torch.float32).to(device)
        X_teste = torch.tensor(tensor_teste, dtype=torch.float32).to(device)
        Y_treino = torch.tensor(rotulos_treinamento, dtype=torch.long).to(device)
        Y_teste = torch.tensor(rotulos_teste, dtype=torch.long).to(device)
        print(f"X_train: {X_treino.shape}")
        print(f"X_test: {X_teste.shape}")
        print(f"Y_train: {Y_treino.shape}")
        print(f"Y_test: {Y_teste.shape}")

        # Model setup
        model = EEGNet(
            n_chans=len(indices),
            n_outputs=len(frequencias_desejadas),
            n_times=tamanho_da_janela,
            kernel_length=(tamanho_da_janela // 2) if tamanho_da_janela > 2 else 1,
        )

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        dataset = TensorDataset(X_treino, Y_treino)
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(X_teste, Y_teste),
            batch_size=10,
            shuffle=False,
        )

        # # Train
        print(f"Training for {epochs} epochs...")
        best_model = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=epochs,
            device=device,
            save_path=exp_dir.joinpath(f"best_model_user_{test_user}.pth"),
        )

        # Evaluate
        # model.load_state_dict(torch.load(exp_dir.joinpath(f"best_model_user_{test_user}.pth")))
        accuracy, recall, f1, cm = evaluate(best_model, test_loader)

        metricas_usuarios.append(
            {
                "usuario": test_user,
                "acuracia": accuracy,
                "recall": recall,
                "f1-score": f1,
                "confusion_matrix": cm,
            }
        )
        print(
            f"User {test_user} Finished: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        # Save metrics (append to support restarting failed runs)
        metrics_path = exp_dir.joinpath("metricas.csv")
        pd.DataFrame([metricas_usuarios[-1]]).to_csv(
            metrics_path,
            mode="a",
            header=not metrics_path.exists(),
            index=False,
        )

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho_da_janela_seg} s.")

print("\n" + "="*100)
print("All experiments completed!")
