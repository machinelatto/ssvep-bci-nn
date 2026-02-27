"""
Run EEGNet+CCA cross-subject experiments with all time windows.
Uses CCA optimization and projections similar to the notebook approach.
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
    evaluate,
    get_windows,
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
freq_cut_high = 50
freq_cut_low = 6
delay = 160

# CCA parameters
num_harmonica = 3
inform_fase = 0

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 36))  # 35 users for cross-subject
frequencias_desejadas = frequencias[:]  # 8 frequencies
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
tamanho_da_janela_seg_list = [0.4, 0.6, 0.8, 1.0]

# Training parameters
epochs = 1000

# Experiment loop for each time window
for tamanho_da_janela_seg in tamanho_da_janela_seg_list:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"CCA_eegnet/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metricas_usuarios = []
    # Leave-one-user-out cross-validation
    for test_user_idx, test_user in enumerate(users):
        print(f"\nProcessing User {test_user}")
        train_users = [u for u in users if u != test_user]
        print(f"Train Users: {train_users}")

        # Concatenate training data from all train_users
        train_data = np.concatenate(
            [all_data[users.index(u)] for u in train_users], axis=-1
        )  # shape: (channels, samples, freqs, trials)
        test_data = all_data[test_user_idx]

        num_canais, _, num_freqs, num_trials_train = train_data.shape
        num_trials_test = test_data.shape[-1]

        # Prepare reference matrices for all frequencies
        Y_train = np.zeros(
            (num_harmonica * 2, tamanho_da_janela * num_trials_train, len(indices))
        )
        # Prepare CCA projection matrix
        X_train = np.zeros(
            (len(occipital_electrodes), tamanho_da_janela * num_trials_train, len(indices))
        )

        # Prepare training and test tensors
        X_train_windows = np.zeros(
            (num_trials_train * len(indices), len(occipital_electrodes), tamanho_da_janela)
        )
        X_test_windows = np.zeros(
            (num_trials_test * len(indices), len(occipital_electrodes), tamanho_da_janela)
        )

        # Prepare labels
        labels_train = []
        labels_test = []

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
            eeg_matrix_test_windows = test_data[
                occipital_electrodes, :tamanho_da_janela, indices[k], :
            ]
            # Reshape to (num_channels, num_timepoints*num_trials) - keep standard BCI format
            eeg_matrix_train = eeg_matrix_train_windows.reshape(len(occipital_electrodes), -1)
            X_train[:, :, k] = eeg_matrix_train

            # Add to window tensors (for later CCA projections and training)
            X_train_windows[k * num_trials_train : (k + 1) * num_trials_train] = eeg_matrix_train_windows.transpose(2, 0, 1)
            X_test_windows[k * num_trials_test : (k + 1) * num_trials_test] = eeg_matrix_test_windows.transpose(2, 0, 1)

            # Add labels
            labels_train.extend([frequencias[indices[k]]] * num_trials_train)
            labels_test.extend([frequencias[indices[k]]] * num_trials_test)


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


        tensor_treinamento = np.zeros(
            [len(indices) * num_trials_train, len(indices), tamanho_da_janela]
        )
        for j in range(num_trials_train):
            for k in range(len(indices)):
                janela_x = X_train_windows[k * num_trials_train + j]  # shape: (num_channels, num_timepoints)
                janela_x = janela_x - np.mean(janela_x, axis=1, keepdims=True)
                # Apply ALL CCA components
                for freq_idx in range(len(indices)):
                    Wx = Combinadores_X[:, freq_idx]
                    projecao_x = np.dot(Wx, janela_x)  # (num_channels,) @ (num_channels, num_timepoints) = (num_timepoints,)
                    tensor_treinamento[k * num_trials_train + j, freq_idx, :] = projecao_x

        tensor_teste = np.zeros(
            [len(indices) * num_trials_test, len(indices), tamanho_da_janela]
        )
        for j in range(num_trials_test):
            for k in range(len(indices)):
                janela_x = X_test_windows[k * num_trials_test + j]  # shape: (num_channels, num_timepoints)
                janela_x = janela_x - np.mean(janela_x, axis=1, keepdims=True)
                # Apply ALL CCA components, not just the one matching the frequency
                for freq_idx in range(len(indices)):
                    Wx = Combinadores_X[:, freq_idx]
                    projecao_x = np.dot(Wx, janela_x)  # (num_channels,) @ (num_channels, num_timepoints) = (num_timepoints,)
                    tensor_teste[k * num_trials_test + j, freq_idx, :] = projecao_x

        # Map labels to indices
        mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))}
        rotulos_treinamento = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in labels_train
            ]
        )
        rotulos_teste = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in labels_test
            ]
        )

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
            n_chans=len(indices),  # Number of CCA components (same as number of frequencies)
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
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(
            TensorDataset(X_teste, Y_teste), batch_size=256, shuffle=False
        )

        # Train
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

        # Save metrics
        df_metricas = pd.DataFrame(metricas_usuarios)
        df_metricas.to_csv(exp_dir.joinpath("metricas.csv"), index=False)

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho_da_janela_seg} s.")

print("\n" + "="*100)
print("All experiments completed!")
