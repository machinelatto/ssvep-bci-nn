"""
Run EEGNet cross-subject experiments with full dataset (35 users, 40 frequencies)
for all time lengths (0.4s, 0.6s, 0.8s, 1.0s).
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

from cross_subject_utils import (
    plot_learning_curves,
    evaluate,
    load_data_from_users,
)


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
    
    # plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    model.load_state_dict(best_model)
    return model


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
freq_cut_high = 70
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
tamanho_da_janela_seg = [0.4, 0.6, 0.8, 1.0]

# Training parameters
epochs = 1000

for tamanho_da_janela_seg_val in tamanho_da_janela_seg:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg_val * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg_val} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"EEGNet_full_dataset/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg_val}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metricas_usuarios = []

    # Leave-one-user-out cross-validation
    for user in range(1, len(all_data) + 1):
        print(f"\nProcessing User {user}")
        n_freqs_sel = len(indices)
        metricas_crossval = []
        users_train = [u for u in range(1, len(all_data) + 1) if u != user]
        user_test = user

        x_train = []
        labels_train = []
        x_test = []
        labels_test = []

        # Train data
        for u in users_train:
            data = all_data[u - 1]
            for session in range(data.shape[3]):
                for freq in range(len(indices)):
                    eeg_trial = data[occipital_electrodes, :, indices[freq], session]
                    eeg_trial = eeg_trial[:, :tamanho_da_janela]
                    x_train.append(eeg_trial)
                    labels_train.extend([frequencias[freq]])
        x_train = np.array(x_train)

        # Test data
        data = all_data[user_test - 1]
        for session in range(data.shape[3]):
            for freq in range(len(indices)):
                eeg_trial = data[occipital_electrodes, :, indices[freq], session]
                eeg_trial = eeg_trial[:, :tamanho_da_janela]
                x_test.append(eeg_trial)
                labels_test.extend([frequencias[freq]])
        x_test = np.array(x_test)

        # Label mapping
        mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))}
        labels_train = torch.tensor([mapeamento[rotulo.item()] for rotulo in labels_train])
        labels_test = torch.tensor([mapeamento[rotulo.item()] for rotulo in labels_test])

        # Convert to tensors
        X_train = torch.from_numpy(x_train.copy()).float().to(device)
        X_test = torch.from_numpy(x_test.copy()).float().to(device)
        Y_train = labels_train.to(torch.long).to(device)
        Y_test = labels_test.to(torch.long).to(device)

        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"Y_train: {Y_train.shape}")
        print(f"Y_test: {Y_test.shape}")

        # Configure model and training
        model = EEGNet(
            n_chans=9,
            n_outputs=len(frequencias_desejadas),
            n_times=tamanho_da_janela,
            kernel_length=(tamanho_da_janela // 2),
        )
        model = model.to(device)

        dataset = TensorDataset(X_train, Y_train)
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(
            TensorDataset(X_test, Y_test), batch_size=10, shuffle=False
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Train
        best_model = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=epochs,
            device=device,
            save_path=exp_dir.joinpath(f"best_model_user_{user}.pth"),
        )

        # Evaluate
        accuracy, recall, f1, cm = evaluate(best_model, test_loader)

        # Store metrics
        metricas_crossval.append(
            {
                "usuario": user,
                "acuracia": accuracy,
                "recall": recall,
                "f1-score": f1,
                "confusion_matrix": cm,
            }
        )

        print(
            f"User {user} Finished: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        # Save metrics
        metricas_usuarios.extend(metricas_crossval)
        df_metricas = pd.DataFrame(metricas_usuarios)
        df_metricas.to_csv(exp_dir.joinpath("metricas.csv"), index=False)

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho_da_janela_seg_val} s.")

print("\nAll experiments completed!")
