"""
Run EEGNet+CCA cross-subject experiments with all time windows.
Uses CCA optimization and projections similar to the notebook approach.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import copy
import gc
from braindecode.models import EEGNet

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_dataset import (
    load_data_from_users,
    load_freq_phase,
    build_tensors_with_cca,
)

from cross_subject_utils import (
    evaluate,
    EarlyStopping,
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
    early_stopping=None,
    use_amp=False,
):
    """Train the model with optional early stopping based on validation metrics."""
    model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in tqdm(range(num_epochs)):
        # Training Phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
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

        # Print progress (less verbose for scripts)
        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        # Early stopping check (if provided)
        if early_stopping is not None:
            if early_stopping(model, val_accuracy, epoch):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            # Fallback: save best model if no early stopping
            if epoch == 0 or val_accuracy > max(val_accuracies[:-1]):
                torch.save(model.state_dict(), save_path)

    # Load best model (either from early stopping or training)
    if early_stopping is not None:
        model = early_stopping.load_best_model(model)
        # torch.save(model.state_dict(), save_path)
    else:
        model.load_state_dict(torch.load(save_path))

    return model


# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = False
seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

print(f"Using device: {device}")

# Load frequency and phase information
frequencias, fases = load_freq_phase()

# Preprocessing parameters
sample_rate = 250
filter_order = 10
freq_cut_high = 80
freq_cut_low = 6
delay = 160

# CCA parameters
num_harmonica = 5
inform_fase = 0
cca_reg = 0

# Electrodes and frequencies of interest
all_occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 36))  # 35 users for cross-subject
users_to_run = users.copy()  # Ex.: [1, 5, 10]
frequencias_desejadas = frequencias[:]  # All frequencies
indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

# Optional CAR configuration on loaded data
apply_car = True
car_reference_channels = occipital_electrodes
car_target_channels = occipital_electrodes

print("Users of interest:", users)
print("Users to run:", users_to_run)
print("Frequencies of interest:", frequencias_desejadas)
print("Indices of frequencies of interest:", indices)

# Load all data
print("\nLoading data from all users...")
all_data = load_data_from_users(
    users=users,
    dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
    visual_delay=delay,
    filter_bandpass=True,
    apply_car=apply_car,
    car_reference_channels=car_reference_channels,
    car_target_channels=car_target_channels,
    sample_rate=sample_rate,
    freq_cut_low=freq_cut_low,
    freq_cut_high=freq_cut_high,
    filter_order=filter_order,
    normalize=True,
)

# Keep only the channels used by the model and detach them as float32 arrays.
# CAR has already been applied using the original channel indices above.
all_data = [
    np.asarray(data[occipital_electrodes, :, :, :], dtype=np.float32)
    for data in all_data
]
model_electrodes = np.arange(len(occipital_electrodes))

# Time window sizes in seconds
tamanho_da_janela_seg_list = [0.6, 0.4]  # in seconds

# Training parameters
epochs = 1000

# Experiment loop for each time window
for tamanho_da_janela_seg in tamanho_da_janela_seg_list:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"/home/mateuschinelatto/Experiments/ssvep-bci-nn/cross-subject/louo_experiments/thesis/cca_eegnet_4_2/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metricas_usuarios = []
    # Leave-one-user-out cross-validation
    for test_user in users_to_run:
        print(f"\nProcessing User {test_user}")
        train_users = [u for u in users if u != test_user]
        print(f"Train Users: {train_users}")

        # Concatenate training data from all train_users
        train_data = np.concatenate(
            [all_data[users.index(u)] for u in train_users], axis=-1
        )  # shape: (channels, samples, freqs, trials)
        test_data = all_data[users.index(test_user)]
        print("Building tensors with CCA...")
        tensor_treinamento, tensor_teste, labels_train, labels_test, channels_for_model = (
            build_tensors_with_cca(
                train_data,
                test_data,
                model_electrodes,
                frequencias,
                fases,
                indices,
                num_harmonica,
                inform_fase,
                tamanho_da_janela,
                mean_center=True,
                apply_subband_filter=False,
                cca_reg=cca_reg,
            )
        )
        del train_data, test_data

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

        X_treino = torch.tensor(tensor_treinamento, dtype=torch.float32)
        X_teste = torch.tensor(tensor_teste, dtype=torch.float32)
        Y_treino = torch.tensor(rotulos_treinamento, dtype=torch.long)
        Y_teste = torch.tensor(rotulos_teste, dtype=torch.long)
        del tensor_treinamento, tensor_teste, labels_train, labels_test
        del rotulos_treinamento, rotulos_teste
        print(f"X_train: {X_treino.shape}")
        print(f"X_test: {X_teste.shape}")
        print(f"Y_train: {Y_treino.shape}")
        print(f"Y_test: {Y_teste.shape}")

        # Model setup
        model = EEGNet(
            n_chans=channels_for_model,
            n_outputs=len(frequencias_desejadas),
            n_times=tamanho_da_janela,
            kernel_length=(sample_rate // 2),
            F1=4,
            D=2,
            drop_prob=0.25
        )

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        # optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Initialize early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=250,
            verbose=False,
            delta=0.0001
        )

        dataset = TensorDataset(X_treino, Y_treino)
        train_size = int(0.9 * len(dataset))
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
            early_stopping=early_stopping,
            use_amp=use_amp,
        )

        # Evaluate
        accuracy, recall, f1, cm = evaluate(best_model, test_loader)

        metricas_usuarios.append(
            {
                "usuario": test_user,
                "acuracia": accuracy,
                "recall": recall,
                "f1-score": f1,
                # "confusion_matrix": cm,
            }
        )
        print(
            f"User {test_user} Finished: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        # Free large per-user allocations before moving to the next LOO fold.
        del (
            X_treino,
            X_teste,
            Y_treino,
            Y_teste,
            dataset,
            train_dataset,
            val_dataset,
            train_loader,
            val_loader,
            test_loader,
            model,
            best_model,
            optimizer,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
