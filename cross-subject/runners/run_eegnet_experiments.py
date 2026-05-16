"""
Run EEGNet cross-subject experiments with full dataset (35 users, 40 frequencies)
for all time lengths.
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
from braindecode.models import EEGNet

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_dataset import (
    load_data_from_users,
    load_freq_phase,
    build_tensors_no_cca,
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
):
    """Train the model with optional early stopping based on validation metrics."""
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

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
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
seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Load frequency and phase information
frequencias, _ = load_freq_phase()

# Preprocessing parameters
filter_order = 10
freq_cut_high = 90
freq_cut_low = 6
sample_rate = 250
delay = 160

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 11))  # 35 users (full dataset)
users_to_run = users.copy()  # Ex.: [1, 5, 10]
frequencias_desejadas = frequencias[:8]  # First 8 frequencies
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

# Time window sizes in seconds
tamanho_da_janela_seg_list = [0.4, 0.6, 0.8, 1.0]

# Training parameters
epochs = 1000

for tamanho_da_janela_seg_val in tamanho_da_janela_seg_list:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg_val * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg_val} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"/home/mateuschinelatto/Experiments/ssvep-bci-nn/cross-subject/louo_experiments/models/EEGNET_8_2_CAR/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg_val}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics = []

    # Leave-one-user-out cross-validation
    for test_user in users_to_run:
        print(f"\nProcessing User {test_user}")
        train_users = [u for u in users if u != test_user]

        train_data = np.concatenate(
            [all_data[users.index(u)] for u in train_users], axis=-1
        )
        test_data = all_data[users.index(test_user)]

        x_train, x_test, labels_train, labels_test, channels_for_model = (
            build_tensors_no_cca(
                train_data,
                test_data,
                occipital_electrodes,
                frequencias,
                indices,
                tamanho_da_janela,
                apply_subband_filter=False,
            )
        )

        # Label mapping
        mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))}
        labels_train = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in labels_train
            ]
        )
        labels_test = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in labels_test
            ]
        )

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
            n_chans=channels_for_model,
            n_outputs=len(frequencias_desejadas),
            n_times=tamanho_da_janela,
            kernel_length=(sample_rate // 2),
            F1=8,
            drop_prob=0.25,
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
            TensorDataset(X_test, Y_test),
            batch_size=10,
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        # optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Initialize early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=500,
            verbose=True,
            delta=0.0001
        )

        # Train
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
        )

        # Evaluate
        accuracy, recall, f1, cm = evaluate(best_model, test_loader)

        # Store metrics
        metrics.append(
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

        # Save metrics (append to support restarting failed runs)
        metrics_path = exp_dir.joinpath("metricas.csv")
        pd.DataFrame([metrics[-1]]).to_csv(
            metrics_path,
            mode="a",
            header=not metrics_path.exists(),
            index=False,
        )

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho_da_janela_seg_val} s.")

print("\nAll experiments completed!")
