"""
Run EEGNet with leave-one-session-out per user.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from braindecode.models import EEGNet
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path("/home/mateuschinelatto/Experiments/ssvep-bci-nn/cross-subject")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cross_subject_utils import EarlyStopping, evaluate
from benchmark_dataset import build_tensors_no_cca, load_data_from_users, load_freq_phase


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

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total
        avg_train_loss = running_loss / len(train_loader)

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

        if early_stopping is not None:
            if early_stopping(model, val_accuracy, epoch):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            if epoch == 0 or val_accuracy > max(val_accuracies[:-1]):
                torch.save(model.state_dict(), save_path)

    if early_stopping is not None:
        model = early_stopping.load_best_model(model)
        torch.save(model.state_dict(), save_path)
    else:
        model.load_state_dict(torch.load(save_path))

    return model


# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = int(42)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Load frequency and phase information
frequencias, _ = load_freq_phase()

# Preprocessing parameters
filter_order = 10
freq_cut_high = 50
freq_cut_low = 6
sample_rate = 250
delay = 160

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 36))
users_to_run = users.copy()
frequencias_desejadas = frequencias[:8]
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
)

# Time window sizes in seconds
tamanho_da_janela_seg = [1.0]

# Training parameters
epochs = 1000

for tamanho_da_janela_seg_val in tamanho_da_janela_seg:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg_val * sample_rate))
    print(f"\n{'=' * 100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg_val} s)")
    print(f"{'=' * 100}")

    exp_root = Path(
        "/home/mateuschinelatto/Experiments/ssvep-bci-nn/cross-subject/louo_experiments/models/eegnet_per_user_loso"
    )
    exp_root.mkdir(parents=True, exist_ok=True)

    for user_id in users_to_run:
        print(f"\nUser {user_id}")
        user_data = all_data[users.index(user_id)]
        num_sessions = user_data.shape[-1]

        exp_dir = exp_root.joinpath(f"user_{user_id}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        metrics = []

        for session_idx in range(num_sessions):
            print(f"Session {session_idx + 1}/{num_sessions}")

            train_data = np.delete(user_data, session_idx, axis=3)
            test_data = user_data[..., session_idx : session_idx + 1]

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

            mapeamento = {
                rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))
            }
            labels_train = torch.tensor(
                [
                    mapeamento[rotulo.item()]
                    if hasattr(rotulo, "item")
                    else mapeamento[rotulo]
                    for rotulo in labels_train
                ]
            )
            labels_test = torch.tensor(
                [
                    mapeamento[rotulo.item()]
                    if hasattr(rotulo, "item")
                    else mapeamento[rotulo]
                    for rotulo in labels_test
                ]
            )

            X_train = torch.from_numpy(x_train.copy()).float().to(device)
            Y_train = labels_train.to(torch.long).to(device)
            X_test = torch.from_numpy(x_test.copy()).float().to(device)
            Y_test = labels_test.to(torch.long).to(device)
            print(f"X_train: {X_train.shape}")
            print(f"Y_train: {Y_train.shape}")
            print(f"X_test: {X_test.shape}")
            print(f"Y_test: {Y_test.shape}")

            model = EEGNet(
                n_chans=channels_for_model,
                n_outputs=len(frequencias_desejadas),
                n_times=tamanho_da_janela,
                kernel_length=(sample_rate // 2),
                F1=8,
                D=2,
                drop_prob=0.25,
            )
            model = model.to(device)

            dataset = TensorDataset(X_train, Y_train)
            val_size = max(1, int(0.1 * len(dataset)))
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed),
            )
            test_dataset = TensorDataset(X_test, Y_test)

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
                test_dataset,
                batch_size=116,
                shuffle=False,
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

            early_stopping = EarlyStopping(
                monitor="val_accuracy",
                patience=500,
                verbose=True,
                delta=0.0001,
            )

            best_model = train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                num_epochs=epochs,
                device=device,
                save_path=exp_dir.joinpath(
                    f"best_model_session_{session_idx + 1}.pth"
                ),
                early_stopping=early_stopping,
            )

            accuracy, recall, f1, _ = evaluate(best_model, test_loader)

            metrics.append(
                {
                    "usuario": user_id,
                    "session": session_idx + 1,
                    "acuracia": accuracy,
                    "recall": recall,
                    "f1-score": f1,
                }
            )

            print(
                f"Finished: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
            )

            metrics_path = exp_dir.joinpath("metricas.csv")
            pd.DataFrame([metrics[-1]]).to_csv(
                metrics_path,
                mode="a",
                header=not metrics_path.exists(),
                index=False,
            )

            print("-" * 50)

        print(f"User {user_id} completed for window size {tamanho_da_janela_seg_val} s.")

print("\nAll experiments completed!")
