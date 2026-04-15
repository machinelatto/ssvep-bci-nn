"""
Run EEGNet+FBCCA cross-subject experiments with all time windows.
Uses CCA optimization per subband and FBCCA feature projections.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from braindecode.models import EEGNet

from cross_subject_utils import (
    evaluate,
    EarlyStopping,
    load_data_from_users,
)
from benchmark_dataset import build_tensors_with_fbcca, build_tensors_with_fbcca_joint, load_freq_phase


class EEGNetWithSubbandMerge(nn.Module):
    """Merge FBCCA subbands with a learnable 1x1 conv, then run EEGNet."""

    def __init__(self, subbands, n_chans, n_outputs, n_times, kernel_length, F1=8, D=2, drop_prob=0.3):
        super().__init__()
        self.subband_combination = nn.Conv2d(subbands, 1, kernel_size=(1, 1), bias=False)
        # Init weights to average merge
        with torch.no_grad():
            self.subband_combination.weight.fill_(1.0 / subbands)

        self.eegnet = EEGNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            kernel_length=kernel_length,
            F1=F1,
            D=D,
            drop_prob=drop_prob,
        )

    def forward(self, x):
        # x: [batch, subbands, channels, time]
        x = self.subband_combination(x)  # [batch, 1, channels, time]
        x = x.squeeze(1)  # [batch, channels, time]
        return self.eegnet(x)


def extract_subband_weights(model):
    """Return learned subband merge weights as a 1D numpy array."""
    return model.subband_combination.weight.detach().cpu().view(-1).numpy()


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
        # Training phase
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

        # Validation phase
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
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
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
    else:
        model.load_state_dict(torch.load(save_path))

    return model


# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

print(f"Using device: {device}")

# Load frequency and phase information
frequencias, fases = load_freq_phase()

# Preprocessing parameters
sample_rate = 250
delay = 160

# CCA parameters
num_harmonica = 3
inform_fase = 0
subban_no = 3

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 36))  # 35 users for cross-subject
users_to_run = list(range(8, 36))  # Ex.: [1, 5, 10]
frequencias_desejadas = frequencias[:]  # 8 frequencies
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
    filter_bandpass=False,
    apply_car=apply_car,
    car_reference_channels=car_reference_channels,
    car_target_channels=car_target_channels,
    sample_rate=sample_rate,
)

# Time window sizes in seconds
tamanho_da_janela_seg_list = [1.0]

# Training parameters
epochs = 1000

# Experiment loop for each time window
for tamanho_da_janela_seg in tamanho_da_janela_seg_list:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))
    print(f"\n{'=' * 100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'=' * 100}")

    exp_dir = Path(
        f"35_40_optimized/FBCCA_ALL_EEGNET/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metricas_usuarios = []
    pesos_subband_usuarios = []

    # Leave-one-user-out cross-validation
    for test_user in users_to_run:
        print(f"\nProcessing User {test_user}")
        train_users = [u for u in users if u != test_user]
        print(f"Train Users: {train_users}")

        # Concatenate training data from all train_users
        train_data = np.concatenate(
            [all_data[users.index(u)] for u in train_users], axis=-1
        )
        test_data = all_data[users.index(test_user)]

        tensor_treinamento, tensor_teste, labels_train, labels_test, channels_for_model = (
            build_tensors_with_fbcca_joint(
                train_data,
                test_data,
                occipital_electrodes,
                frequencias,
                fases,
                indices,
                num_harmonica,
                inform_fase,
                tamanho_da_janela,
                mean_center=True,
                subban_no=subban_no,
                sampling_rate=sample_rate,
            )
        )

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
        model = EEGNetWithSubbandMerge(
            subbands=subban_no,
            n_chans=channels_for_model,
            n_outputs=len(frequencias_desejadas),
            n_times=tamanho_da_janela,
            kernel_length=(sample_rate // 2),
            F1=8,
            D=2,
            drop_prob=0.25,
        )

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.000075)

        # Initialize early stopping
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=500,
            verbose=False,
            delta=0.0001,
        )

        dataset = TensorDataset(X_treino, Y_treino)
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
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
        )

        learned_weights = extract_subband_weights(best_model)
        print(
            "Learned subband weights (user "
            f"{test_user}): {[float(f'{w:.6f}') for w in learned_weights]}"
        )

        # Evaluate
        accuracy, recall, f1, cm = evaluate(best_model, test_loader)

        user_metrics = {
            "usuario": test_user,
            "acuracia": accuracy,
            "recall": recall,
            "f1-score": f1,
            # "confusion_matrix": cm,
        }

        metricas_usuarios.append(user_metrics)
        pesos_subband_usuarios.extend(
            [
                {
                    "usuario": test_user,
                    "subband": sub_idx,
                    "peso": float(weight),
                }
                for sub_idx, weight in enumerate(learned_weights, start=1)
            ]
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
        df_pesos = pd.DataFrame(pesos_subband_usuarios)
        df_pesos.to_csv(exp_dir.joinpath("subband_weights.csv"), index=False)

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho_da_janela_seg} s.")

print("\n" + "=" * 100)
print("All experiments completed!")
