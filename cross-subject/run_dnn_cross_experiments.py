"""
Run DNN cross-subject experiments with all time windows.
Uses subband filtering with no CCA optimization.
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

from cross_subject_utils import (
    evaluate,
    load_data_from_users,
    filter_signals_subbands,
)


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
delay = 160

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 36))  # 10 users for cross-subject
frequencias_desejadas = frequencias[:]  # 8 frequencies
indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

print("Users of interest:", users)
print("Frequencies of interest:", frequencias_desejadas)
print("Indices of frequencies of interest:", indices)

# Load all data (no bandpass filtering)
print("\nLoading data from all users...")
all_data = load_data_from_users(
    dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
    users=users,
    visual_delay=delay,
    filter_bandpass=False,
    sample_rate=sample_rate,
)

# Time window sizes in seconds
tamanho_da_janela_seg_list = [0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0]

# Training parameters
epochs = 1000

# Experiment loop for each time window
for tamanho_da_janela_seg in tamanho_da_janela_seg_list:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"dnn_full_dataset/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metricas_usuarios = []

    # Leave-one-user-out cross-validation
    for user in range(1, len(all_data) + 1):
        print(f"\nProcessing User {user}")
        users_train = [u for u in range(1, len(all_data) + 1) if u != user]
        user_test = user

        x_train = []
        labels_train = []
        x_test = []
        labels_test = []

        # Train users
        for u in users_train:
            data = all_data[u - 1]
            for session in range(data.shape[3]):
                for freq in range(len(indices)):
                    eeg_trial = data[occipital_electrodes, :, indices[freq], session]
                    eeg_trial = eeg_trial[:, :tamanho_da_janela]
                    x_train.append(eeg_trial)
                    labels_train.extend([frequencias[freq]])
        x_train = np.array(x_train)
        # Apply subband filtering: [batch, subbands, channels, time]
        x_train = filter_signals_subbands(x_train, subban_no=3, sampling_rate=250)

        # Test user
        data = all_data[user_test - 1]
        for session in range(data.shape[3]):
            for freq in range(len(indices)):
                eeg_trial = data[occipital_electrodes, :, indices[freq], session]
                eeg_trial = eeg_trial[:, :tamanho_da_janela]
                x_test.append(eeg_trial)
                labels_test.extend([frequencias[freq]])
        x_test = np.array(x_test)
        x_test = filter_signals_subbands(x_test, subban_no=3, sampling_rate=250)

        # Label mapping
        mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))}
        labels_train = torch.tensor([mapeamento[rotulo.item()] for rotulo in labels_train])
        labels_test = torch.tensor([mapeamento[rotulo.item()] for rotulo in labels_test])

        X_train = torch.from_numpy(x_train.copy()).float().to(device)
        X_test = torch.from_numpy(x_test.copy()).float().to(device)
        Y_train = labels_train.to(torch.long).to(device)
        Y_test = labels_test.to(torch.long).to(device)

        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"Y_train: {Y_train.shape}")
        print(f"Y_test: {Y_test.shape}")

        # Model setup
        model = SSVEPDNN(
            num_classes=len(frequencias_desejadas),
            channels=len(occipital_electrodes),
            samples=tamanho_da_janela,
            subbands=3,
        ).to(device)

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
        print(f"Training for {epochs} epochs...")
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

        metricas_usuarios.append(
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
        df_metricas = pd.DataFrame(metricas_usuarios)
        df_metricas.to_csv(exp_dir.joinpath("metricas.csv"), index=False)

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho_da_janela_seg} s.")

print("\n" + "="*100)
print("All experiments completed!")
