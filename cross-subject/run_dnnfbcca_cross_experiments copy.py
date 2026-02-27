"""
Run DNN+CCA cross-subject experiments with all time windows.
Uses CCA optimization per subband and subband-specific filtering.
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

# CCA parameters
num_harmonica = 5
inform_fase = 0

# Electrodes and frequencies of interest
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
users = list(range(1, 11))  # 10 users for cross-subject
frequencias_desejadas = frequencias[:8]  # 8 frequencies
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
        f"CCA_dnn_8_10/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
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

        # Prepare reference matrices for all frequencies (no window separation)
        Y_train = np.zeros(
            (tamanho_da_janela * num_trials_train, num_harmonica * 2, len(indices))
        )
        Y_test = np.zeros(
            (tamanho_da_janela * num_trials_test, num_harmonica * 2, len(indices))
        )
        for k in indices:
            y_train = reference_matrix(
                num_harmonica,
                inform_fase,
                num_trials_train,
                frequencias[k],
                fases,
                tamanho_da_janela,
            )
            Y_train[:, :, k] = y_train
            y_test = reference_matrix(
                num_harmonica,
                inform_fase,
                num_trials_test,
                frequencias[k],
                fases,
                tamanho_da_janela,
            )
            Y_test[:, :, k] = y_test

        # Extract EEG data for each frequency (before subband filtering)
        X_train = np.zeros(
            (
                tamanho_da_janela * num_trials_train,
                3,
                len(occipital_electrodes),
                len(indices),
            )
        )
        X_test = np.zeros(
            (
                tamanho_da_janela * num_trials_test,
                3,
                len(occipital_electrodes),
                len(indices),
            )
        )

        for k in range(len(indices)):
            # For training: each trial is a single window
            eeg_matrix_train = train_data[
                occipital_electrodes, :tamanho_da_janela, indices[k], :
            ].reshape(-1, len(occipital_electrodes), tamanho_da_janela)
            eeg_matrix_train = filter_signals_subbands(
                eeg_matrix_train, subban_no=3, sampling_rate=250
            )
            eeg_matrix_test = test_data[
                occipital_electrodes, :tamanho_da_janela, indices[k], :
            ].reshape(-1, len(occipital_electrodes), tamanho_da_janela)
            eeg_matrix_test = filter_signals_subbands(
                eeg_matrix_test, subban_no=3, sampling_rate=250
            )

            eeg_matrix_train = np.moveaxis(eeg_matrix_train, -1, 0)
            eeg_matrix_test = np.moveaxis(eeg_matrix_test, -1, 0)
            eeg_matrix_train = np.concatenate(eeg_matrix_train, axis=0)
            eeg_matrix_test = np.concatenate(eeg_matrix_test, axis=0)

            X_train[:, :, :, k] = eeg_matrix_train
            X_test[:, :, :, k] = eeg_matrix_test

        # CCA optimization (across all training data, per subband)
        Combinadores_Y = []
        Combinadores_X = []
        for i in range(1):
            Combinadores_Y_sub = []
            Combinadores_X_sub = []
            for k in range(len(indices)):
                Wx, Wy, _ = CCA(X_train[:, i, :, k], Y_train[:, :, k])
                Combinadores_Y_sub.append(Wy)
                Combinadores_X_sub.append(Wx)
            Combinadores_X.append(np.column_stack(Combinadores_X_sub))
            Combinadores_Y.append(np.column_stack(Combinadores_Y_sub))
        Combinadores_X = np.array(Combinadores_X)  # shape: (3, len(indices), channels)
        Combinadores_Y = np.array(Combinadores_Y)  # shape: (3, len(indices), 2*num_harmonica)

        # Split into windows
        X_teste_janelas = []
        X_treino_janelas = []
        Y_teste_janelas = []
        Y_treino_janelas = []

        for k in range(len(indices)):
            X_t, numero_janelas_teste = get_windows(
                X_test[:, :, :, k], tamanho_da_janela, include_last=False
            )
            Y_t, _ = get_windows(Y_test[:, :, k], tamanho_da_janela, include_last=False)

            X_v, numero_janelas_treino = get_windows(
                X_train[:, :, :, k], tamanho_da_janela, include_last=False
            )
            Y_v, _ = get_windows(Y_train[:, :, k], tamanho_da_janela, include_last=False)

            X_teste_janelas.append(X_t)
            Y_teste_janelas.append(Y_t)

            X_treino_janelas.append(X_v)
            Y_treino_janelas.append(Y_v)

        # Build training tensor with CCA projections
        rotulos_treinamento = []
        tensor_treinamento = np.zeros(
            [len(indices) * numero_janelas_treino, 3, len(indices), tamanho_da_janela]
        )
        cont = 0

        for m in range(len(indices)):
            for j in range(numero_janelas_treino):
                rotulos_treinamento.append(frequencias[indices[m]])
                cont_1 = 0
                for w in range(len(indices)):
                    for subband in range(3):
                        Wx = Combinadores_X[subband, :, w]
                        janela_x = X_treino_janelas[m][j][:, subband, :]
                        janela_x = janela_x - np.mean(janela_x, axis=0, keepdims=True)
                        projecao_x = np.dot(Wx, janela_x.T)
                        tensor_treinamento[cont, subband, cont_1, :] = projecao_x

                    cont_1 += 1
                cont += 1

        # Build test tensor with CCA projections
        rotulos_teste = []
        tensor_teste = np.zeros(
            [len(indices) * numero_janelas_teste, 3, len(indices), tamanho_da_janela]
        )
        cont = 0

        for m in range(len(indices)):
            for j in range(numero_janelas_teste):
                rotulos_teste.append(frequencias[indices[m]])
                cont_1 = 0
                for w in range(len(indices)):
                    for subband in range(3):
                        Wx = Combinadores_X[subband, :, w]
                        janela_x = X_teste_janelas[m][j][:, subband, :]
                        janela_x = janela_x - np.mean(janela_x, axis=0, keepdims=True)
                        projecao_x = np.dot(Wx, janela_x.T)
                        tensor_teste[cont, subband, cont_1, :] = projecao_x

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
        rotulos_teste = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in rotulos_teste
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
        model = SSVEPDNN(
            num_classes=len(frequencias_desejadas),
            channels=len(indices),
            samples=tamanho_da_janela,
            subbands=3,
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
            TensorDataset(X_teste, Y_teste), batch_size=10, shuffle=False
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
