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
users = list(range(1, 36))
frequencias_desejadas = frequencias[:]
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
tamanho_da_janela_seg_list = [0.8, 1.0]


# Experiment loop for each time window
for tamanho_da_janela_seg in tamanho_da_janela_seg_list:
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))
    print(f"\n{'='*100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'='*100}")

    exp_dir = Path(
        f"CCA_trainable_eegnet_full_dataset/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
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

        X_train = np.zeros(
            (tamanho_da_janela * num_trials_train, len(occipital_electrodes), len(indices))
        )
        X_test = np.zeros(
            (tamanho_da_janela * num_trials_test, len(occipital_electrodes), len(indices))
        )

        for k in range(len(indices)):
            # Extract training data for this frequency
            eeg_matrix_train = train_data[
                occipital_electrodes, :tamanho_da_janela, indices[k], :
            ]
            eeg_matrix_test = test_data[
                occipital_electrodes, :tamanho_da_janela, indices[k], :
            ]
            # Transpose so each row represents a sample
            eeg_matrix_train = np.transpose(eeg_matrix_train)
            eeg_matrix_test = np.transpose(eeg_matrix_test)
            eeg_matrix_train = np.concatenate(eeg_matrix_train, axis=0)
            eeg_matrix_test = np.concatenate(eeg_matrix_test, axis=0)

            X_train[:, :, k] = eeg_matrix_train
            X_test[:, :, k] = eeg_matrix_test

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
        X_teste_janelas = []
        X_treino_janelas = []
        Y_teste_janelas = []
        Y_treino_janelas = []

        for k in range(len(indices)):
            X_t, numero_janelas_teste = get_windows(
                X_test[:, :, k], tamanho_da_janela, include_last=False
            )
            Y_t, _ = get_windows(Y_test[:, :, k], tamanho_da_janela, include_last=False)

            X_v, numero_janelas_treino = get_windows(
                X_train[:, :, k], tamanho_da_janela, include_last=False
            )
            Y_v, _ = get_windows(Y_train[:, :, k], tamanho_da_janela, include_last=False)

            X_teste_janelas.append(X_t)
            Y_teste_janelas.append(Y_t)

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
                janela_x = X_treino_janelas[m][j]
                rotulos_treinamento.append(frequencias[indices[m]])
                cont_1 = 0
                for w in range(len(indices)):
                    Wx = Combinadores_X[:, w]
                    projecao_x = np.dot(Wx, janela_x.T)
                    tensor_treinamento[cont, cont_1, :] = projecao_x
                    cont_1 += 1
                cont += 1

        # Build test tensor with CCA projections
        rotulos_teste = []
        tensor_teste = np.zeros(
            [len(indices) * numero_janelas_teste, len(indices), tamanho_da_janela]
        )
        cont = 0

        for m in range(len(indices)):
            for j in range(numero_janelas_teste):
                janela_x = X_teste_janelas[m][j]
                rotulos_teste.append(frequencias[indices[m]])
                cont_1 = 0

                for w in range(len(indices)):
                    Wx = Combinadores_X[:, w]
                    projecao_x = np.dot(Wx, janela_x.T)
                    tensor_teste[cont, cont_1, :] = projecao_x
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
            TensorDataset(X_teste, Y_teste), batch_size=10, shuffle=False
        )


        # Evaluate
        model.load_state_dict(torch.load(exp_dir.joinpath(f"best_model_user_{test_user}.pth")))
        accuracy, recall, f1, cm = evaluate(model, test_loader)
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
