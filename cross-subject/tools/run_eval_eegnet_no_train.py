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
    evaluate,
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
tamanho_da_janela_seg_list = [0.4]


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
    for test_user_idx, test_user in enumerate(users):
        # if test_user != 2 and tamanho_da_janela_seg == 0.4:
        #     continue
        print(f"\nProcessing User {test_user}")
        train_users = [u for u in users if u != test_user]
        print(f"Train Users: {train_users}")

        test_data = all_data[test_user_idx]
        num_trials_test = test_data.shape[-1]

        Y_test = np.zeros(
            (tamanho_da_janela, num_harmonica * 2, len(indices))
        )
        for k in indices:
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
        rotulos_teste = []
        tensor_teste = np.zeros(
            [len(indices) * num_trials_test, len(indices), tamanho_da_janela]
        )

        for k in range(len(indices)):
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
        X_teste = torch.tensor(tensor_teste, dtype=torch.float32).to(device)
        Y_teste = torch.tensor(rotulos_teste, dtype=torch.long).to(device)
        print(f"X_test: {X_teste.shape}")
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
