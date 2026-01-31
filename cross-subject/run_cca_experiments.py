import pandas as pd
from pathlib import Path
import numpy as np
import scipy


# # Utilities
from cross_subject_utils import (
    evaluate,
    load_data_from_users,
)

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    recall_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
from cca import CCA_otimizacao, matriz_referencia

def evaluate(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Test set Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # classes = np.unique(np.concatenate((all_labels, all_preds)))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    # plt.show()
    return accuracy, recall, f1, cm


# # Cross Subject
freq_phase_path = (
    "/home/mateuschinelatto/Experiments/data/benchmark/Freq_Phase.mat"
)
freq_phase = scipy.io.loadmat(freq_phase_path)
frequencias = np.round(freq_phase["freqs"], 2).ravel()
fases = freq_phase["phases"]

# Parâmetros do pré-processamento
sample_rate = 250
filter_order = 10
freq_cut_high = 70
freq_cut_low = 6
delay = 160

# Parâmetros do CCA
num_harmonica = 5
inform_fase = 0

# Usuários
# users = list(range(1, 11))  # Usuários de 1 a 10
users = list(range(1, 36))  # Usuários de 1 a 35
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
frequencias_desejadas = frequencias[:] # Todas as frequências
indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

print("Usuários de interesse:", users)
print(f"Frequencies used: {frequencias_desejadas}")
print(f"Frequencies indices: {indices}")

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

# Parâmetros de janelas e sessões
tamanho_da_janela_seg = [0.4, 0.6, 0.8, 1.0]  # em segundos

for tamanho in tamanho_da_janela_seg:
    tamanho_da_janela = int(np.ceil(tamanho * sample_rate))
    print(f"Tamanho da janela: {tamanho_da_janela} samples ({tamanho} s)")

    exp_dir = Path(
        f"CCA_full_dataset/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho}_s/"
    )

    # Cross-Subject EEGNet Training (single window per trial, no window separation)
    metricas_usuarios = []
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare cross-subject splits
    for test_user_idx, test_user in enumerate(users):
        print(f"Processando Usuário {test_user}")

        test_data = all_data[test_user_idx]
        num_trials_test = test_data.shape[-1]

        Y_test = np.zeros((tamanho_da_janela, num_harmonica * 2, len(indices)))
        for k in indices:
            y_test = matriz_referencia(
                num_harmonica,
                inform_fase,
                1,
                frequencias[k],
                fases,
                tamanho_da_janela,
            )
            Y_test[:, :, k] = y_test

        labels = []
        predictions = []
        for k in range(len(indices)):
            for session in range(num_trials_test):
                # For training: each trial is a single window
                eeg_matrix_test = test_data[
                    occipital_electrodes, :tamanho_da_janela, indices[k], session
                ]

                # Transpõe os dados para que cada linha represente uma amostra
                eeg_matrix_test = np.transpose(eeg_matrix_test)
                labels.append(k)
                corrs = np.zeros(len(indices))
                for freq in range(len(indices)):
                    Wx, Wy, corr = CCA_otimizacao(eeg_matrix_test, Y_test[:, :, freq])
                    corrs[freq] = corr
                predicted_label = np.argmax(corrs)
                predictions.append(predicted_label)

        accuracy, recall, f1, cm = evaluate(labels, predictions)
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
            f"Test User {test_user} Finished: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        # Salvar as métricas de cada usuário
        df_metricas = pd.DataFrame(metricas_usuarios)
        df_metricas.to_csv(exp_dir.joinpath("metricas.csv"), index=False)

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho} s.")
    print("=" * 100)