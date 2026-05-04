import pandas as pd
from pathlib import Path
import numpy as np
import scipy

from cross_subject_utils import (
    evaluate,
    load_data_from_users,
)

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
from cca import CCA, reference_matrix

def evaluate(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="weighted")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
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
freq_cut_high = 50
freq_cut_low = 6
delay = 160

# Parâmetros do CCA
num_harmonica = 3
inform_fase = 0

# Usuários
users = list(range(1, 36))  # Usuários de 1 a 35
users_to_run = users.copy()  # Ex.: [1, 5, 10]
occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
frequencias_desejadas = frequencias[:] # Todas as frequências
indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

# Optional CAR configuration on loaded data
apply_car = True
car_reference_channels = occipital_electrodes
car_target_channels = occipital_electrodes

print("Usuários de interesse:", users)
print("Usuários para executar:", users_to_run)
print(f"Frequencies used: {frequencias_desejadas}")
print(f"Frequencies indices: {indices}")

all_data = load_data_from_users(
    dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
    users=users,
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

# Parâmetros de janelas e sessões
tamanho_da_janela_seg = [1.0]  # em segundos

for tamanho in tamanho_da_janela_seg:
    tamanho_da_janela = int(np.ceil(tamanho * sample_rate))
    print(f"Tamanho da janela: {tamanho_da_janela} samples ({tamanho} s)")

    exp_dir = Path(
        f"35_40_optimized/CCA_CAR/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho}_s/"
    )

    # Cross-Subject EEGNet Training (single window per trial, no window separation)
    metricas_usuarios = []
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare cross-subject splits
    for test_user in users_to_run:
        print(f"Processando Usuário {test_user}")

        test_data = all_data[users.index(test_user)]
        num_trials_test = test_data.shape[-1]

        Y_test = np.zeros((num_harmonica * 2, tamanho_da_janela, len(indices)))
        for k in range(len(indices)):
            y_test = reference_matrix(
                num_harmonica,
                inform_fase,
                1,
                frequencias[indices[k]],
                fases,
                tamanho_da_janela,
            )
            Y_test[:, :, k] = y_test

        labels = []
        predictions = []
        for k in range(len(indices)):
            for session in range(num_trials_test):
                # Extract EEG for this trial: shape (num_channels, num_timepoints)
                eeg_matrix_test = test_data[
                    occipital_electrodes, :tamanho_da_janela, indices[k], session
                ]
                # NO TRANSPOSE - keep standard BCI format: (num_channels, num_timepoints)
                labels.append(indices[k])
                corrs = np.zeros(len(indices))
                for freq in range(len(indices)):
                    # Y_test[:, :, freq] has shape (num_harmonics*2, num_timepoints)
                    Wx, Wy, corr = CCA(eeg_matrix_test, Y_test[:, :, freq])
                    corrs[freq] = corr
                predicted_label = np.argmax(corrs)
                predictions.append(indices[predicted_label])

        accuracy, recall, f1, cm = evaluate(labels, predictions)
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
            f"Test User {test_user} Finished: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        # Salvar as métricas de cada usuário (append para permitir retomada)
        metrics_path = exp_dir.joinpath("metricas.csv")
        pd.DataFrame([metricas_usuarios[-1]]).to_csv(
            metrics_path,
            mode="a",
            header=not metrics_path.exists(),
            index=False,
        )

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho} s.")
    print("=" * 100)