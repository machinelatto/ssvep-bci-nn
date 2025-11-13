import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    recall_score,
    accuracy_score,
)
import torch
from tqdm import tqdm
import tqdm.notebook


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot learning curves for training and validation loss and accuracy.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_accuracies (list): List of training accuracies.
        val_accuracies (list): List of validation accuracies.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves: Training and Validation")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curves: Training and Validation")
    plt.tight_layout()
    plt.show()


def filter_signals_subbands(eeg_signals, subban_no, sampling_rate):
    """Filter EEG signals and divide into sub-bands. High cutoff is fixed at 90 Hz. Low cutoff starts at 8 Hz and increases in steps of 8 Hz.

    Args:
        eeg_signals (np.ndarray): Input EEG signals of shape (samples, channels, time).
        subban_no (int): Number of sub-bands to create.
        sampling_rate (int): Sampling rate of the EEG signals.

    Returns:
        np.ndarray: Filtered EEG signals divided into sub-bands. Has shape (samples, subbands, channels, time).
    """
    samples, total_channels, sample_length = eeg_signals.shape

    all_data = np.zeros((samples, subban_no, total_channels, sample_length))
    print(f"All data shape before filtering: {eeg_signals.shape}")

    # Bandpass filters
    high_cutoff = [90] * subban_no

    samples, total_channels, sample_length = eeg_signals.shape

    all_data = np.zeros((samples, subban_no, total_channels, sample_length))

    # Bandpass filters
    high_cutoff = [90] * subban_no
    low_cutoff = [i for i in range(8, 8 * (subban_no + 1), 8)]
    filter_order = 2
    passband_ripple = 1
    bp_filters = []

    for i in range(subban_no):
        b, a = scipy.signal.cheby1(
            filter_order,
            passband_ripple,
            [low_cutoff[i], high_cutoff[i]],
            btype="band",
            fs=sampling_rate,
        )
        bp_filters.append((b, a))

    # Filtering
    for sample in range(samples):
        tmp_raw = eeg_signals[sample]
        for sub_band in range(subban_no):
            processed_signal = np.zeros((total_channels, sample_length))
            b, a = bp_filters[sub_band]

            for ch_idx in range(total_channels):
                processed_signal[ch_idx] = scipy.signal.filtfilt(b, a, tmp_raw[ch_idx])

            all_data[sample, sub_band, :, :] = processed_signal
    print(f"All data shape after filtering: {all_data.shape}")
    return all_data


def get_windows(eeg_matrix, window_size, include_last=False):
    """Extract sliding windows from the EEG matrix.

    Args:
        eeg_matrix (np.ndarray): Input EEG matrix of shape (time, channels).
        window_size (int): Size of each window.
        include_last (bool, optional): Whether to include the last window if it's smaller than window_size. Defaults to False.

    Returns:
        list: List of extracted windows.
    """
    total_samples = eeg_matrix.shape[0]
    num_windows = total_samples // window_size
    windows = []
    for i in range(0, total_samples, window_size):
        window = eeg_matrix[i : i + window_size]
        windows.append(window)
    if not include_last and total_samples % window_size != 0:
        windows.pop()
    return windows, num_windows


def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Test set Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    classes = np.unique(np.concatenate((all_labels, all_preds)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    plt.show()
    return accuracy, recall, f1, cm


def bandpass_filter(
    dados, taxa_amostragem, freq_corte_low, freq_corte_high, ordem_filtro
):
    """
    Filtra dados EEG utilizando um filtro Butterworth passa-banda.
    Parâmetros:
    dados (ndarray): Dados do EEG com formato (número de eletrodos, número de amostras, número de frequências, número de trials).
    taxa_amostragem (int): Frequência de amostragem dos sinais EEG (Hz).
    freq_corte_low (float): Frequência de corte inferior do filtro passa-banda (Hz).
    freq_corte_high (float): Frequência de corte superior do filtro passa-banda (Hz).
    ordem_filtro (int): Ordem do filtro Butterworth.

    Retorna:
    ndarray: Dados EEG filtrados.
    """

    # **Construção do filtro passa-banda**
    # Cria o filtro passa-banda com os parâmetros especificados
    b, a = scipy.signal.butter(
        ordem_filtro,
        [freq_corte_low, freq_corte_high],
        btype="bandpass",
        analog=False,
        output="ba",
        fs=taxa_amostragem,
    )

    # **Filtragem dos dados**
    # Realiza o processo de filtragem para todas as frequências, trials e eletrodos
    num_eletrodos, num_amostras, num_freqs, num_trials = dados.shape

    filtered_data = np.zeros_like(dados)
    # Filtra os dados para cada frequência, trial e eletrodo
    for f in range(num_freqs):  # Para cada frequência de estimulação
        for trial in range(num_trials):  # Para cada trial
            for eletrodo in range(num_eletrodos):  # Para cada eletrodo
                # Filtra o sinal com o filtro de fase zero
                eletrodo_filtrado = scipy.signal.filtfilt(
                    b, a, dados[eletrodo, :, f, trial]
                )
                # Substitui o dado original pelo filtrado
                filtered_data[eletrodo, :, f, trial] = eletrodo_filtrado

    return filtered_data


def load_data_from_users(
    users,
    visual_delay=160,
    dataset_path="C:/Users/machi/Documents/Mestrado/repos/data/benchmark/",
    filter_bandpass=False,
    sample_rate=250,
    freq_cut_low=6,
    freq_cut_high=70,
    filter_order=10,
):
    all_data = []
    for user in tqdm.notebook.tqdm(users, desc="Carregando dados dos usuários"):
        file_path = f"{dataset_path}/S{user}.mat"
        data = scipy.io.loadmat(file_path)["data"]
        if filter_bandpass:
            data = bandpass_filter(
                data, sample_rate, freq_cut_low, freq_cut_high, filter_order
            )
        data = data[:, (visual_delay) : (visual_delay + 1250), :, :]
        all_data.append(data)
    return all_data
