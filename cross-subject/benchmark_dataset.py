"""
Shared data loading and tensor-building utilities for benchmark SSVEP dataset.
"""

import numpy as np
import scipy
import scipy.io
from tqdm import tqdm

from cca import CCA, reference_matrix


def bandpass_filter(
    dados, taxa_amostragem, freq_corte_low, freq_corte_high, ordem_filtro
):
    """Filter EEG data with Butterworth bandpass in benchmark data layout."""
    b, a = scipy.signal.butter(
        ordem_filtro,
        [freq_corte_low, freq_corte_high],
        btype="bandpass",
        analog=False,
        output="ba",
        fs=taxa_amostragem,
    )

    num_eletrodos, _, num_freqs, num_trials = dados.shape
    filtered_data = np.zeros_like(dados)

    for f in range(num_freqs):
        for trial in range(num_trials):
            for eletrodo in range(num_eletrodos):
                eletrodo_filtrado = scipy.signal.filtfilt(
                    b, a, dados[eletrodo, :, f, trial]
                )
                filtered_data[eletrodo, :, f, trial] = eletrodo_filtrado

    return filtered_data


def load_data_from_users(
    users,
    visual_delay=160,
    dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
    filter_bandpass=False,
    sample_rate=250,
    freq_cut_low=6,
    freq_cut_high=70,
    filter_order=10,
):
    """Load benchmark users from .mat files and slice post-stimulus interval."""
    all_data = []
    for user in tqdm(users, desc="Carregando dados dos usuarios"):
        file_path = f"{dataset_path}/S{user}.mat"
        data = scipy.io.loadmat(file_path)["data"]
        if filter_bandpass:
            data = bandpass_filter(
                data, sample_rate, freq_cut_low, freq_cut_high, filter_order
            )
        data = data[:, visual_delay : (visual_delay + 1250), :, :]
        all_data.append(data)
    return all_data


def filter_signals_subbands(eeg_signals, subban_no, sampling_rate):
    """Filter signals into subbands; input shape must be (samples, channels, time)."""
    samples, total_channels, sample_length = eeg_signals.shape
    all_data = np.zeros((samples, subban_no, total_channels, sample_length))

    high_cutoff = [50] * subban_no
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

    for sample in range(samples):
        tmp_raw = eeg_signals[sample]
        for sub_band in range(subban_no):
            processed_signal = np.zeros((total_channels, sample_length))
            b, a = bp_filters[sub_band]

            for ch_idx in range(total_channels):
                processed_signal[ch_idx] = scipy.signal.filtfilt(b, a, tmp_raw[ch_idx])

            all_data[sample, sub_band, :, :] = processed_signal

    return all_data


def load_freq_phase(
    freq_phase_path="/home/mateuschinelatto/Experiments/data/benchmark/Freq_Phase.mat",
):
    """Load frequencies and phases from benchmark metadata file."""
    freq_phase = scipy.io.loadmat(freq_phase_path)
    frequencias = np.round(freq_phase["freqs"], 2).ravel()
    fases = freq_phase["phases"]
    return frequencias, fases


def build_tensors_with_cca(
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
    apply_subband_filter=True,
    subban_no=3,
    sampling_rate=250,
):
    """Build CCA-projected train/test tensors and labels."""
    num_trials_train = train_data.shape[-1]
    num_trials_test = test_data.shape[-1]

    y_train = np.zeros(
        (num_harmonica * 2, tamanho_da_janela * num_trials_train, len(indices))
    )
    x_train_cca = np.zeros(
        (len(occipital_electrodes), tamanho_da_janela * num_trials_train, len(indices))
    )

    x_train_windows = np.zeros(
        (num_trials_train * len(indices), len(occipital_electrodes), tamanho_da_janela)
    )
    x_test_windows = np.zeros(
        (num_trials_test * len(indices), len(occipital_electrodes), tamanho_da_janela)
    )

    labels_train = []
    labels_test = []

    for k in range(len(indices)):
        y_train[:, :, k] = reference_matrix(
            num_harmonica,
            inform_fase,
            num_trials_train,
            frequencias[indices[k]],
            fases,
            tamanho_da_janela,
        )

        eeg_matrix_train_windows = train_data[
            occipital_electrodes, :tamanho_da_janela, indices[k], :
        ]
        eeg_matrix_test_windows = test_data[
            occipital_electrodes, :tamanho_da_janela, indices[k], :
        ]

        eeg_matrix_train = eeg_matrix_train_windows.transpose(0, 2, 1).reshape(
            len(occipital_electrodes), -1
        )
        x_train_cca[:, :, k] = eeg_matrix_train

        x_train_windows[k * num_trials_train : (k + 1) * num_trials_train] = (
            eeg_matrix_train_windows.transpose(2, 0, 1)
        )
        x_test_windows[k * num_trials_test : (k + 1) * num_trials_test] = (
            eeg_matrix_test_windows.transpose(2, 0, 1)
        )

        labels_train.extend([frequencias[indices[k]]] * num_trials_train)
        labels_test.extend([frequencias[indices[k]]] * num_trials_test)

    combinadores_x = []
    for k in range(len(indices)):
        wx, _, _ = CCA(x_train_cca[:, :, k], y_train[:, :, k])
        combinadores_x.append(wx)
    combinadores_x = np.column_stack(combinadores_x)

    tensor_treinamento = np.zeros(
        [len(indices) * num_trials_train, len(indices), tamanho_da_janela]
    )
    for j in range(num_trials_train):
        for k in range(len(indices)):
            janela_x = x_train_windows[k * num_trials_train + j]
            if mean_center:
                janela_x = janela_x - np.mean(janela_x, axis=1, keepdims=True)
            for freq_idx in range(len(indices)):
                wx = combinadores_x[:, freq_idx]
                projecao_x = np.dot(wx, janela_x)
                tensor_treinamento[k * num_trials_train + j, freq_idx, :] = projecao_x

    tensor_teste = np.zeros(
        [len(indices) * num_trials_test, len(indices), tamanho_da_janela]
    )
    for j in range(num_trials_test):
        for k in range(len(indices)):
            janela_x = x_test_windows[k * num_trials_test + j]
            if mean_center:
                janela_x = janela_x - np.mean(janela_x, axis=1, keepdims=True)
            for freq_idx in range(len(indices)):
                wx = combinadores_x[:, freq_idx]
                projecao_x = np.dot(wx, janela_x)
                tensor_teste[k * num_trials_test + j, freq_idx, :] = projecao_x

    if apply_subband_filter:
        tensor_treinamento = filter_signals_subbands(
            tensor_treinamento, subban_no=subban_no, sampling_rate=sampling_rate
        )
        tensor_teste = filter_signals_subbands(
            tensor_teste, subban_no=subban_no, sampling_rate=sampling_rate
        )

    return tensor_treinamento, tensor_teste, labels_train, labels_test, len(indices)


def build_tensors_no_cca(
    train_data,
    test_data,
    occipital_electrodes,
    frequencias,
    indices,
    tamanho_da_janela,
    apply_subband_filter=True,
    subban_no=3,
    sampling_rate=250,
):
    """Build non-CCA train/test tensors and labels."""
    x_train = []
    x_test = []
    labels_train = []
    labels_test = []

    for sessao in range(train_data.shape[3]):
        for freq_idx in range(len(indices)):
            eeg_trial = train_data[
                occipital_electrodes, :tamanho_da_janela, indices[freq_idx], sessao
            ]
            x_train.append(eeg_trial)
            labels_train.append(frequencias[indices[freq_idx]])

    for sessao in range(test_data.shape[3]):
        for freq_idx in range(len(indices)):
            eeg_trial = test_data[
                occipital_electrodes, :tamanho_da_janela, indices[freq_idx], sessao
            ]
            x_test.append(eeg_trial)
            labels_test.append(frequencias[indices[freq_idx]])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    if apply_subband_filter:
        x_train = filter_signals_subbands(
            x_train, subban_no=subban_no, sampling_rate=sampling_rate
        )
        x_test = filter_signals_subbands(
            x_test, subban_no=subban_no, sampling_rate=sampling_rate
        )

    return x_train, x_test, labels_train, labels_test, len(occipital_electrodes)