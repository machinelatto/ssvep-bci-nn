"""Shared preprocessing and tensor-building helpers for SSVEP datasets."""

from __future__ import annotations

import numpy as np
import scipy

from cca import CCA, reference_matrix


def bandpass_filter(
    dados,
    taxa_amostragem,
    freq_corte_low,
    freq_corte_high,
    ordem_filtro,
):
    """Filter EEG data with a Butterworth bandpass in (ch, time, freq, trial) layout."""
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


def filter_signals_subbands(eeg_signals, subban_no, sampling_rate):
    """Filter signals into subbands; input shape must be (samples, channels, time)."""
    samples, total_channels, sample_length = eeg_signals.shape
    all_data = np.zeros(
        (samples, subban_no, total_channels, sample_length),
        dtype=eeg_signals.dtype,
    )

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
            processed_signal = np.zeros(
                (total_channels, sample_length),
                dtype=eeg_signals.dtype,
            )
            b, a = bp_filters[sub_band]

            for ch_idx in range(total_channels):
                processed_signal[ch_idx] = scipy.signal.filtfilt(b, a, tmp_raw[ch_idx])

            all_data[sample, sub_band, :, :] = processed_signal

    return all_data


def split_trials_into_windows(
    data,
    window_size,
    mode="single",
    window_overlap=None,
):
    """Split (ch, time, freqs, trials) into windows.

    Args:
        data: Input array with shape (channels, time, num_freqs, num_trials).
        window_size: Number of time samples per window.
        mode: "single" or "multiple".
            - "single": one window from the beginning of each trial.
            - "multiple": sliding windows across each trial.
        window_overlap: Overlap in samples for "multiple" mode.

    Returns:
        segmented: Array with shape (channels, window_size, num_freqs, num_trials * num_windows).
    """
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError(
            f"Expected input shape (channels, time, freqs, trials), got {data.shape}."
        )

    channels, total_time, num_freqs, num_trials = data.shape
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if window_size > total_time:
        raise ValueError(
            f"window_size ({window_size}) cannot be greater than signal length ({total_time})."
        )

    mode = str(mode).lower()
    if mode not in {"single", "multiple"}:
        raise ValueError("mode must be either 'single' or 'multiple'.")

    if mode == "single":
        starts = [0]
    else:
        if window_overlap is not None:
            overlap = int(window_overlap)
            if overlap < 0 or overlap >= window_size:
                raise ValueError("window_overlap must be in [0, window_size).")
            step = window_size - overlap
        else:
            step = window_size

        if step <= 0:
            raise ValueError("Computed step must be > 0.")

        starts = list(range(0, total_time - window_size + 1, step))
        if not starts:
            starts = [0]

    num_windows = len(starts)
    segmented = np.zeros(
        (channels, window_size, num_freqs, num_trials * num_windows),
        dtype=data.dtype,
    )

    out_trial = 0
    for trial_idx in range(num_trials):
        for start in starts:
            end = start + window_size
            segmented[:, :, :, out_trial] = data[:, start:end, :, trial_idx]
            out_trial += 1

    return segmented


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

    x_train_cca = np.zeros(
        (len(occipital_electrodes), tamanho_da_janela * num_trials_train, len(indices))
    )

    x_train_windows = np.zeros(
        (num_trials_train * len(indices), len(occipital_electrodes), tamanho_da_janela),
        dtype=np.float32,
    )
    x_test_windows = np.zeros(
        (num_trials_test * len(indices), len(occipital_electrodes), tamanho_da_janela),
        dtype=np.float32,
    )

    labels_train = []
    labels_test = []

    for k in range(len(indices)):
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
        y_ref = reference_matrix(
            num_harmonica,
            inform_fase,
            num_trials_train,
            frequencias[indices[k]],
            fases,
            tamanho_da_janela,
        )
        wx, _, _ = CCA(x_train_cca[:, :, k], y_ref)
        combinadores_x.append(wx)
    combinadores_x = np.column_stack(combinadores_x)

    tensor_treinamento = np.zeros(
        [len(indices) * num_trials_train, len(indices), tamanho_da_janela],
        dtype=np.float32,
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
        [len(indices) * num_trials_test, len(indices), tamanho_da_janela],
        dtype=np.float32,
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


def build_tensors_with_cca_joint(
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
    """Build CCA-projected train/test tensors and labels using concatenated CCA."""
    num_trials_train = train_data.shape[-1]
    num_trials_test = test_data.shape[-1]

    total_time_train = tamanho_da_janela * num_trials_train * len(indices)

    x_train_cca = np.zeros((len(occipital_electrodes), total_time_train))

    x_train_windows = np.zeros(
        (num_trials_train * len(indices), len(occipital_electrodes), tamanho_da_janela),
        dtype=np.float32,
    )
    x_test_windows = np.zeros(
        (num_trials_test * len(indices), len(occipital_electrodes), tamanho_da_janela),
        dtype=np.float32,
    )

    labels_train = []
    labels_test = []

    for k in range(len(indices)):
        start_idx = k * tamanho_da_janela * num_trials_train
        end_idx = (k + 1) * tamanho_da_janela * num_trials_train

        eeg_matrix_train_windows = train_data[
            occipital_electrodes, :tamanho_da_janela, indices[k], :
        ]
        eeg_matrix_test_windows = test_data[
            occipital_electrodes, :tamanho_da_janela, indices[k], :
        ]

        eeg_matrix_train = eeg_matrix_train_windows.transpose(0, 2, 1).reshape(
            len(occipital_electrodes), -1
        )
        x_train_cca[:, start_idx:end_idx] = eeg_matrix_train

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
        y_ref = reference_matrix(
            num_harmonica,
            inform_fase,
            num_trials_train * len(indices),
            frequencias[indices[k]],
            fases,
            tamanho_da_janela,
        )
        wx, _, _ = CCA(x_train_cca, y_ref)
        combinadores_x.append(wx)

    combinadores_x = np.column_stack(combinadores_x)

    tensor_treinamento = np.zeros(
        [len(indices) * num_trials_train, len(indices), tamanho_da_janela],
        dtype=np.float32,
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
        [len(indices) * num_trials_test, len(indices), tamanho_da_janela],
        dtype=np.float32,
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

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)

    if apply_subband_filter:
        x_train = filter_signals_subbands(
            x_train, subban_no=subban_no, sampling_rate=sampling_rate
        )
        x_test = filter_signals_subbands(
            x_test, subban_no=subban_no, sampling_rate=sampling_rate
        )

    return x_train, x_test, labels_train, labels_test, len(occipital_electrodes)


def build_tensors_with_fbcca(
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
    subban_no=3,
    sampling_rate=250,
):
    """Build CCA-projected train/test tensors and labels using Filter Bank CCA."""
    num_trials_train = train_data.shape[-1]
    num_trials_test = test_data.shape[-1]

    labels_train = []
    labels_test = []

    x_train_windows_raw = []
    x_test_windows_raw = []

    for k in range(len(indices)):
        eeg_matrix_train_windows = train_data[
            occipital_electrodes, :tamanho_da_janela, indices[k], :
        ]
        eeg_matrix_test_windows = test_data[
            occipital_electrodes, :tamanho_da_janela, indices[k], :
        ]

        x_train_windows_raw.append(eeg_matrix_train_windows.transpose(2, 0, 1))
        x_test_windows_raw.append(eeg_matrix_test_windows.transpose(2, 0, 1))

        labels_train.extend([frequencias[indices[k]]] * num_trials_train)
        labels_test.extend([frequencias[indices[k]]] * num_trials_test)

    x_train_windows_raw = np.concatenate(x_train_windows_raw, axis=0)
    x_test_windows_raw = np.concatenate(x_test_windows_raw, axis=0)

    x_train_windows_sub = filter_signals_subbands(
        x_train_windows_raw, subban_no=subban_no, sampling_rate=sampling_rate
    )
    x_test_windows_sub = filter_signals_subbands(
        x_test_windows_raw, subban_no=subban_no, sampling_rate=sampling_rate
    )

    combinadores_x = []
    for sub in range(subban_no):
        comb_x_sub = []
        for k in range(len(indices)):
            start_idx = k * num_trials_train
            end_idx = (k + 1) * num_trials_train
            train_k_sub = x_train_windows_sub[start_idx:end_idx, sub, :, :]
            x_train_cca_sub = np.transpose(train_k_sub, (1, 0, 2)).reshape(
                len(occipital_electrodes), -1
            )
            y_ref = reference_matrix(
                num_harmonica,
                inform_fase,
                num_trials_train,
                frequencias[indices[k]],
                fases,
                tamanho_da_janela,
            )
            wx, _, _ = CCA(x_train_cca_sub, y_ref)
            comb_x_sub.append(wx)
        combinadores_x.append(np.column_stack(comb_x_sub))
    combinadores_x = np.array(combinadores_x)

    tensor_treinamento = np.zeros(
        [len(indices) * num_trials_train, subban_no, len(indices), tamanho_da_janela],
        dtype=np.float32,
    )
    for idx_trial in range(len(indices) * num_trials_train):
        janela_x = x_train_windows_sub[idx_trial]
        if mean_center:
            janela_x = janela_x - np.mean(janela_x, axis=2, keepdims=True)
        for freq_idx in range(len(indices)):
            for sub in range(subban_no):
                wx = combinadores_x[sub, :, freq_idx]
                tensor_treinamento[idx_trial, sub, freq_idx, :] = np.dot(wx, janela_x[sub])

    tensor_teste = np.zeros(
        [len(indices) * num_trials_test, subban_no, len(indices), tamanho_da_janela],
        dtype=np.float32,
    )
    for idx_trial in range(len(indices) * num_trials_test):
        janela_x = x_test_windows_sub[idx_trial]
        if mean_center:
            janela_x = janela_x - np.mean(janela_x, axis=2, keepdims=True)
        for freq_idx in range(len(indices)):
            for sub in range(subban_no):
                wx = combinadores_x[sub, :, freq_idx]
                tensor_teste[idx_trial, sub, freq_idx, :] = np.dot(wx, janela_x[sub])

    return tensor_treinamento, tensor_teste, labels_train, labels_test, len(indices)


def build_tensors_with_fbcca_joint(
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
    subban_no=3,
    sampling_rate=250,
):
    """Build CCA-projected train/test tensors and labels using joint Filter Bank CCA."""
    num_trials_train = train_data.shape[-1]
    num_trials_test = test_data.shape[-1]

    total_time_train = tamanho_da_janela * num_trials_train * len(indices)

    x_train_cca = np.zeros((subban_no, len(occipital_electrodes), total_time_train))

    labels_train = []
    labels_test = []

    x_train_windows_raw = []
    x_test_windows_raw = []

    for k in range(len(indices)):
        eeg_matrix_train_windows = train_data[
            occipital_electrodes, :tamanho_da_janela, indices[k], :
        ]
        eeg_matrix_test_windows = test_data[
            occipital_electrodes, :tamanho_da_janela, indices[k], :
        ]

        x_train_windows_raw.append(eeg_matrix_train_windows.transpose(2, 0, 1))
        x_test_windows_raw.append(eeg_matrix_test_windows.transpose(2, 0, 1))

        labels_train.extend([frequencias[indices[k]]] * num_trials_train)
        labels_test.extend([frequencias[indices[k]]] * num_trials_test)

    x_train_windows_raw = np.concatenate(x_train_windows_raw, axis=0)
    x_test_windows_raw = np.concatenate(x_test_windows_raw, axis=0)

    x_train_windows_sub = filter_signals_subbands(
        x_train_windows_raw, subban_no=subban_no, sampling_rate=sampling_rate
    )
    x_test_windows_sub = filter_signals_subbands(
        x_test_windows_raw, subban_no=subban_no, sampling_rate=sampling_rate
    )

    for sub in range(subban_no):
        x_train_cca[sub] = np.transpose(
            x_train_windows_sub[:, sub, :, :], (1, 0, 2)
        ).reshape(len(occipital_electrodes), -1)

    combinadores_x = []
    for sub in range(subban_no):
        comb_x_sub = []
        for k in range(len(indices)):
            y_ref = reference_matrix(
                num_harmonica,
                inform_fase,
                num_trials_train * len(indices),
                frequencias[indices[k]],
                fases,
                tamanho_da_janela,
            )
            wx, _, _ = CCA(x_train_cca[sub], y_ref)
            comb_x_sub.append(wx)
        combinadores_x.append(np.column_stack(comb_x_sub))
    combinadores_x = np.array(combinadores_x)

    tensor_treinamento = np.zeros(
        [len(indices) * num_trials_train, subban_no, len(indices), tamanho_da_janela],
        dtype=np.float32,
    )
    for idx_trial in range(len(indices) * num_trials_train):
        janela_x = x_train_windows_sub[idx_trial]
        if mean_center:
            janela_x = janela_x - np.mean(janela_x, axis=2, keepdims=True)
        for freq_idx in range(len(indices)):
            for sub in range(subban_no):
                wx = combinadores_x[sub, :, freq_idx]
                tensor_treinamento[idx_trial, sub, freq_idx, :] = np.dot(wx, janela_x[sub])

    tensor_teste = np.zeros(
        [len(indices) * num_trials_test, subban_no, len(indices), tamanho_da_janela],
        dtype=np.float32,
    )
    for idx_trial in range(len(indices) * num_trials_test):
        janela_x = x_test_windows_sub[idx_trial]
        if mean_center:
            janela_x = janela_x - np.mean(janela_x, axis=2, keepdims=True)
        for freq_idx in range(len(indices)):
            for sub in range(subban_no):
                wx = combinadores_x[sub, :, freq_idx]
                tensor_teste[idx_trial, sub, freq_idx, :] = np.dot(wx, janela_x[sub])

    return tensor_treinamento, tensor_teste, labels_train, labels_test, len(indices)


__all__ = [
    "bandpass_filter",
    "build_tensors_no_cca",
    "build_tensors_with_cca",
    "build_tensors_with_cca_joint",
    "build_tensors_with_fbcca",
    "build_tensors_with_fbcca_joint",
    "filter_signals_subbands",
    "split_trials_into_windows",
]
