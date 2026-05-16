"""
Shared data loading and tensor-building utilities for DESTINE SSVEP dataset.

Expected file naming pattern:
    ssvep_<freq>_Hz_training_subject_<subject>_session_<session>.mat

Each .mat file is expected to contain a 2D EEG array (time, channels), usually in
the variable "storageDataAcquirement".
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import scipy
import scipy.io
from tqdm import tqdm

from ssvep_shared import (
    bandpass_filter,
    build_tensors_no_cca as _build_tensors_no_cca,
    build_tensors_with_cca as _build_tensors_with_cca,
    build_tensors_with_cca_joint as _build_tensors_with_cca_joint,
    build_tensors_with_fbcca as _build_tensors_with_fbcca,
    build_tensors_with_fbcca_joint as _build_tensors_with_fbcca_joint,
    filter_signals_subbands,
    split_trials_into_windows,
)


FILENAME_RE = re.compile(
    r"ssvep_(?P<freq>\d+(?:\.\d+)?)_Hz_training_subject_(?P<subject>.+?)_session_(?P<session>\d+)\.mat$",
    re.IGNORECASE,
)

DEFAULT_CHANNEL_NAMES = [
    "O1",
    "O2",
    "Oz",
    "POz",
    "Pz",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "P1",
    "P2",
    "Cz",
    "C1",
    "C2",
    "CPz",
    "FCz",
]

# Local CAR neighborhoods inspired by Teste14BCI_CAR_example.m.
# Each entry is: central channel -> neighborhood channels used in the average.
LOCAL_CAR_GROUPS = {
    "O1": ["O1", "PO7", "PO3", "POz", "Oz"],
    "O2": ["O2", "Oz", "POz", "PO4", "PO8"],
    "Oz": ["Oz", "O1", "PO3", "POz", "PO4", "O2"],
    "POz": ["POz", "O2", "Oz", "O1", "PO3", "P1", "Pz", "P2", "PO4"],
    "Pz": ["Pz", "P1", "CPz", "P2", "POz"],
    "PO3": ["PO3", "O1", "PO7", "P1", "POz", "Oz"],
    "PO4": ["PO4", "PO8", "O2", "Oz", "POz", "P2"],
    "PO7": ["PO7", "PO3", "O1"],
    "PO8": ["PO8", "PO4", "O2"],
    "P1": ["P1", "CPz", "Pz", "POz", "PO3"],
    "P2": ["P2", "CPz", "Pz", "POz", "PO4"],
    "Cz": ["Cz", "C1", "C2", "FCz", "CPz"],
    "C1": ["C1", "FCz", "Cz", "CPz"],
    "C2": ["C2", "FCz", "Cz", "CPz"],
    "CPz": ["CPz", "C1", "Cz", "C2", "P1", "Pz", "P2"],
    "FCz": ["FCz", "C1", "Cz", "C2"],
}


def parse_destine_filename(file_path: str | Path):
    """Parse DESTINE filename and return (frequency_hz, subject_id, session_id)."""
    name = Path(file_path).name
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid DESTINE filename format: {name}")

    freq_raw = float(match.group("freq"))
    # In DESTINE filenames, 7.5 Hz is encoded as "75".
    frequency = 7.5 if np.isclose(freq_raw, 75.0) else freq_raw
    subject = match.group("subject").strip().lower()
    session = int(match.group("session"))
    return frequency, subject, session


def discover_recordings(dataset_path):
    """Discover files and return nested mapping: subject -> freq -> session -> path."""
    dataset_path = Path(dataset_path)
    # DESTINE is organized in subject subfolders; recurse from dataset root.
    files = sorted(dataset_path.rglob("ssvep_*_Hz_training_subject_*_session_*.mat"))

    recordings = defaultdict(lambda: defaultdict(dict))
    for file_path in files:
        try:
            freq, subject, session = parse_destine_filename(file_path)
        except ValueError:
            continue
        recordings[subject][freq][session] = file_path

    return recordings


def load_mat_eeg(file_path, mat_key="storageDataAcquirement"):
    """Load one DESTINE .mat file and return EEG as (channels, time)."""
    mat = scipy.io.loadmat(file_path)

    if mat_key in mat:
        data = mat[mat_key]
    else:
        non_meta_keys = [k for k in mat.keys() if not k.startswith("__")]
        if len(non_meta_keys) != 1:
            raise KeyError(
                f"Could not find key '{mat_key}' in {file_path} and there is not a single fallback variable. "
                f"Available keys: {non_meta_keys}"
            )
        data = mat[non_meta_keys[0]]

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D EEG matrix in {file_path}, got shape {data.shape}")

    # Source files are stored as (time, channels). Convert to (channels, time).
    if data.shape[0] > data.shape[1]:
        data = data.T

    return data.astype(np.float32, copy=False)


def apply_local_car(eeg_matrix, channel_names=None, local_groups=None):
    """Apply local CAR using channel neighborhoods (input shape: channels x time)."""
    eeg_matrix = np.asarray(eeg_matrix)
    if eeg_matrix.ndim != 2:
        raise ValueError(f"Expected 2D EEG matrix, got shape {eeg_matrix.shape}")

    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES
    if local_groups is None:
        local_groups = LOCAL_CAR_GROUPS

    if eeg_matrix.shape[0] != len(channel_names):
        raise ValueError(
            f"Number of channels in data ({eeg_matrix.shape[0]}) does not match channel_names ({len(channel_names)})."
        )

    name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    out = eeg_matrix.copy()

    for center_name, neighbors in local_groups.items():
        if center_name not in name_to_idx:
            continue

        center_idx = name_to_idx[center_name]
        neighbor_indices = [name_to_idx[n] for n in neighbors if n in name_to_idx]
        if not neighbor_indices:
            continue

        out[center_idx, :] = eeg_matrix[center_idx, :] - np.mean(
            eeg_matrix[neighbor_indices, :], axis=0
        )

    return out


def apply_global_car(eeg_matrix, reference_channels=None, target_channels=None):
    """Apply standard CAR to EEG matrix (input shape: channels x time)."""
    eeg_matrix = np.asarray(eeg_matrix)
    num_channels = eeg_matrix.shape[0]

    if reference_channels is None:
        reference_channels = np.arange(num_channels)
    if target_channels is None:
        target_channels = np.arange(num_channels)

    reference_channels = np.asarray(reference_channels, dtype=int)
    target_channels = np.asarray(target_channels, dtype=int)

    ref = np.mean(eeg_matrix[reference_channels, :], axis=0)
    out = eeg_matrix.copy()
    out[target_channels, :] = eeg_matrix[target_channels, :] - ref[np.newaxis, :]
    return out


def build_tensors_no_cca(*args, **kwargs):
    """Wrapper with debug shape prints for non-CCA tensor building."""
    x_train, x_test, labels_train, labels_test, channels = _build_tensors_no_cca(
        *args, **kwargs
    )
    print(
        "[DESTINE] build_tensors_no_cca -> "
        f"x_train={x_train.shape}, x_test={x_test.shape}, "
        f"labels_train={len(labels_train)}, labels_test={len(labels_test)}, "
        f"channels={channels}"
    )
    return x_train, x_test, labels_train, labels_test, channels


def build_tensors_with_cca(*args, **kwargs):
    """Wrapper with debug shape prints for CCA tensor building."""
    x_train, x_test, labels_train, labels_test, channels = _build_tensors_with_cca(
        *args, **kwargs
    )
    print(
        "[DESTINE] build_tensors_with_cca -> "
        f"x_train={x_train.shape}, x_test={x_test.shape}, "
        f"labels_train={len(labels_train)}, labels_test={len(labels_test)}, "
        f"channels={channels}"
    )
    return x_train, x_test, labels_train, labels_test, channels


def build_tensors_with_cca_joint(*args, **kwargs):
    """Wrapper with debug shape prints for joint CCA tensor building."""
    x_train, x_test, labels_train, labels_test, channels = _build_tensors_with_cca_joint(
        *args, **kwargs
    )
    print(
        "[DESTINE] build_tensors_with_cca_joint -> "
        f"x_train={x_train.shape}, x_test={x_test.shape}, "
        f"labels_train={len(labels_train)}, labels_test={len(labels_test)}, "
        f"channels={channels}"
    )
    return x_train, x_test, labels_train, labels_test, channels


def build_tensors_with_fbcca(*args, **kwargs):
    """Wrapper with debug shape prints for FBCCA tensor building."""
    x_train, x_test, labels_train, labels_test, channels = _build_tensors_with_fbcca(
        *args, **kwargs
    )
    print(
        "[DESTINE] build_tensors_with_fbcca -> "
        f"x_train={x_train.shape}, x_test={x_test.shape}, "
        f"labels_train={len(labels_train)}, labels_test={len(labels_test)}, "
        f"channels={channels}"
    )
    return x_train, x_test, labels_train, labels_test, channels


def build_tensors_with_fbcca_joint(*args, **kwargs):
    """Wrapper with debug shape prints for joint FBCCA tensor building."""
    x_train, x_test, labels_train, labels_test, channels = _build_tensors_with_fbcca_joint(
        *args, **kwargs
    )
    print(
        "[DESTINE] build_tensors_with_fbcca_joint -> "
        f"x_train={x_train.shape}, x_test={x_test.shape}, "
        f"labels_train={len(labels_train)}, labels_test={len(labels_test)}, "
        f"channels={channels}"
    )
    return x_train, x_test, labels_train, labels_test, channels


def load_freq_phase(dataset_path, users=None):
    """Infer frequencies from available files and return (frequencies, phases)."""
    recordings = discover_recordings(dataset_path)

    if users is not None:
        users_set = {str(u).strip().lower() for u in users}
        freqs = set()
        for subject, by_freq in recordings.items():
            if subject in users_set:
                freqs.update(by_freq.keys())
    else:
        freqs = set()
        for by_freq in recordings.values():
            freqs.update(by_freq.keys())

    frequencias = np.array(sorted(freqs), dtype=np.float32)
    fases = np.zeros_like(frequencias, dtype=np.float32)
    return frequencias, fases


def load_data_from_users(
    users,
    dataset_path,
    frequencies=None,
    sessions=None,
    mat_key="storageDataAcquirement",
    filter_bandpass=True,
    sample_rate=256,
    freq_cut_low=6,
    freq_cut_high=50,
    filter_order=10,
    apply_car=False,
    car_mode="global",
    car_reference_channels=None,
    car_target_channels=None,
    channel_names=None,
    local_car_groups=None,
    window_size=None,
    window_mode="single",
    window_overlap=None,
    strict=True,
    normalize=True,
):
    """Load DESTINE users and build benchmark-compatible arrays.

    Returns:
        list[np.ndarray]: One entry per user, each with shape:
            (channels, time, num_freqs, num_trials)
    """
    recordings = discover_recordings(dataset_path)
    all_data = []

    frequencies_set = None
    if frequencies is not None:
        frequencies_set = {float(f) for f in frequencies}

    sessions_set = None
    if sessions is not None:
        sessions_set = {int(s) for s in sessions}

    for user in tqdm(users, desc="Loading DESTINE users"):
        subject = str(user).strip().lower()
        if subject not in recordings:
            available = sorted(recordings.keys())
            raise FileNotFoundError(
                f"No files found for subject '{subject}'. Available subjects: {available}"
            )

        by_freq = recordings[subject]
        selected_freqs = sorted(by_freq.keys())
        if frequencies_set is not None:
            selected_freqs = [f for f in selected_freqs if f in frequencies_set]

        if not selected_freqs:
            raise ValueError(f"No frequencies available for subject '{subject}' with the given filter.")

        loaded_trials_by_freq = {}
        for freq in selected_freqs:
            by_session = by_freq[freq]
            selected_sessions = sorted(by_session.keys())
            if sessions_set is not None:
                selected_sessions = [s for s in selected_sessions if s in sessions_set]

            if not selected_sessions:
                if strict:
                    raise ValueError(
                        f"No sessions for subject '{subject}', frequency {freq} with the given filter."
                    )
                continue

            trials = []
            for session in selected_sessions:
                eeg = load_mat_eeg(by_session[session], mat_key=mat_key)

                if apply_car:
                    if car_mode == "local":
                        eeg = apply_local_car(
                            eeg,
                            channel_names=channel_names,
                            local_groups=local_car_groups,
                        )
                    elif car_mode == "global":
                        eeg = apply_global_car(
                            eeg,
                            reference_channels=car_reference_channels,
                            target_channels=car_target_channels,
                        )
                    else:
                        raise ValueError(
                            f"Invalid car_mode='{car_mode}'. Use 'local' or 'global'."
                        )

                trials.append(eeg)

            if trials:
                loaded_trials_by_freq[freq] = trials

        if not loaded_trials_by_freq:
            raise ValueError(f"No data loaded for subject '{subject}'.")

        num_channels = next(iter(loaded_trials_by_freq.values()))[0].shape[0]
        num_time = next(iter(loaded_trials_by_freq.values()))[0].shape[1]

        trial_counts = [len(v) for v in loaded_trials_by_freq.values()]
        if len(set(trial_counts)) != 1:
            if strict:
                raise ValueError(
                    f"Unequal number of sessions across frequencies for subject '{subject}': {trial_counts}"
                )
            num_trials = min(trial_counts)
        else:
            num_trials = trial_counts[0]

        final_freqs = sorted(loaded_trials_by_freq.keys())
        user_data = np.zeros(
            (num_channels, num_time, len(final_freqs), num_trials), dtype=np.float32
        )

        for f_idx, freq in enumerate(final_freqs):
            for trial_idx in range(num_trials):
                user_data[:, :, f_idx, trial_idx] = loaded_trials_by_freq[freq][trial_idx]

        if filter_bandpass:
            user_data = bandpass_filter(
                user_data,
                taxa_amostragem=sample_rate,
                freq_corte_low=freq_cut_low,
                freq_corte_high=freq_cut_high,
                ordem_filtro=filter_order,
            )

        if window_size is not None:
            user_data = split_trials_into_windows(
                user_data,
                window_size=int(window_size),
                mode=window_mode,
                window_overlap=window_overlap,
            )

        if normalize:
            mean = np.mean(user_data, axis=1, keepdims=True)
            std = np.std(user_data, axis=1, keepdims=True)
            std[std == 0] = 1.0  # Prevent division by zero
            user_data = (user_data - mean) / std

        all_data.append(user_data)

    return all_data


__all__ = [
    "DEFAULT_CHANNEL_NAMES",
    "LOCAL_CAR_GROUPS",
    "apply_global_car",
    "apply_local_car",
    "bandpass_filter",
    "build_tensors_no_cca",
    "build_tensors_with_cca",
    "build_tensors_with_cca_joint",
    "build_tensors_with_fbcca",
    "build_tensors_with_fbcca_joint",
    "discover_recordings",
    "filter_signals_subbands",
    "load_data_from_users",
    "load_freq_phase",
    "load_mat_eeg",
    "parse_destine_filename",
    "split_trials_into_windows",
]
