"""Shared data loading and tensor-building utilities for benchmark SSVEP dataset."""

from __future__ import annotations

import numpy as np
import scipy.io
from tqdm import tqdm

from ssvep_shared import (
    bandpass_filter,
    build_tensors_no_cca,
    build_tensors_with_cca,
    build_tensors_with_cca_joint,
    build_tensors_with_fbcca,
    build_tensors_with_fbcca_joint,
    filter_signals_subbands,
    split_trials_into_windows,
)
from cross_subject_utils import car_filter


def load_data_from_users(
    users,
    visual_delay=160,
    dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
    filter_bandpass=False,
    apply_car=False,
    car_reference_channels=None,
    car_target_channels=None,
    sample_rate=250,
    freq_cut_low=6,
    freq_cut_high=70,
    filter_order=10,
    window_size=None,
    window_mode="single",
    window_overlap=None,
    normalize=False,
):
    """Load benchmark users from .mat files and slice post-stimulus interval.

    The baseline excerpt is always selected as [visual_delay : visual_delay + 1250].
    Optionally, each trial can then be CAR-referenced, split into windows,
    and normalized.
    """
    all_data = []
    for user in tqdm(users, desc="Carregando dados dos usuarios"):
        file_path = f"{dataset_path}/S{user}.mat"
        data = scipy.io.loadmat(file_path)["data"]
        if filter_bandpass:
            data = bandpass_filter(
                data, sample_rate, freq_cut_low, freq_cut_high, filter_order
            )

        # Keep the benchmark-compatible excerpt used by existing experiments.
        data = data[:, visual_delay : (visual_delay + 1250), :, :]

        if apply_car:

            data_car = data.copy()
            _, _, num_freqs, num_trials = data_car.shape

            for freq_idx in range(num_freqs):
                for trial_idx in range(num_trials):
                    data_car[:, :, freq_idx, trial_idx] = car_filter(
                        data_car[:, :, freq_idx, trial_idx],
                        reference_channels=car_reference_channels,
                        target_channels=car_target_channels,
                    )
            data = data_car

        if window_size is not None:
            data = split_trials_into_windows(
                data,
                window_size=int(window_size),
                mode=window_mode,
                window_overlap=window_overlap,
            )

        if normalize:
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            std[std == 0] = 1.0  # Prevent division by zero
            data = (data - mean) / std

        all_data.append(data)
    return all_data


def load_freq_phase(
    freq_phase_path="/home/mateuschinelatto/Experiments/data/benchmark/Freq_Phase.mat",
):
    """Load frequencies and phases from benchmark metadata file."""
    freq_phase = scipy.io.loadmat(freq_phase_path)
    frequencias = np.round(freq_phase["freqs"], 2).ravel()
    fases = freq_phase["phases"]
    return frequencias, fases


def build_tensors_ttcca(*args, **kwargs):
    """Placeholder for TTCCA tensor builder.

    This function was previously unfinished and remains intentionally unimplemented.
    """
    raise NotImplementedError("build_tensors_ttcca is not implemented yet.")


__all__ = [
    "bandpass_filter",
    "build_tensors_no_cca",
    "build_tensors_ttcca",
    "build_tensors_with_cca",
    "build_tensors_with_cca_joint",
    "build_tensors_with_fbcca",
    "build_tensors_with_fbcca_joint",
    "filter_signals_subbands",
    "load_data_from_users",
    "load_freq_phase",
    "split_trials_into_windows",
]
