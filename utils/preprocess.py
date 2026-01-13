import numpy as np
import scipy
from scipy.signal import cheby1, filtfilt
from utils import get_desired_freqs_and_classes
from pathlib import Path


def filter_signals_subbands(eeg_signals, subban_no, sampling_rate):
    subjects, samples_per_subject, total_channels, sample_length = eeg_signals.shape
    AllData = np.zeros(
        (subjects, samples_per_subject, subban_no, total_channels, sample_length)
    )

    # Bandpass filters
    high_cutoff = [90] * subban_no
    low_cutoff = [i for i in range(8, 8 * (subban_no + 1), 8)]
    filter_order = 2
    passband_ripple = 1
    bp_filters = []

    for i in range(subban_no):
        b, a = cheby1(
            filter_order,
            passband_ripple,
            [low_cutoff[i], high_cutoff[i]],
            btype="band",
            fs=sampling_rate,
        )
        bp_filters.append((b, a))

    # Filtering
    for subject in range(subjects):
        sub_data = eeg_signals[subject]
        print(f"S{subject+1} data shape: {sub_data.shape}")
        for sample in range(samples_per_subject):
            tmp_raw = sub_data[sample]
            for sub_band in range(subban_no):
                processed_signal = np.zeros((total_channels, sample_length))
                b, a = bp_filters[sub_band]

                for ch_idx in range(total_channels):
                    processed_signal[ch_idx] = filtfilt(b, a, tmp_raw[ch_idx])

                AllData[subject, sample, sub_band, :, :] = processed_signal
    print(f"All data shape after filtering: {AllData.shape}")
    return AllData


def pre_process_dataset(
    dataset_path, channels, classes, sampling_rate, subban_no=0, split_trials=False
):
    subjects_files = [file for file in dataset_path.glob("*.npy")]
    freq_phase = scipy.io.loadmat(dataset_path.joinpath("Freq_Phase.mat"))
    freqs = np.round(freq_phase["freqs"], 2)

    signals = []
    labels = []

    def sort_key(path):
        subject_number = int(path.stem.split("_")[0][1:])
        return subject_number

    sorted_paths = sorted(subjects_files, key=sort_key)

    for subject_file in sorted_paths:
        windows = np.load(subject_file)
        if windows.ndim == 5:
            num_windows = windows.shape[4]
        else:
            num_windows = 1
        subj_signals, subj_labels = get_desired_freqs_and_classes(
            windows, freqs, channels, classes, num_windows
        )

        trials, samples_per_trial, n_channels, n_points = subj_signals.shape
        num_samples = samples_per_trial * trials
        signals.append(subj_signals.reshape([num_samples, n_channels, n_points]))
        labels.append(subj_labels.reshape(num_samples))

    signals = np.array(signals)
    labels = np.array(labels)
    print(signals.shape)
    print(labels.shape)
    labels = labels.reshape([35, *subj_labels.shape]) if split_trials else labels

    if subban_no >= 2:
        print(f"Dividing signals into {subban_no} sub-bands.")
        processed = filter_signals_subbands(signals, subban_no, sampling_rate)
        processed = (
            processed.reshape(
                [35, trials, samples_per_trial, subban_no, n_channels, n_points]
            )
            if split_trials
            else processed
        )
        np.save(
            f"pre_processed_{subban_no}_subbands_{len(classes)}_classes{split_trials}.npy",
            processed,
        )
        np.save(f"labels_{len(classes)}_classes{split_trials}.npy", labels)
        return processed, labels
    else:
        signals = (
            signals.reshape([35, *subj_signals.shape]) if split_trials else signals
        )
        np.save(f"processed_{len(classes)}_classes{split_trials}.npy", signals)
        np.save(f"labels_processed_{len(classes)}_classes{split_trials}.npy", labels)
        return signals, labels


if __name__ == "__main__":
    DATASET = Path("../../processed_datasets/sinais_filtrados_6_70_Hz_janelas_1s/")

    selected_freqs = np.array(
        [np.round(i, 2) for i in np.linspace(8, 15.8, 40)]
    )  # Frequências de interesse
    # selected_freqs = np.array([i for i in range(8, 16)])
    # selected_freqs = np.array([8.2, 10.8, 12.6, 15.4])
    channels = [
        47,
        53,
        54,
        55,
        56,
        57,
        60,
        61,
        62,
    ]  # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2

    s, l = pre_process_dataset(
        DATASET, channels, selected_freqs, 250, 1, split_trials=True
    )
    print(s.shape)
    print(l.shape)
