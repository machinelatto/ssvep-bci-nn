import numpy as np
import scipy
from scipy.signal import cheby1, filtfilt
from utils.utils import get_desired_freqs_and_classes


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


def pre_process_dataset(dataset_path, channels, classes, sampling_rate, subban_no=0):
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
    if subban_no >= 2:
        print(f"Dividing signal into {subban_no} sub-bands.")
        processed = filter_signals_subbands(signals, subban_no, sampling_rate)
        np.save(
            f"pre_processed_{subban_no}_subbands_{len(classes)}_classes.npy",
            processed,
        )
        np.save(f"labels_{len(classes)}_classes.npy", labels)
        return processed, labels
    else:
        np.save(f"processed_{len(classes)}_classes.npy", signals)
        np.save(f"labels_processed_{len(classes)}_classes.npy", labels)
        return signals, labels
