"""
Run DESTINE CCA cross-subject experiments via CLI.

Example:
python run_destine_cca_experiments.py \
    --users subject_01,subject_02,subject_03,subject_04,subject_05,subject_06,subject_07,subject_08,subject_09,subject_10 \
  --frequencies 6,7.5,12,15,20,30 \
  --sessions 1-8 \
  --window 1.0 \
  --car-mode local
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from cca import CCA
from destine_dataset import load_data_from_users


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw):
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def parse_int_ranges(raw):
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            start_i = int(start)
            end_i = int(end)
            step = 1 if end_i >= start_i else -1
            values.extend(list(range(start_i, end_i + step, step)))
        else:
            values.append(int(chunk))
    return values


def build_reference_matrix(freq, num_harmonics, num_samples, sample_rate, phase=0.0):
    """Build reference matrix with shape (num_harmonics*2, num_samples)."""
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    y = []
    for k in range(1, num_harmonics + 1):
        y.append(np.sin(2 * np.pi * k * freq * t + phase))
        y.append(np.cos(2 * np.pi * k * freq * t + phase))
    return np.array(y, dtype=np.float32)


def evaluate(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    cm = confusion_matrix(labels, predictions)
    return accuracy, recall, f1, cm


def parse_args():
    parser = argparse.ArgumentParser(description="Run DESTINE CCA cross-subject experiments")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/mateuschinelatto/Experiments/data/DESTINE",
    )
    parser.add_argument(
        "--users",
        type=str,
        default="subject_01,subject_02,subject_03,subject_04,subject_05,subject_06,subject_07,subject_08,subject_09,subject_10",
        help="Comma-separated subject IDs.",
    )
    parser.add_argument(
        "--test-users",
        type=str,
        default=None,
        help="Optional comma-separated test users. Default: all users from --users.",
    )
    parser.add_argument(
        "--frequencies",
        type=str,
        default="6,7.5,12,15,20,30",
        help="Comma-separated frequencies in Hz.",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="1-8",
        help="Session list/ranges, e.g. '1-8' or '1,2,4-6'.",
    )
    parser.add_argument("--window", type=float, default=1.0, help="Window in seconds.")
    parser.add_argument(
        "--loader-window-mode",
        type=str,
        choices=["single", "multiple"],
        default="single",
        help="Loader windowing mode.",
    )
    parser.add_argument(
        "--loader-window-overlap",
        type=int,
        default=None,
        help="Optional overlap (in samples) for loader multiple-window mode.",
    )
    parser.add_argument("--sample-rate", type=int, default=256)
    parser.add_argument("--num-harmonics", type=int, default=3)
    parser.add_argument(
        "--car-mode",
        type=str,
        choices=["none", "local", "global"],
        default="global",
    )
    parser.add_argument("--no-bandpass", action="store_true")
    parser.add_argument("--freq-cut-low", type=float, default=6.0)
    parser.add_argument("--freq-cut-high", type=float, default=100.0)
    parser.add_argument("--filter-order", type=int, default=10)
    parser.add_argument(
        "--occipital-electrodes",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        help="Comma-separated channel indices used for CCA input.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="destine_results/cca",
    )
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--allow-missing", dest="strict", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    users = parse_str_list(args.users)
    test_users = parse_str_list(args.test_users) if args.test_users else users.copy()
    frequencies = parse_float_list(args.frequencies)
    sessions = parse_int_ranges(args.sessions)
    occipital_electrodes = np.array(parse_int_ranges(args.occipital_electrodes), dtype=int)
    window_samples = int(np.ceil(args.window * args.sample_rate))

    if args.loader_window_overlap is not None and args.loader_window_mode != "multiple":
        raise ValueError("--loader-window-overlap can only be used with --loader-window-mode multiple.")

    print(f"Users: {users}")
    print(f"Test users: {test_users}")
    print(f"Frequencies: {frequencies}")
    print(f"Sessions: {sessions}")

    all_data = load_data_from_users(
        users=users,
        dataset_path=args.dataset_path,
        frequencies=frequencies,
        sessions=sessions,
        filter_bandpass=not args.no_bandpass,
        sample_rate=args.sample_rate,
        freq_cut_low=args.freq_cut_low,
        freq_cut_high=args.freq_cut_high,
        filter_order=args.filter_order,
        apply_car=args.car_mode != "none",
        car_mode="local" if args.car_mode == "local" else "global",
        window_size=window_samples,
        window_mode=args.loader_window_mode,
        window_overlap=args.loader_window_overlap,
        strict=args.strict,
    )

    freq_arr = np.array(frequencies, dtype=np.float32)
    indices = list(range(len(freq_arr)))
    ref_bank = np.stack(
        [
            build_reference_matrix(
                freq=freq_arr[idx],
                num_harmonics=args.num_harmonics,
                num_samples=window_samples,
                sample_rate=args.sample_rate,
            )
            for idx in indices
        ],
        axis=0,
    )

    exp_dir = Path(args.results_dir).joinpath(
        f"{len(users)}_users_{len(frequencies)}_freqs_{args.window}_s"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for test_user in test_users:
        print(f"\nProcessing test user: {test_user}")
        test_data = all_data[users.index(test_user)]
        num_trials_test = test_data.shape[-1]
        test_samples = len(indices) * num_trials_test
        print(
            "Sample counts -> "
            f"train: 0 (CCA does not use supervised training samples), "
            f"test: {test_samples}"
        )

        labels = []
        predictions = []

        for class_idx in indices:
            for trial_idx in range(num_trials_test):
                eeg_matrix_test = test_data[
                    occipital_electrodes,
                    :window_samples,
                    class_idx,
                    trial_idx,
                ]

                labels.append(class_idx)
                corrs = np.zeros(len(indices), dtype=np.float32)
                for ref_idx in indices:
                    _, _, corr = CCA(eeg_matrix_test, ref_bank[ref_idx])
                    corrs[ref_idx] = corr

                predictions.append(int(np.argmax(corrs)))

        accuracy, recall, f1, _ = evaluate(labels, predictions)
        metrics.append(
            {
                "usuario": test_user,
                "acuracia": accuracy,
                "recall": recall,
                "f1-score": f1,
            }
        )
        print(
            f"User {test_user} done: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        metrics_path = exp_dir.joinpath("metricas.csv")
        pd.DataFrame([metrics[-1]]).to_csv(
            metrics_path,
            mode="a",
            header=not metrics_path.exists(),
            index=False,
        )

    print("\nAll DESTINE CCA experiments completed.")


if __name__ == "__main__":
    main()
