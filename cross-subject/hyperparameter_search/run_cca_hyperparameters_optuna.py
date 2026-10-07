"""Search CCA harmonics and high-pass filter cutoff with Optuna.

The objective is the mean leave-one-user-out frequency-classification accuracy
of the analytical CCA classifier used by runners/run_cca_experiments.py.
"""

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cca import CCA, reference_matrix
from benchmark_dataset import load_data_from_users


SAMPLE_RATE = 250
DELAY = 160
FREQ_CUT_LOW = 6
INFORM_PHASE = 0
USERS = list(range(1, 36))
OCCIPITAL_ELECTRODES = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
PADRONIZATION_EPSILON = 1e-8


def load_frequency_data(dataset_path, users, freq_cut_high):
    """Load and preprocess all users for one high-frequency cutoff."""
    return load_data_from_users(
        dataset_path=str(dataset_path),
        users=users,
        visual_delay=DELAY,
        filter_bandpass=True,
        apply_car=True,
        car_reference_channels=OCCIPITAL_ELECTRODES,
        car_target_channels=OCCIPITAL_ELECTRODES,
        sample_rate=SAMPLE_RATE,
        freq_cut_low=FREQ_CUT_LOW,
        freq_cut_high=freq_cut_high,
        filter_order=10,
    )


def evaluate_user(
    test_data,
    frequencies,
    phases,
    indices,
    num_harmonica,
    window_size,
    apply_padronization,
):
    """Evaluate one held-out user with analytical CCA."""
    labels = []
    predictions = []

    references = [
        reference_matrix(
            num_harmonica,
            INFORM_PHASE,
            1,
            frequencies[index],
            phases,
            window_size,
        )
        for index in indices
    ]

    for frequency_position, frequency_index in enumerate(indices):
        for trial in range(test_data.shape[-1]):
            eeg_matrix = test_data[
                OCCIPITAL_ELECTRODES,
                :window_size,
                frequency_index,
                trial,
            ]
            if apply_padronization:
                channel_mean = np.mean(eeg_matrix, axis=1, keepdims=True)
                channel_std = np.maximum(
                    np.std(eeg_matrix, axis=1, keepdims=True),
                    PADRONIZATION_EPSILON,
                )
                eeg_matrix = (eeg_matrix - channel_mean) / channel_std
            correlations = np.empty(len(indices), dtype=float)
            for candidate_position, reference in enumerate(references):
                _, _, correlation = CCA(eeg_matrix, reference)
                correlations[candidate_position] = correlation

            labels.append(frequency_index)
            predictions.append(indices[int(np.argmax(correlations))])

    return float(np.mean(np.asarray(labels) == np.asarray(predictions)))


def evaluate_configuration(
    all_data,
    frequencies,
    phases,
    users,
    indices,
    num_harmonica,
    window_size,
    apply_padronization,
    trial=None,
):
    """Return mean LOUO accuracy and the accuracy of every held-out user."""
    user_accuracies = []

    for user_position, user in enumerate(users):
        accuracy = evaluate_user(
            all_data[user_position],
            frequencies,
            phases,
            indices,
            num_harmonica,
            window_size,
            apply_padronization,
        )
        user_accuracies.append({"usuario": user, "acuracia": accuracy})

        if trial is not None:
            trial.report(float(np.mean([row["acuracia"] for row in user_accuracies])), user_position)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean([row["acuracia"] for row in user_accuracies])), user_accuracies


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna search for CCA harmonics and high filter cutoff."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/home/mateuschinelatto/Experiments/data/benchmark"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("CCA_optuna"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--min-harmonics", type=int, default=1)
    parser.add_argument("--max-harmonics", type=int, default=5)
    parser.add_argument(
        "--freq-cut-high-values",
        type=int,
        nargs="+",
        default=[90, 80, 70, 60, 50],
        help="Candidate high-frequency cutoffs in Hz.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = output_dir / "cca_hyperparameters.db"
    storage = f"sqlite:///{storage_path.resolve()}"
    study_name = "cca_num_harmonica_freq_cut_high_padronization_v7"

    freq_phase = scipy.io.loadmat(args.dataset_path / "Freq_Phase.mat")
    frequencies = np.round(freq_phase["freqs"], 2).ravel()
    phases = freq_phase["phases"]
    indices = list(range(len(frequencies)))
    window_size = int(np.ceil(args.window_seconds * SAMPLE_RATE))

    active_cutoff = None
    active_data = None

    def get_data(freq_cut_high):
        nonlocal active_cutoff, active_data
        if freq_cut_high != active_cutoff:
            if active_data is not None:
                del active_data
                gc.collect()
            print(f"Loading data with freq_cut_high={freq_cut_high} Hz")
            active_data = load_frequency_data(
                args.dataset_path,
                USERS,
                freq_cut_high,
            )
            active_cutoff = freq_cut_high
        return active_data

    def objective(trial):
        num_harmonica = trial.suggest_int(
            "num_harmonica",
            args.min_harmonics,
            args.max_harmonics,
        )
        freq_cut_high = trial.suggest_categorical(
            "freq_cut_high",
            args.freq_cut_high_values,
        )
        apply_padronization = trial.suggest_categorical(
            "apply_padronization",
            [True, False],
        )
        all_data = get_data(freq_cut_high)
        mean_accuracy, _ = evaluate_configuration(
            all_data,
            frequencies,
            phases,
            USERS,
            indices,
            num_harmonica,
            window_size,
            apply_padronization,
            trial=trial,
        )
        print(
            f"Trial {trial.number}: harmonics={num_harmonica}, "
            f"freq_cut_high={freq_cut_high}, "
            f"padronization={apply_padronization}, "
            f"mean_accuracy={mean_accuracy:.4f}"
        )
        return mean_accuracy

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    )
    completed_trials = len(study.trials)
    completed_params = {
        (
            trial.params.get("freq_cut_high"),
            trial.params.get("num_harmonica"),
            trial.params.get("apply_padronization"),
        )
        for trial in study.trials
        if trial.params.get("freq_cut_high") is not None
        and trial.params.get("num_harmonica") is not None
        and trial.params.get("apply_padronization") is not None
    }
    for freq_cut_high in args.freq_cut_high_values:
        for num_harmonica in range(args.min_harmonics, args.max_harmonics + 1):
            for apply_padronization in (True, False):
                if (
                    freq_cut_high,
                    num_harmonica,
                    apply_padronization,
                ) not in completed_params:
                    study.enqueue_trial(
                        {
                            "freq_cut_high": freq_cut_high,
                            "num_harmonica": num_harmonica,
                            "apply_padronization": apply_padronization,
                        }
                    )
    remaining_trials = max(0, args.n_trials - completed_trials)
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True)

    trials_path = output_dir / "tuning_results.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    best_params = study.best_params
    best_data = get_data(best_params["freq_cut_high"])
    best_accuracy, per_user = evaluate_configuration(
        best_data,
        frequencies,
        phases,
        USERS,
        indices,
        best_params["num_harmonica"],
        window_size,
        best_params["apply_padronization"],
    )
    pd.DataFrame(per_user).to_csv(output_dir / "best_per_user_results.csv", index=False)

    print("\nBest configuration:")
    print(f"  num_harmonica: {best_params['num_harmonica']}")
    print(f"  freq_cut_high: {best_params['freq_cut_high']} Hz")
    print(f"  apply_padronization: {best_params['apply_padronization']}")
    print(f"  mean_accuracy: {best_accuracy:.4f}")
    print(f"\nSaved trials to: {trials_path}")
    print(f"Saved per-user results to: {output_dir / 'best_per_user_results.csv'}")


if __name__ == "__main__":
    main()
