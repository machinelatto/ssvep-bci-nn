"""
Run SmallDNN cross-subject hyperparameter tuning with Optuna.
Supports CCA preprocessing and non-CCA preprocessing via CLI flags.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.samplers import TPESampler
from optuna.trial import Trial
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd().parent))

from cca import CCA, reference_matrix
from cross_subject_utils import filter_signals_subbands, load_data_from_users
from smalldnn import SMALLDNN


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=100,
    device=0,
    save_path="best_model.pth",
    trial=None,
):
    """Train model and return best validation accuracy."""
    best_val_accuracy = -float("inf")
    best_model = None
    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        if trial is not None:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if best_model is not None:
        model.load_state_dict(best_model)

    return model, best_val_accuracy


def objective_per_user(
    trial: Trial,
    x_train,
    y_train,
    tamanho_da_janela,
    channels,
    num_classes,
    device,
    seed,
    num_epochs=25,
):
    """Optuna objective function for one left-out user."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.9)
    n_filters = trial.suggest_categorical("n_filters", [16, 32, 64, 96, 120])

    dataset = TensorDataset(x_train, y_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = SMALLDNN(
        num_classes=num_classes,
        channels=channels,
        samples=tamanho_da_janela,
        subbands=3,
        n_filters=n_filters,
        dropout_rate=dropout_rate,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    _, best_val_accuracy = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path="/tmp/trial_model_smalldnn.pth",
        trial=trial,
    )
    return best_val_accuracy


def tune_hyperparameters_per_user(
    x_train,
    y_train,
    tamanho_da_janela,
    channels,
    num_classes,
    device,
    seed,
    n_trials=20,
    tuning_epochs=25,
):
    """Run Optuna tuning for a single test user split."""
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective_per_user(
            trial,
            x_train,
            y_train,
            tamanho_da_janela,
            channels,
            num_classes,
            device,
            seed,
            num_epochs=tuning_epochs,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n{'=' * 60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (accuracy): {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 60}\n")

    return study.best_params, study


def build_training_data_with_cca(
    train_data,
    occipital_electrodes,
    frequencias,
    fases,
    indices,
    num_harmonica,
    inform_fase,
    tamanho_da_janela,
):
    """Build training tensors with CCA preprocessing."""
    _, _, _, num_trials_train = train_data.shape

    y_train = np.zeros(
        (num_harmonica * 2, tamanho_da_janela * num_trials_train, len(indices))
    )
    x_train_cca = np.zeros(
        (len(occipital_electrodes), tamanho_da_janela * num_trials_train, len(indices))
    )
    x_train_windows = np.zeros(
        (num_trials_train * len(indices), len(occipital_electrodes), tamanho_da_janela)
    )
    labels_train = []

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
        eeg_matrix_train = eeg_matrix_train_windows.transpose(0, 2, 1).reshape(
            len(occipital_electrodes), -1
        )

        x_train_cca[:, :, k] = eeg_matrix_train
        x_train_windows[k * num_trials_train : (k + 1) * num_trials_train] = (
            eeg_matrix_train_windows.transpose(2, 0, 1)
        )
        labels_train.extend([frequencias[indices[k]]] * num_trials_train)

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
            janela_x = janela_x - np.mean(janela_x, axis=1, keepdims=True)
            for freq_idx in range(len(indices)):
                wx = combinadores_x[:, freq_idx]
                projecao_x = np.dot(wx, janela_x)
                tensor_treinamento[k * num_trials_train + j, freq_idx, :] = projecao_x

    tensor_treinamento = filter_signals_subbands(
        tensor_treinamento, subban_no=3, sampling_rate=250
    )

    return tensor_treinamento, labels_train


def build_training_data_no_cca(
    train_data,
    occipital_electrodes,
    frequencias,
    indices,
    tamanho_da_janela,
):
    """Build training tensors without CCA preprocessing."""
    x_train = []
    labels_train = []

    for session in range(train_data.shape[3]):
        for freq_idx in range(len(indices)):
            eeg_trial = train_data[
                occipital_electrodes, :tamanho_da_janela, indices[freq_idx], session
            ]
            x_train.append(eeg_trial)
            labels_train.append(frequencias[indices[freq_idx]])

    x_train = np.array(x_train)
    x_train = filter_signals_subbands(x_train, subban_no=3, sampling_rate=250)
    return x_train, labels_train


def parse_args(use_cca_default=None):
    parser = argparse.ArgumentParser(description="SmallDNN Optuna hyperparameter search")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--tuning-epochs", type=int, default=25)
    parser.add_argument("--user-start", type=int, default=1)
    parser.add_argument("--user-end", type=int, default=35)
    parser.add_argument("--num-freqs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cca", dest="use_cca", action="store_true")
    parser.add_argument("--no-use-cca", dest="use_cca", action="store_false")

    if use_cca_default is None:
        parser.set_defaults(use_cca=True)
    else:
        parser.set_defaults(use_cca=use_cca_default)

    return parser.parse_args()


def main(use_cca_default=None):
    args = parse_args(use_cca_default=use_cca_default)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Using device: {device}")
    print(f"use_cca={args.use_cca}")

    freq_phase_path = "/home/mateuschinelatto/Experiments/data/benchmark/Freq_Phase.mat"
    freq_phase = scipy.io.loadmat(freq_phase_path)
    frequencias = np.round(freq_phase["freqs"], 2).ravel()
    fases = freq_phase["phases"]

    sample_rate = 250
    delay = 160
    num_harmonica = 3
    inform_fase = 0

    occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
    users = list(range(args.user_start, args.user_end + 1))
    frequencias_desejadas = frequencias[: args.num_freqs]
    indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

    # Optional CAR configuration on loaded data
    apply_car = False
    car_reference_channels = occipital_electrodes
    car_target_channels = None

    print("Users of interest:", users)
    print("Frequencies of interest:", frequencias_desejadas)

    print("\nLoading data from all users...")
    all_data = load_data_from_users(
        dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
        users=users,
        visual_delay=delay,
        filter_bandpass=False,
        apply_car=apply_car,
        car_reference_channels=car_reference_channels,
        car_target_channels=car_target_channels,
        sample_rate=sample_rate,
    )

    tamanho_da_janela_seg = args.window
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))

    print(f"\n{'=' * 100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'=' * 100}")

    mode_dir = "SMALLDNN_CCA_optuna" if args.use_cca else "SMALLDNN_optuna"
    exp_dir = Path(
        f"{mode_dir}/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    tuning_results = []

    for test_user_idx, test_user in enumerate(users):
        print(f"\n{'#' * 80}")
        print(f"Processing User {test_user} ({test_user_idx + 1}/{len(users)})")
        print(f"{'#' * 80}")
        train_users = [u for u in users if u != test_user]

        train_data = np.concatenate(
            [all_data[users.index(u)] for u in train_users], axis=-1
        )

        if args.use_cca:
            x_train_np, labels_train = build_training_data_with_cca(
                train_data,
                occipital_electrodes,
                frequencias,
                fases,
                indices,
                num_harmonica,
                inform_fase,
                tamanho_da_janela,
            )
            channels = len(indices)
        else:
            x_train_np, labels_train = build_training_data_no_cca(
                train_data,
                occipital_electrodes,
                frequencias,
                indices,
                tamanho_da_janela,
            )
            channels = len(occipital_electrodes)

        mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))}
        y_train = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in labels_train
            ]
        )

        x_train = torch.from_numpy(x_train_np.copy()).float().to(device)
        y_train = y_train.to(torch.long).to(device)

        print(f"X_train: {x_train.shape}")
        print(f"Y_train: {y_train.shape}")

        best_params, study = tune_hyperparameters_per_user(
            x_train,
            y_train,
            tamanho_da_janela,
            channels=channels,
            num_classes=len(frequencias_desejadas),
            device=device,
            seed=seed,
            n_trials=args.n_trials,
            tuning_epochs=args.tuning_epochs,
        )

        tuning_results.append(
            {
                "usuario": test_user,
                "use_cca": args.use_cca,
                "best_learning_rate": best_params.get("learning_rate"),
                "best_batch_size": best_params.get("batch_size"),
                "best_weight_decay": best_params.get("weight_decay"),
                "best_dropout_rate": best_params.get("dropout_rate"),
                "best_n_filters": best_params.get("n_filters"),
                "best_tuning_accuracy": study.best_value,
                "n_trials": len(study.trials),
            }
        )

        print(f"User {test_user} Finished: Best Accuracy={study.best_value:.4f}")

        df_tuning = pd.DataFrame(tuning_results)
        df_tuning.to_csv(exp_dir.joinpath("tuning_results.csv"), index=False)

    print("\n" + "=" * 100)
    print("All hyperparameter tuning completed!")


if __name__ == "__main__":
    main()
