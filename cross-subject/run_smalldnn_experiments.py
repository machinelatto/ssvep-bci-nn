"""
Run SmallDNN cross-subject experiments.
Supports CCA preprocessing and non-CCA preprocessing via CLI flags.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from cross_subject_utils import (
    evaluate,
    EarlyStopping,
    load_data_from_users,
)
from benchmark_dataset import (
    build_tensors_no_cca,
    build_tensors_with_cca,
    load_freq_phase,
)
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
    early_stopping=None,
):
    """Train the model with optional early stopping based on validation metrics."""
    model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

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

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        if early_stopping is not None:
            if early_stopping(model, val_accuracy, epoch):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            if epoch == 0 or val_accuracy > max(val_accuracies[:-1]):
                torch.save(model.state_dict(), save_path)

    if early_stopping is not None:
        model = early_stopping.load_best_model(model)
    else:
        model.load_state_dict(torch.load(save_path))

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run SmallDNN cross-subject experiments")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--n-filters", type=int, default=120)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user-start", type=int, default=1)
    parser.add_argument("--user-end", type=int, default=10)
    parser.add_argument("--num-freqs", type=int, default=8)
    parser.add_argument(
        "--subbands",
        type=int,
        default=1,
        help="Number of subbands to generate in preprocessing. New SMALLDNN expects 1.",
    )
    parser.add_argument(
        "--subband-merge",
        type=str,
        choices=["first", "mean"],
        default="first",
        help="How to collapse to one subband if preprocessing returns multiple subbands.",
    )
    parser.add_argument("--results-subdir", type=str, default=None)
    parser.add_argument("--use-cca", dest="use_cca", action="store_true")
    parser.add_argument("--no-use-cca", dest="use_cca", action="store_false")
    parser.set_defaults(use_cca=True)
    return parser.parse_args()


def ensure_single_subband(x_np, merge_strategy="first"):
    """Ensure tensor shape is compatible with SMALLDNN ([N, C, T] or [N, 1, C, T])."""
    if x_np.ndim == 3:
        return x_np
    if x_np.ndim == 4 and x_np.shape[1] == 1:
        return x_np
    if x_np.ndim == 4 and x_np.shape[1] > 1:
        if merge_strategy == "mean":
            return x_np.mean(axis=1, keepdims=True)
        return x_np[:, :1, :, :]
    raise ValueError(f"Unexpected input tensor shape for SMALLDNN: {x_np.shape}")


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Using device: {device}")
    print(f"use_cca={args.use_cca}")
    print(f"subbands={args.subbands}, subband_merge={args.subband_merge}")

    frequencias, fases = load_freq_phase()

    sample_rate = 250
    delay = 160
    num_harmonica = 3
    inform_fase = 0

    occipital_electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
    users = list(range(args.user_start, args.user_end + 1))
    users_to_run = users.copy()  # Ex.: [1, 5, 10]
    frequencias_desejadas = frequencias[: args.num_freqs]
    indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

    # Optional CAR configuration on loaded data
    apply_car = True
    car_reference_channels = occipital_electrodes
    car_target_channels = occipital_electrodes

    print("Users of interest:", users)
    print("Users to run:", users_to_run)
    print("Frequencies of interest:", frequencias_desejadas)
    print("Indices of frequencies of interest:", indices)

    print("\nLoading data from all users...")
    all_data = load_data_from_users(
        users=users,
        visual_delay=delay,
        dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
        sample_rate=sample_rate,
        filter_bandpass=False,
        apply_car=apply_car,
        car_reference_channels=car_reference_channels,
        car_target_channels=car_target_channels,
    )

    tamanho_da_janela_seg = args.window
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))
    epochs = args.epochs

    print(f"\n{'=' * 100}")
    print(f"Window size: {tamanho_da_janela} samples ({tamanho_da_janela_seg} s)")
    print(f"{'=' * 100}")

    if args.results_subdir is not None:
        base_subdir = args.results_subdir
    else:
        base_subdir = "CCA_small_DNN" if args.use_cca else "small_DNN"

    exp_dir = Path(
        f"35_8_optimized/{base_subdir}_CAR/{len(users)}_users_{len(frequencias_desejadas)}_freqs_{tamanho_da_janela_seg}_s/"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    metricas_usuarios = []

    for test_user in users_to_run:
        print(f"\nProcessing User {test_user}")
        train_users = [u for u in users if u != test_user]
        print(f"Train Users: {train_users}")

        train_data = np.concatenate(
            [all_data[users.index(u)] for u in train_users], axis=-1
        )
        test_data = all_data[users.index(test_user)]

        if args.use_cca:
            x_train_np, x_test_np, labels_train, labels_test, channels_for_model = build_tensors_with_cca(
                train_data,
                test_data,
                occipital_electrodes,
                frequencias,
                fases,
                indices,
                num_harmonica,
                inform_fase,
                tamanho_da_janela,
                apply_subband_filter=args.subbands > 0,
                subban_no=max(1, args.subbands),
            )
        else:
            x_train_np, x_test_np, labels_train, labels_test, channels_for_model = build_tensors_no_cca(
                train_data,
                test_data,
                occipital_electrodes,
                frequencias,
                indices,
                tamanho_da_janela,
                apply_subband_filter=args.subbands > 0,
                subban_no=max(1, args.subbands),
            )

        # New SMALLDNN expects one subband channel in 4D tensors.
        x_train_np = ensure_single_subband(x_train_np, merge_strategy=args.subband_merge)
        x_test_np = ensure_single_subband(x_test_np, merge_strategy=args.subband_merge)

        mapeamento = {
            rotulo: i for i, rotulo in enumerate(sorted(frequencias_desejadas))
        }
        rotulos_treinamento = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in labels_train
            ]
        )
        rotulos_teste = torch.tensor(
            [
                mapeamento[rotulo.item()] if hasattr(rotulo, "item") else mapeamento[rotulo]
                for rotulo in labels_test
            ]
        )

        x_treino = torch.tensor(x_train_np, dtype=torch.float32).to(device)
        x_teste = torch.tensor(x_test_np, dtype=torch.float32).to(device)
        y_treino = torch.tensor(rotulos_treinamento, dtype=torch.long).to(device)
        y_teste = torch.tensor(rotulos_teste, dtype=torch.long).to(device)
        print(f"X_train: {x_treino.shape}")
        print(f"X_test: {x_teste.shape}")
        print(f"Y_train: {y_treino.shape}")
        print(f"Y_test: {y_teste.shape}")

        model = SMALLDNN(
            num_classes=len(frequencias_desejadas),
            channels=channels_for_model,
            samples=tamanho_da_janela,
            subbands=1,
            n_filters=args.n_filters,
            dropout_rate=args.dropout_rate,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            verbose=True,
            delta=args.delta,
        )

        dataset = TensorDataset(x_treino, y_treino)
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(x_teste, y_teste),
            batch_size=10,
            shuffle=False,
        )

        print(f"Training for {epochs} epochs...")
        best_model = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=epochs,
            device=device,
            save_path=exp_dir.joinpath(f"best_model_user_{test_user}.pth"),
            early_stopping=early_stopping,
        )

        accuracy, recall, f1, cm = evaluate(best_model, test_loader)

        metricas_usuarios.append(
            {
                "usuario": test_user,
                "acuracia": accuracy,
                "recall": recall,
                "f1-score": f1,
                "confusion_matrix": cm,
                "use_cca": args.use_cca,
            }
        )
        print(
            f"User {test_user} Finished: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        metrics_path = exp_dir.joinpath("metricas.csv")
        pd.DataFrame([metricas_usuarios[-1]]).to_csv(
            metrics_path,
            mode="a",
            header=not metrics_path.exists(),
            index=False,
        )

        print("-" * 50)

    print(f"Experiment completed for window size {tamanho_da_janela_seg} s.")
    print("\n" + "=" * 100)
    print("All experiments completed!")


if __name__ == "__main__":
    main()
