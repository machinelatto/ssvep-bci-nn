"""
Run DESTINE DNN+FBCCA cross-subject experiments via CLI.
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

from cross_subject_utils import EarlyStopping, evaluate
from destine_dataset import (
    build_tensors_with_fbcca,
    build_tensors_with_fbcca_joint,
    load_data_from_users,
)
from dnn import SSVEPDNN


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


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    save_path,
    early_stopping,
):
    model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = running_train_loss / max(1, len(train_loader))
        train_accuracy = train_correct / max(1, train_total)

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = running_val_loss / max(1, len(val_loader))
        val_accuracy = val_correct / max(1, val_total)

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        if early_stopping(model, val_accuracy, epoch):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    best_model = early_stopping.load_best_model(model)
    torch.save(best_model.state_dict(), save_path)
    return best_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run DESTINE DNN+FBCCA cross-subject experiments")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/mateuschinelatto/Experiments/data/DESTINE",
    )
    parser.add_argument(
        "--users",
        type=str,
        default="subject_01,subject_02,subject_03",
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--num-harmonics", type=int, default=3)
    parser.add_argument("--inform-fase", type=int, default=0)
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
        help="Comma-separated channel indices used for model input.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="destine_results/dnnfbcca",
    )
    parser.add_argument(
        "--fbcca-mode",
        type=str,
        choices=["joint", "per-frequency"],
        default="joint",
    )
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--allow-missing", dest="strict", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    users = parse_str_list(args.users)
    test_users = parse_str_list(args.test_users) if args.test_users else users.copy()
    frequencies = parse_float_list(args.frequencies)
    sessions = parse_int_ranges(args.sessions)
    occipital_electrodes = np.array(parse_int_ranges(args.occipital_electrodes), dtype=int)
    window_samples = int(np.ceil(args.window * args.sample_rate))

    if args.loader_window_overlap is not None and args.loader_window_mode != "multiple":
        raise ValueError("--loader-window-overlap can only be used with --loader-window-mode multiple.")

    print(f"Using device: {device}")
    print(f"Users: {users}")
    print(f"Test users: {test_users}")
    print(f"Frequencies: {frequencies}")
    print(f"Sessions: {sessions}")
    print(f"FBCCA mode: {args.fbcca_mode}")

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
    phases = np.zeros_like(freq_arr, dtype=np.float32)
    indices = list(range(len(freq_arr)))
    exp_dir = Path(args.results_dir).joinpath(
        f"{len(users)}_users_{len(frequencies)}_freqs_{args.window}_s"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    build_tensors_fn = (
        build_tensors_with_fbcca_joint
        if args.fbcca_mode == "joint"
        else build_tensors_with_fbcca
    )

    metrics = []
    for test_user in test_users:
        print(f"\nProcessing test user: {test_user}")
        train_users = [u for u in users if u != test_user]

        train_data = np.concatenate([all_data[users.index(u)] for u in train_users], axis=-1)
        test_data = all_data[users.index(test_user)]

        x_train, x_test, labels_train, labels_test, channels_for_model = build_tensors_fn(
            train_data=train_data,
            test_data=test_data,
            occipital_electrodes=occipital_electrodes,
            frequencias=freq_arr,
            fases=phases,
            indices=indices,
            num_harmonica=args.num_harmonics,
            inform_fase=args.inform_fase,
            tamanho_da_janela=window_samples,
            mean_center=True,
            subban_no=3,
            sampling_rate=args.sample_rate,
        )
        print(
            "Sample counts -> "
            f"train: {len(labels_train)}, "
            f"test: {len(labels_test)}"
        )

        mapping = {label: i for i, label in enumerate(sorted(freq_arr))}
        y_train = torch.tensor([mapping[float(v)] for v in labels_train], dtype=torch.long)
        y_test = torch.tensor([mapping[float(v)] for v in labels_test], dtype=torch.long)

        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        x_test_t = torch.tensor(x_test, dtype=torch.float32)

        dataset = TensorDataset(x_train_t, y_train)
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False)
        test_loader = DataLoader(
            TensorDataset(x_test_t, y_test),
            batch_size=args.test_batch_size,
            shuffle=False,
        )

        model = SSVEPDNN(
            num_classes=len(frequencies),
            channels=channels_for_model,
            samples=window_samples,
            subbands=3,
            first_dropout=0.5,
            second_dropout=0.5,
            third_dropout=0.95,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
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

        best_model = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.epochs,
            device=device,
            save_path=exp_dir.joinpath(f"best_model_user_{test_user}.pth"),
            early_stopping=early_stopping,
        )

        accuracy, recall, f1, _ = evaluate(best_model, test_loader)
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

    print("\nAll DESTINE DNN+FBCCA experiments completed.")


if __name__ == "__main__":
    main()