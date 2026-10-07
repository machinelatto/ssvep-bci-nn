"""Compare benchmark subjects with t-SNE across the available tensor builders.

For every subject, the tensor builder is fitted with the remaining subjects and
the held-out subject is appended to the common t-SNE input. This preserves the
leave-one-user-out setup used by the exploratory notebook while producing one
plot containing all subjects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE, UMAP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_dataset import build_tensors_no_cca, load_data_from_users, load_freq_phase
from ssvep_shared import build_tensors_with_cca, build_tensors_with_fbcca


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("/home/mateuschinelatto/Experiments/data/benchmark"))
    parser.add_argument("--freq-phase-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("images/tsne_all_subjects"))
    parser.add_argument("--users", nargs="+", type=int, default=list(range(1, 36)))
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_builder_inputs(
    train_data: list[np.ndarray],
    test_data: np.ndarray,
    users: list[int],
    test_user: int,
    electrodes: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
    indices: list[int],
    window_size: int,
):
    """Build all four tensor variants for one held-out subject."""
    train_users = [user for user in users if user != test_user]
    train = np.concatenate(
        [train_data[users.index(user)] for user in train_users], axis=-1
    )
    common = {
        "occipital_electrodes": electrodes,
        "frequencias": frequencies,
        "indices": indices,
        "tamanho_da_janela": window_size,
        "apply_subband_filter": False,
    }
    no_cca = build_tensors_no_cca(train, test_data, **common)
    cca = build_tensors_with_cca(
        train,
        test_data,
        fases=phases,
        num_harmonica=3,
        inform_fase=0,
        **common,
    )
    fbcca = build_tensors_with_fbcca(
        train,
        test_data,
        fases=phases,
        num_harmonica=3,
        inform_fase=0,
        occipital_electrodes=electrodes,
        frequencias=frequencies,
        indices=indices,
        tamanho_da_janela=window_size,
    )
    return {"raw": no_cca[1], "cca": cca[1], "fbcca": fbcca[1]}, {
        "raw": no_cca[3], "cca": cca[3], "fbcca": fbcca[3]
    }


def flatten_for_tsne(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(tensor.shape[0], -1).astype(np.float32)


def collect_subject_tensors(
    normalized_data: list[np.ndarray],
    raw_data: list[np.ndarray],
    users: list[int],
    electrodes: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
    indices: list[int],
    window_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    tensors = {"raw": [], "normalized": [], "cca": [], "fbcca": []}
    frequencies_by_method = {method: [] for method in tensors}
    users_by_method = {method: [] for method in tensors}

    for subject_index, user in enumerate(users):
        print(f"Building tensors for subject {user} ({subject_index + 1}/{len(users)})")
        normalized_result, labels = make_builder_inputs(
            normalized_data,
            normalized_data[subject_index],
            users,
            user,
            electrodes,
            frequencies,
            phases,
            indices,
            window_size,
        )
        raw_result, _ = make_builder_inputs(
            raw_data,
            raw_data[subject_index],
            users,
            user,
            electrodes,
            frequencies,
            phases,
            indices,
            window_size,
        )
        tensors["raw"].append(flatten_for_tsne(raw_result["raw"]))
        tensors["normalized"].append(flatten_for_tsne(normalized_result["raw"]))
        tensors["cca"].append(flatten_for_tsne(normalized_result["cca"]))
        tensors["fbcca"].append(flatten_for_tsne(normalized_result["fbcca"]))
        for method in frequencies_by_method:
            method_labels = labels["raw"] if method in ("raw", "normalized") else labels[method]
            frequencies_by_method[method].extend(method_labels)
            users_by_method[method].extend([user] * len(method_labels))

    return (
        {method: np.concatenate(values) for method, values in tensors.items() if values},
        {method: np.asarray(values) for method, values in frequencies_by_method.items()},
        {method: np.asarray(values) for method, values in users_by_method.items()},
    )


def make_plot(dataframes: dict[str, pd.DataFrame], column: str, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=300)
    methods = [("raw", "Dados Brutos"), ("normalized", "Z-score"), ("cca", "CCA"), ("fbcca", "FBCCA")]
    values = sorted(dataframes["raw"][column].unique())
    palette = dict(zip(values, sns.color_palette("tab10", n_colors=len(values))))

    for axis, (method, title) in zip(axes.flat, methods):
        sns.scatterplot(
            data=dataframes[method], x="tsne_0", y="tsne_1", hue=column,
            palette=palette, s=45, alpha=0.8, linewidth=0, ax=axis,
        )
        axis.set_title(f"t-SNE: {title}")
        axis.legend(title=column, loc="best", fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    freq_phase_path = args.freq_phase_path or args.dataset_path / "Freq_Phase.mat"
    frequencies, phases = load_freq_phase(freq_phase_path)
    indices = list(range(min(8, len(frequencies))))
    electrodes = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
    sample_rate = 250
    window_size = int(sample_rate * args.window_seconds)

    load_options = {
        "users": args.users,
        "dataset_path": str(args.dataset_path),
        "visual_delay": 160,
        "filter_bandpass": True,
        "apply_car": True,
        "car_reference_channels": electrodes,
        "car_target_channels": electrodes,
        "sample_rate": sample_rate,
        "freq_cut_low": 6,
        "freq_cut_high": 50,
        "filter_order": 10,
        "window_size": window_size,
        "window_mode": "single",
        "window_overlap": 0,
    }
    normalized_data = load_data_from_users(normalize=True, **load_options)
    raw_data = load_data_from_users(normalize=False, **load_options)
    tensors, labels_frequency, labels_user = collect_subject_tensors(
        normalized_data,
        raw_data,
        args.users,
        electrodes,
        frequencies,
        phases,
        indices,
        window_size,
    )

    tsne_dataframes = {}
    for method, tensor in tensors.items():
        embedding = UMAP(
            n_components=2,
            perplexity=args.perplexity,
            random_state=args.seed,
            max_iter=1000,
        ).fit_transform(tensor)
        tsne_dataframes[method] = pd.DataFrame({
            "tsne_0": embedding[:, 0],
            "tsne_1": embedding[:, 1],
            "Frequency": labels_frequency[method],
            "User": labels_user[method],
        })

    make_plot(tsne_dataframes, "Frequency", args.output_dir / "tsne_by_frequency.png")
    make_plot(tsne_dataframes, "User", args.output_dir / "tsne_by_user.png")
    print(f"Saved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()