import argparse
import numpy as np
import scipy.io
import torch

from cross_subject_utils import load_data_from_users, get_windows
from cca import CCA, reference_matrix


def build_script_path(all_data, users, test_user, occipital, indices, frequencias, fases, num_harmonica, inform_fase, tamanho_da_janela):
    test_user_idx = users.index(test_user)
    train_users = [u for u in users if u != test_user]

    train_data = np.concatenate([all_data[users.index(u)] for u in train_users], axis=-1)
    test_data = all_data[test_user_idx]

    num_trials_train = train_data.shape[-1]
    num_trials_test = test_data.shape[-1]
    n_ch = len(occipital)
    n_freq = len(indices)

    y_train = np.zeros((num_harmonica * 2, tamanho_da_janela * num_trials_train, n_freq))
    x_train = np.zeros((n_ch, tamanho_da_janela * num_trials_train, n_freq))

    x_train_windows = np.zeros((num_trials_train * n_freq, n_ch, tamanho_da_janela))
    x_test_windows = np.zeros((num_trials_test * n_freq, n_ch, tamanho_da_janela))

    labels_train = []
    labels_test = []

    for k in range(n_freq):
        yk_train = reference_matrix(
            num_harmonica,
            inform_fase,
            num_trials_train,
            frequencias[indices[k]],
            fases,
            tamanho_da_janela,
        )
        y_train[:, :, k] = yk_train

        eeg_train_windows = train_data[occipital, :tamanho_da_janela, indices[k], :]
        eeg_test_windows = test_data[occipital, :tamanho_da_janela, indices[k], :]

        x_train[:, :, k] = eeg_train_windows.transpose(0, 2, 1).reshape(n_ch, -1)
        x_train_windows[k * num_trials_train : (k + 1) * num_trials_train] = eeg_train_windows.transpose(2, 0, 1)
        x_test_windows[k * num_trials_test : (k + 1) * num_trials_test] = eeg_test_windows.transpose(2, 0, 1)

        labels_train.extend([frequencias[indices[k]]] * num_trials_train)
        labels_test.extend([frequencias[indices[k]]] * num_trials_test)

    comb_x = []
    for k in range(n_freq):
        wx, _, _ = CCA(x_train[:, :, k], y_train[:, :, k])
        comb_x.append(wx)
    comb_x = np.column_stack(comb_x)

    tensor_train = np.zeros((n_freq * num_trials_train, n_freq, tamanho_da_janela))
    tensor_test = np.zeros((n_freq * num_trials_test, n_freq, tamanho_da_janela))

    for j in range(num_trials_train):
        for k in range(n_freq):
            janela_x = x_train_windows[k * num_trials_train + j]
            for freq_idx in range(n_freq):
                wx = comb_x[:, freq_idx]
                tensor_train[k * num_trials_train + j, freq_idx, :] = np.dot(wx, janela_x)

    for j in range(num_trials_test):
        for k in range(n_freq):
            janela_x = x_test_windows[k * num_trials_test + j]
            for freq_idx in range(n_freq):
                wx = comb_x[:, freq_idx]
                tensor_test[k * num_trials_test + j, freq_idx, :] = np.dot(wx, janela_x)

    mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias[: len(indices)]))}
    y_train_lbl = np.array([
        mapeamento[r.item()] if hasattr(r, "item") else mapeamento[r] for r in labels_train
    ])
    y_test_lbl = np.array([
        mapeamento[r.item()] if hasattr(r, "item") else mapeamento[r] for r in labels_test
    ])

    return {
        "X_train": x_train,
        "Y_train": y_train,
        "Combinadores_X": comb_x,
        "tensor_train": tensor_train,
        "tensor_test": tensor_test,
        "labels_train": y_train_lbl,
        "labels_test": y_test_lbl,
        "num_trials_train": num_trials_train,
        "num_trials_test": num_trials_test,
    }


def build_notebook_path(all_data, users, test_user, occipital, indices, frequencias, fases, num_harmonica, inform_fase, tamanho_da_janela):
    test_user_idx = users.index(test_user)
    train_users = [u for u in users if u != test_user]

    train_data = np.concatenate([all_data[users.index(u)] for u in train_users], axis=-1)
    test_data = all_data[test_user_idx]

    num_trials_train = train_data.shape[-1]
    num_trials_test = test_data.shape[-1]
    n_ch = len(occipital)
    n_freq = len(indices)

    y_train = np.zeros((tamanho_da_janela * num_trials_train, num_harmonica * 2, n_freq))
    y_test = np.zeros((tamanho_da_janela * num_trials_test, num_harmonica * 2, n_freq))

    for k in indices:
        yk_train = reference_matrix(
            num_harmonica,
            inform_fase,
            num_trials_train,
            frequencias[k],
            fases,
            tamanho_da_janela,
        )
        y_train[:, :, k] = yk_train.T

        yk_test = reference_matrix(
            num_harmonica,
            inform_fase,
            num_trials_test,
            frequencias[k],
            fases,
            tamanho_da_janela,
        )
        y_test[:, :, k] = yk_test.T

    x_train = np.zeros((tamanho_da_janela * num_trials_train, n_ch, n_freq))
    x_test = np.zeros((tamanho_da_janela * num_trials_test, n_ch, n_freq))

    for k in range(n_freq):
        eeg_matrix_train = train_data[occipital, :tamanho_da_janela, indices[k], :]
        eeg_matrix_test = test_data[occipital, :tamanho_da_janela, indices[k], :]

        eeg_matrix_train = np.transpose(eeg_matrix_train)
        eeg_matrix_test = np.transpose(eeg_matrix_test)

        eeg_matrix_train = np.concatenate(eeg_matrix_train, axis=0)
        eeg_matrix_test = np.concatenate(eeg_matrix_test, axis=0)

        x_train[:, :, k] = eeg_matrix_train
        x_test[:, :, k] = eeg_matrix_test

    comb_x = []
    for k in range(n_freq):
        wx, _, _ = CCA(x_train[:, :, k].T, y_train[:, :, k].T)
        comb_x.append(wx)
    comb_x = np.column_stack(comb_x)

    x_test_windows = []
    x_train_windows = []

    for k in range(n_freq):
        x_t, n_win_test = get_windows(x_test[:, :, k], tamanho_da_janela, include_last=False)
        x_v, n_win_train = get_windows(x_train[:, :, k], tamanho_da_janela, include_last=False)
        x_test_windows.append(x_t)
        x_train_windows.append(x_v)

    labels_train = []
    tensor_train = np.zeros((n_freq * num_trials_train, n_freq, tamanho_da_janela))
    cont = 0

    for m in range(n_freq):
        for j in range(n_win_train):
            janela_x = x_train_windows[m][j]
            labels_train.append(frequencias[indices[m]])
            cont_1 = 0
            for w in range(n_freq):
                wx = comb_x[:, w]
                proj = np.dot(wx, janela_x.T)
                tensor_train[cont, cont_1, :] = proj
                cont_1 += 1
            cont += 1

    labels_test = []
    tensor_test = np.zeros((n_freq * num_trials_test, n_freq, tamanho_da_janela))
    cont = 0

    for m in range(n_freq):
        for j in range(n_win_test):
            janela_x = x_test_windows[m][j]
            labels_test.append(frequencias[indices[m]])
            cont_1 = 0
            for w in range(n_freq):
                wx = comb_x[:, w]
                proj = np.dot(wx, janela_x.T)
                tensor_test[cont, cont_1, :] = proj
                cont_1 += 1
            cont += 1

    mapeamento = {rotulo: i for i, rotulo in enumerate(sorted(frequencias[: len(indices)]))}
    y_train_lbl = np.array([
        mapeamento[r.item()] if hasattr(r, "item") else mapeamento[r] for r in labels_train
    ])
    y_test_lbl = np.array([
        mapeamento[r.item()] if hasattr(r, "item") else mapeamento[r] for r in labels_test
    ])

    return {
        "X_train": x_train,
        "Y_train": y_train,
        "Combinadores_X": comb_x,
        "tensor_train": tensor_train,
        "tensor_test": tensor_test,
        "labels_train": y_train_lbl,
        "labels_test": y_test_lbl,
        "num_trials_train": num_trials_train,
        "num_trials_test": num_trials_test,
    }


def report_diff(name, a, b, atol=1e-7):
    same_shape = a.shape == b.shape
    if not same_shape:
        print(f"[FAIL] {name}: shape mismatch {a.shape} vs {b.shape}")
        return

    max_abs = float(np.max(np.abs(a - b)))
    mean_abs = float(np.mean(np.abs(a - b)))
    allclose = bool(np.allclose(a, b, atol=atol, rtol=1e-5))
    tag = "PASS" if allclose else "FAIL"
    print(f"[{tag}] {name}: shape={a.shape}, max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-user", type=int, default=1)
    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    sample_rate = 250
    filter_order = 10
    freq_cut_high = 50
    freq_cut_low = 6
    delay = 160

    num_harmonica = 3
    inform_fase = 0
    tamanho_da_janela_seg = 1.0
    tamanho_da_janela = int(np.ceil(tamanho_da_janela_seg * sample_rate))

    occipital = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
    users = list(range(1, 11))

    freq_phase = scipy.io.loadmat("/home/mateuschinelatto/Experiments/data/benchmark/Freq_Phase.mat")
    frequencias = np.round(freq_phase["freqs"], 2).ravel()
    fases = freq_phase["phases"]
    frequencias_desejadas = frequencias[:8]
    indices = [np.where(frequencias == freq)[0][0] for freq in frequencias_desejadas]

    print("Loading data once...")
    all_data = load_data_from_users(
        dataset_path="/home/mateuschinelatto/Experiments/data/benchmark/",
        users=users,
        visual_delay=delay,
        filter_bandpass=True,
        sample_rate=sample_rate,
        freq_cut_low=freq_cut_low,
        freq_cut_high=freq_cut_high,
        filter_order=filter_order,
    )

    script = build_script_path(
        all_data, users, args.test_user, occipital, indices, frequencias, fases,
        num_harmonica, inform_fase, tamanho_da_janela
    )
    notebook = build_notebook_path(
        all_data, users, args.test_user, occipital, indices, frequencias, fases,
        num_harmonica, inform_fase, tamanho_da_janela
    )

    print("\n=== Canonicalization checks (notebook->script orientation) ===")
    report_diff("X_train canonical", script["X_train"], np.transpose(notebook["X_train"], (1, 0, 2)))
    report_diff("Y_train canonical", script["Y_train"], np.transpose(notebook["Y_train"], (1, 0, 2)))

    print("\n=== CCA / tensor equality checks ===")
    report_diff("Combinadores_X", script["Combinadores_X"], notebook["Combinadores_X"], atol=1e-6)
    report_diff("tensor_train", script["tensor_train"], notebook["tensor_train"], atol=1e-6)
    report_diff("tensor_test", script["tensor_test"], notebook["tensor_test"], atol=1e-6)

    lbl_train_equal = np.array_equal(script["labels_train"], notebook["labels_train"])
    lbl_test_equal = np.array_equal(script["labels_test"], notebook["labels_test"])
    print(f"[{'PASS' if lbl_train_equal else 'FAIL'}] labels_train equality: {lbl_train_equal}")
    print(f"[{'PASS' if lbl_test_equal else 'FAIL'}] labels_test equality: {lbl_test_equal}")

    print("\n=== Shape summary ===")
    print("script tensor_train:", script["tensor_train"].shape)
    print("notebook tensor_train:", notebook["tensor_train"].shape)
    print("script tensor_test:", script["tensor_test"].shape)
    print("notebook tensor_test:", notebook["tensor_test"].shape)


if __name__ == "__main__":
    main()
