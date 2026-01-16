import numpy as np
import scipy.io
from scipy import signal
from pathlib import Path
import argparse
import re

# Parâmetros do filtro
SAMPLING_RATE = 250  # Frequência de amostragem (Hz)
F_LOW = 6  # Frequência de corte inferior (Hz)
F_HIGH = 70  # Frequência de corte superior (Hz)
ORDER = 10  # Ordem do filtro


def filter_signal(original_signal: np.ndarray, filter_b, filter_a):
    # **Filtragem dos dados**
    # Aqui é realizado o processo de filtragem para todas as frequências, trials e eletrodos.

    num_eletrodos, num_amostras, num_freqs, num_trials = original_signal.shape

    filtered_signal = original_signal.copy()
    # Foi empregada a função filfilt da biblioteca Scipy para criar um filtro de fase zero,
    # garantindo
    # que a fase original dos sinais EEG
    # não seja distorcida durante o processo de filtragem.
    for f in range(num_freqs):  # Para cada frequência de estimulação
        for trial in range(num_trials):  # Para cada trial
            for eletrodo in range(num_eletrodos):  # Para cada eletrodo
                eletrodo_filtrado = signal.filtfilt(
                    filter_b, filter_a, filtered_signal[eletrodo, :, f, trial]
                )  # Filtragem com filtfilt
                filtered_signal[eletrodo, :, f, trial] = (
                    eletrodo_filtrado  # Substitui o dado original pelo filtrado
                )

    if np.any(np.isnan(filtered_signal)) or np.any(filtered_signal == 0):
        print("O tensor possui valores nulos ou 'não numéricos' (NaN).")
        return None
    else:
        print("O tensor não possui valores nulos ou 'não numéricos' (NaN).")
        return filtered_signal


def split_in_windows(eeg, num_amostras_janela, delay):
    # Obtém as dimensões dos dados EEG (tensor)
    num_eletrodos, num_amostras, num_freqs, num_trials = eeg.shape
    # Calcula o número de janelas possíveis, com base no número de amostras
    num_janelas = int((num_amostras - 250) / num_amostras_janela)

    # Cria um tensor vazio para armazenar os dados segmentados em janelas
    tensor = np.zeros(
        [num_eletrodos, num_amostras_janela, num_freqs, num_trials, num_janelas]
    )

    # itera sobre as frequências de estimulação
    for f in range(num_freqs):
        # Itera sobre as trials (sessões de aquisição)
        for t in range(num_trials):
            # Extrai o trial de interesse, aplicando o delay (tempo de latência) para segmentação
            trial = eeg[:, slice(delay, delay + 1250), f, t]

            # Itera sobre o numero de janelas, onde cada janela corresponde a
            # uma seção temporal do sinal EEG de 1 segundo
            for i in range(num_janelas):
                # Define os índices de início e fim da janela
                inicio = i * num_amostras_janela
                fim = (i + 1) * num_amostras_janela
                janela = trial[:, inicio:fim]  # Extrai os dados da janela

                # Salva a janela no tensor
                tensor[:, :, f, t, i] = janela
    if np.any(np.isnan(tensor)) or np.any(tensor == 0):
        print("O tensor possui valores nulos ou 'não numéricos' (NaN).")
    else:
        print("O tensor não possui valores nulos ou 'não numéricos' (NaN).")
    return tensor


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text for text in re.split(r"(\d+)", str(s))
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Filter and splt benchmark dataset in windows",
    )
    parser.add_argument("dataset")
    parser.add_argument("samples")
    parser.add_argument("output")

    args = parser.parse_args()
    dataset = Path(args.dataset)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    num_samples = int(args.samples)

    # Create filter
    b, a = signal.butter(
        ORDER,
        [F_LOW, F_HIGH],
        btype="bandpass",
        analog=False,
        output="ba",
        fs=SAMPLING_RATE,
    )
    freq_phase = scipy.io.loadmat(dataset.joinpath("Freq_Phase.mat"))
    freqs = np.round(freq_phase["freqs"], 2)
    subj_files = [f for f in Path(dataset).rglob("S*.mat")]
    subj_files = sorted(subj_files, key=natural_sort_key)
    for file in subj_files:
        subject = file.stem
        print(subject)
        og_signal = scipy.io.loadmat(file)["data"]
        filtered = filter_signal(og_signal, b, a)
        splitted = split_in_windows(filtered, num_samples, 160)
        segs = num_samples / SAMPLING_RATE
        filename = (
            f"{subject}_passa-banda_{F_LOW}_{F_HIGH}_Hz"
            + "_janelas_"
            + str(segs).replace(".", "_")
            + "s"
            + ".npy"
        )
        output_file = output_path.joinpath(filename)
        np.save(output_file, splitted)
