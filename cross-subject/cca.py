import numpy as np


def CCA(X: np.ndarray, Y: np.ndarray):
    """Canonical Correlation Analysis for SSVEP BCI

    Standard format for EEG/BCI applications:
    - X: EEG signals, shape (num_channels, num_timepoints)
    - Y: Reference signals, shape (num_features, num_timepoints)

    Args:
        X (np.ndarray): EEG signal matrix, shape (num_channels, num_timepoints)
        Y (np.ndarray): Reference signal matrix, shape (num_features, num_timepoints)

    Returns:
        Wx (np.ndarray): Spatial filter for X (num_channels,)
        Wy (np.ndarray): Spatial filter for Y (num_features,)
        correlation (float): Canonical correlation value
    """
    # Transpose to (num_timepoints, num_channels/features) for covariance calculation
    X = X.T  # shape: (num_timepoints, num_channels)
    Y = Y.T  # shape: (num_timepoints, num_features)

    # Calcula as linhas e colunas da matriz X
    linhas_X, colunas_X = X.shape
    # O número de amostras da matriz X é igual ao número de linhas de X
    num_amostras = linhas_X
    # Concatena X e Y
    V = np.concatenate((X, Y), axis=1)

    # Calcula a matriz S, matriz de covariância de X e Y.
    S = (1 / num_amostras) * (
        V.T @ V
        - (1 / num_amostras)
        * V.T
        @ np.ones((num_amostras, 1))
        @ np.ones((1, num_amostras))
        @ V
    )

    # Autocovariância de X
    Cxx = S[:colunas_X, :colunas_X] + 1e-6 * np.eye(colunas_X)
    # Autocovariância de Y
    Cyy = S[colunas_X:, colunas_X:] + 1e-6 * np.eye(S.shape[0] - colunas_X)
    # Covariância entre X e Y
    Cxy = S[:colunas_X, colunas_X:]

    # Calcula os autovalores e os autovetores de Cxx
    autovalores, autovetores = np.linalg.eig(Cxx)

    # Calcula a raiz quadrada dos autovalores
    raiz_autovalores = np.sqrt(autovalores)

    # Constrói a matriz diagonal dos autovalores
    raiz_lambda = np.diag(raiz_autovalores)

    # Calcula a inversa da matriz de autovetores
    inv_autovetores = np.linalg.inv(autovetores)

    # Calcula a raiz quadrada da matriz Cxx
    raiz_Cxx = np.dot(np.dot(autovetores, raiz_lambda), inv_autovetores)

    # Calcula a inversa da matriz raiz quadrada
    inv_raiz_Cxx = np.linalg.inv(raiz_Cxx)

    # Calcula os autovalores e os autovetores de Cyy
    autovalores, autovetores = np.linalg.eig(Cyy)

    # Calcula a raiz quadrada dos autovalores
    raiz_autovalores = np.sqrt(autovalores)

    # Constrói a matriz diagonal dos autovalores
    raiz_lambda = np.diag(raiz_autovalores)

    # Calcula a inversa da matriz de autovetores
    inv_autovetores = np.linalg.inv(autovetores)

    # Calcula a raiz quadrada da matriz Cyy
    raiz_Cyy = np.dot(np.dot(autovetores, raiz_lambda), inv_autovetores)

    # Calcula a inversa da matriz raiz quadrada
    inv_raiz_Cyy = np.linalg.inv(raiz_Cyy)

    # Calcula a matriz Kappa
    K = np.dot(inv_raiz_Cxx, np.dot(Cxy, inv_raiz_Cyy))

    # Decomposição da matriz Kappa, usando o método de decomposição em valores singulares
    Gamma, Lambda, Delta = np.linalg.svd(K)

    # Inversa da matriz Delta
    Delta = Delta.T

    # Calcula os combinadores lineares Wx e Wy.
    Wx = np.dot(inv_raiz_Cxx, Gamma[:, 0])
    Wy = np.dot(inv_raiz_Cyy, Delta[:, 0])

    correlation = Lambda[0]

    # Retorna os combinadores lineares.
    return Wx, Wy, Lambda[0]


def reference_matrix(
    numero_de_harmonicas=3, fase_inicial=0, sessoes=1, frequencia=10, fase=0, numero_de_amostras=250
):
    """Generate reference signals for SSVEP CCA

    Returns shape (num_harmonics*2, num_timepoints*num_sessions) following standard BCI convention:
    - Rows: features (sine/cosine pairs at different harmonics)
    - Columns: time points

    Args:
        numero_de_harmonicas: Number of harmonics to generate
        fase_inicial: Initial phase flag (0 = no phase, else use phase)
        sessoes: Number of sessions to tile
        frequencia: Frequency in Hz
        fase: Phase offset
        numero_de_amostras: Number of time samples per session

    Returns:
        Y: Reference signal matrix, shape (num_harmonics*2, num_timepoints*num_sessions)
    """
    # Taxa de amostragem
    dt = 1 / 250
    # Número de amostras
    n = np.arange(numero_de_amostras)
    # Vetor de tempo
    t = dt * n
    y = []
    if fase_inicial == 0:
        theta = 0
    else:
        theta = fase

    # Gerando sinais senoidais e cossenoidais
    for k in range(1, numero_de_harmonicas + 1):
        y1 = np.sin(2 * np.pi * k * frequencia * t + theta)
        y2 = np.cos(2 * np.pi * k * frequencia * t + theta)
        y.append(y1)
        y.append(y2)
    # Keep as (num_harmonics*2, num_timepoints) - standard BCI format
    y = np.array(y)  # shape: (num_harmonics*2, num_timepoints)
    Y = np.tile(y, (1, sessoes))  # shape: (num_harmonics*2, num_timepoints*num_sessions)
    return Y
