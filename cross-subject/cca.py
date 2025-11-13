import numpy as np


def CCA_otimizacao(X, Y):

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
    Cxx = S[:colunas_X, :colunas_X]
    # Autocovariância de Y
    Cyy = S[colunas_X:, colunas_X:]
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


def matriz_referencia(
    numero_de_harmonicas, fase_inicial, sessoes, frequencia, fase, numero_de_amostras
):
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

    print(f"theta: {theta}")
    # Gerando sinais senoidais e cossenoidais
    for k in range(1, numero_de_harmonicas + 1):
        y1 = np.sin(2 * np.pi * k * frequencia * t + theta)
        y2 = np.cos(2 * np.pi * k * frequencia * t + theta)
        y.append(y1)
        y.append(y2)
    # Transpõe o array Y
    y = np.array(y)
    y = np.transpose(y)
    # Repete o array para coincidir com o número de sessões
    print(f"y shape: {y.shape}")
    Y = np.tile(y, (sessoes, 1))
    print(f"Y shape: {Y.shape}")
    # Retorna a Matriz de sinais de referência
    return Y
