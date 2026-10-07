# Experimentos com EEGNET, LOUO (Leave-One-User-Out) e indeitificação do porque a EEGNET+CCA com os 35 usuários não performa melhor que a EEGNET.

## Cenário anterior (Até a qualificação)

Neste cenário comparamos a EEGNET com 10 usuários e 8 frequências com o método CCA+EEGNET e observamos que CCA+EEGNET performa melhor:

![Acurácia - Cenário Anterior](images/acuracia_it_cross.png)

## Cenário atual

Após a qualificação diversas melhorias foram aplicadas no pré-processamento e no modelo da EEGNET:

- Adição do filtro CAR
- Adição de normalização de cada sinal de entrada
- Redução da janela do filtro passa banda de 6-70 Hz para 6-50 Hz
- Utilzação de 3 harmônicas para o método CCA, ao invés de 5 harmônicas

- Tuning de hyperparâmetros:
    - Aumento do tamanho do kernel da filtragem temporal da EEGNET -> metade da frequência de amostragem (125), antes era metade do tamanho da janela de entrada (0.4s: 50, 0.6s: 75, 0.8s: 100, 1.0s: 125)
    - Aumento do número de filtros temporais e espaciais da EEGNET -> 8 temporais e 16 espaciais
    - Ajuste de learning rate e batch size
    - Regularização L2 no otimizador Adam -> AdamW

- Aumento do número de épocas de treinamento
- Adição de early stopping (1000 épocas de treino, mas com early stopping de 500 épocas sem melhora na acurácia de validação)
- Adição de mais usuários (total de 35 usuários)

Após essas melhorias, a EEGNET tem a mesma performance da EEGNET+CCA, o que é um resultado interessante, pois a EEGNET+CCA tem um pré-processamento mais complexo.

#### 35 usuários 40 freqs - EEGNET

![Acurácia - Cenário Atual 35 usuários todas as freqs](images/acuracia_it_35_40.png)

#### 35 usuários 8 freqs
![Acurácia - Cenário Atual 35 usuários 1s](images/accuracy_itr_comparison_bars_1s_35_8_best.png)


## Experimentos e Hipóteses para explicar os resultados atuais

### Comparação dos resultados ainda para 10 usuários mas com o processamento novo (CAR, normalização, filtro de 6-50 Hz e 3 harmônicas para o CCA):

![Acurácia - Cenário Atual 10 usuários 1s](images/now_vs_quali_NEW.png)

Podemos perceber que 

### H1: Os 10 usuários iniciais tinham padrões de resposta ao estímulo mais semelhantes entre si, o que favorecia a EEGNET+CCA, enquanto os 35 usuários atuais tem respostas mais diferentes entre si, o que não favorece nem a EEGNET+CCA nem a EEGNET.

Caso os 10 usuários iniciais tivessem respostas mais semelhantes entre si, isso poderia favorecer a EEGNET+CCA, pois o pré processamento com CCA é baseado em concatenar os dados de todos os usuários e extrair as componentes comuns entre eles. Se os usuários tivessem respostas mais semelhantes, isso poderia facilitar a extração dessas componentes comuns e melhorar a performance da EEGNET+CCA. Por outro lado, se os 35 usuários atuais tivessem respostas mais diferentes entre si, isso poderia dificultar a extração dessas componentes comuns e não favorecer nem a EEGNET+CCA.

Notebook de análise de similaridade entre os usuários: [similarity_analysis.ipynb](notebooks/similarity_analysis.ipynb)

Não foi possível concluir se há maior similaridade entre os 10 usuários iniciais do que entre os 35 usuários atuais.

### H2: O pré-processamento mais complexo da EEGNET+CCA não é necessário para a EEGNET performar bem quando aumentamos o numero de usuários, filtros, épocas de treinamento e adicionamos o filtro CAR e a normalização.

Mesmo com o pré-processamento mais complexo da EEGNET+CCA, ela tem uma performance similar à EEGNET+CCA, o que sugere que o pré-processamento não é necessário para a EEGNET performar bem quando aumentamos o numero e tamanho dos filtros, e adicionamos o filtro CAR e a normalização. Modelos maiores podem aprender a extrair as características relevantes dos dados sem a necessidade de um pré-processamento tão complexo.

Isso também pode ser um motivo pelo qual aplicar o CCA antes da DNN não trouxe melhorias de desempenho.

Avaliar um classificador mais simples e janelas menores


## Próximos passos

- O método havia funcionado muito bem intra-subject, então talvez uma abordagem ensemble/MOE
- Avaliar melhor a questão do tamanho do modelo