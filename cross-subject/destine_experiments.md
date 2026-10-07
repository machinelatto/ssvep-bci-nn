# Destine

- 10 users
- 6 frequencias + olho fechado: 0, 6, 7.5, 12, 15, 20, 30
- 8 sessões de 12 segundos por frequencia.
- Ordem dos eletrodos: O1, O2, Oz, POz, Pz, PO3, PO4, PO7, PO8, P1, P2, Cz, C1, C2, CPz, FCz

## Testes com 6 frequenncias

### Parâmetros base

lr=0.001 weight-decay=0 -> sem regularização a princípio

1000 epocas com early stopping se passar 500 sem melhorar a validação, paciencia 0.0001

### CAR

Com car ficou melhor, usei o CAR com todos os 16 eletrodos do dataset.

### Normalização ou nao da entrada

Sem normalizar as entradas a rede apárentemente nao aprende nada:

![no-norm](imgs/destine_accuracy_itr_comparison_bars_1s_10_6_eegnets_single.png)

Normalizando cada amostra com média 0 e desvio padrão unoitário melhora os resultados:

![norm](imgs/destine_accuracy_itr_comparison_bars_1s_10_6_eegnets_single_norm.png)

### Adicionar regularização para evitar overfitting

Adicionar regularização L2 por meio do weight decay melhora ligeiramente

lr=0.001 wd=0.0001 -> AdamW

![reg](imgs/destine_accuracy_itr_comparison_bars_1s_10_6_eegnets_single_norm_reg.png)


### Janelamento ou nao

Sem janelamento são muito poucas amostras.

Para 6 frequencias temos apenas 6*8=48 sessões por usuário. Logo 48 para teste e 432 para treino

Teste 2 janelamentos:

#### janelamento de 256 amostras sem overlap

12 * 48 = 576 amostras por usuário ->  5148 no teste


![multi](imgs/destine_accuracy_itr_comparison_bars_1s_10_6_eegnets_multi_norm_reg.png)


#### janelamento com overlap de 128 amostras

![overlap](imgs/destine_accuracy_itr_comparison_bars_1s_10_6_eegnets_multi_overlap_norm_reg.png)

## Testes com 3 frequencias

### Janelamento