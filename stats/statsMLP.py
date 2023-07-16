import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Carregar os dados da planilha Excel
df = pd.read_excel('resultados_benchmarkMLP.xlsx')

# Extrair as acurácias médias para cada dataset
mnist_acuracias = df[df['Dataset'] == 'MNIST']['Acurácia Média'].values
fashion_mnist_acuracias = df[df['Dataset'] == 'Fashion MNIST']['Acurácia Média'].values

# Exibir os vetores de acurácias médias
print("Acurácias Médias do MNIST:", mnist_acuracias)
print("Acurácias Médias do Fashion MNIST:", fashion_mnist_acuracias)

# Cálculo das diferenças entre as médias das acurácias
diferencas = np.array(mnist_acuracias) - np.array(fashion_mnist_acuracias)

# Verificação da normalidade das diferenças usando o teste de Shapiro-Wilk
normalidade_teste = stats.shapiro(diferencas)
p_valor = normalidade_teste.pvalue

# Plot dos histogramas das diferenças
plt.hist(diferencas, bins='auto')
plt.xlabel('Diferenças nas Acurácias')
plt.ylabel('Frequência')
plt.title('Histograma das Diferenças nas Acurácias')
plt.show()

# Realização do teste t independente
t_statistic, p_value = stats.ttest_ind(mnist_acuracias, fashion_mnist_acuracias)

# Exibição dos resultados
print("Diferenças entre as médias das acurácias:", diferencas)
print("P-valor do teste de normalidade (Shapiro-Wilk):", p_valor)

if p_valor < 0.05:
    print("As diferenças não seguem uma distribuição normal.")
else:
    print("As diferenças seguem uma distribuição normal.")

print("Valor do t-estatístico:", t_statistic)
print("Valor p:", p_value)

# Realização do teste de hipótese
nivel_significancia = 0.05

if p_value < nivel_significancia:
    print("Rejeita-se a hipótese nula. Há uma diferença significativa nas médias das acurácias.")
else:
    print("Não se rejeita a hipótese nula. Não há evidência suficiente para afirmar uma diferença significativa nas médias das acurácias.")
