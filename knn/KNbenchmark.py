# @Author: Roberio Santos
# @Data: 1 de julho de 2023
# @Descrição: Realiza o benchmark do classificador KNeighborsClassifier
#             nos datasets MNIST e Fashion MNIST.

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Carregar os datasets MNIST e Fashion MNIST
mnist = fetch_openml('mnist_784', version=1)
fashion_mnist = fetch_openml('Fashion-MNIST', version=1)

datasets = [('MNIST', mnist), ('Fashion MNIST', fashion_mnist)]

# Definir as combinações de parâmetros
parameters = [
    {'weights': 'distance', 'n_neighbors': 5, 'p': 1},
    {'weights': 'distance', 'n_neighbors': 9, 'p': 1},
    {'weights': 'uniform', 'n_neighbors': 9, 'p': 1},
    {'weights': 'uniform', 'n_neighbors': 5, 'p': 1},
    {'weights': 'distance', 'n_neighbors': 5, 'p': 2},
    {'weights': 'distance', 'n_neighbors': 9, 'p': 2},
    {'weights': 'uniform', 'n_neighbors': 5, 'p': 2},
    {'weights': 'uniform', 'n_neighbors': 9, 'p': 2},
    {'weights': 'distance', 'n_neighbors': 1, 'p': 2},
    {'weights': 'uniform', 'n_neighbors': 1, 'p': 2},
    {'weights': 'uniform', 'n_neighbors': 1, 'p': 1},
    {'weights': 'distance', 'n_neighbors': 1, 'p': 1}
]

benchmark_results = []

for dataset_name, dataset in datasets:
    print(f"Executando KNeighborsClassifier no conjunto de dados {dataset_name}:")
    for params in parameters:
        print(f"Parâmetros: {params}")
        accuracies = []
        execution_times = []
        for i in range(5):
            # Dividir o conjunto de dados em treinamento e teste
            X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=i)
            
            # Criar e ajustar o classificador KNeighborsClassifier com os parâmetros especificados
            classifier = KNeighborsClassifier(**params)
            
            start_time = time.time()
            classifier.fit(X_train, y_train)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Fazer previsões no conjunto de teste
            y_pred = classifier.predict(X_test)
            
            # Calcular a acurácia e armazenar os resultados
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            execution_times.append(execution_time)

            # obs. os tempos médios de execução calculados referem-se ao tempo de execução do treinamento
            print(f"Iteração {i+1}: Acurácia = {accuracy:.4f}, Tempo de execução = {execution_time:.4f} segundos")
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        benchmark_results.append({'Dataset': dataset_name, 'Parâmetros': params, 'Acurácia Média': avg_accuracy, 'Tempo Médio': avg_execution_time})
        print(f"Média das acurácias obtidas: {avg_accuracy:.4f}")
        print(f"Média dos tempos de execução: {avg_execution_time:.4f} segundos")
        print()

# Criar o dataframe com os resultados do benchmark
benchmark_df = pd.DataFrame(benchmark_results)

# Imprimir o dataframe
print("Resultados do Benchmark:")
print(benchmark_df)

# Salvar o dataframe em uma planilha Excel
benchmark_df.to_excel('resultados_benchmark.xlsx', index=False)