# -*- coding: utf-8 -*-
"""
Utilitários e funções auxiliares para análise e visualização
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def detect_outliers_iqr(series):
    """
    Detecta outliers usando Intervalo Interquartil (IQR)
    
    Args:
        series: Série pandas
        
    Returns:
        tuple: (outliers mask, lower bound, upper bound)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    
    return outliers, lower_bound, upper_bound


def analyze_outliers(series, column_name):
    """
    Analisa e imprime informações sobre outliers de uma coluna
    
    Args:
        series: Série pandas
        column_name (str): Nome da coluna
    """
    outliers, lower, upper = detect_outliers_iqr(series)
    
    print(f"\n{column_name} - Estatísticas:")
    print(f"Q1: {series.quantile(0.25):.2f}")
    print(f"Q3: {series.quantile(0.75):.2f}")
    print(f"IQR: {(series.quantile(0.75) - series.quantile(0.25)):.2f}")
    print(f"\nLimites:")
    print(f"Lower bound: {lower:.2f}")
    print(f"Upper bound: {upper:.2f}")
    print(f"Outliers detectados: {outliers.sum()} ({outliers.sum()/len(series)*100:.1f}%)")


def plot_pca(X_scaled, y, n_components=2):
    """
    Visualiza dados usando PCA
    
    Args:
        X_scaled: Features escaladas
        y: Labels
        n_components (int): Número de componentes PCA
        
    Returns:
        DataFrame: Dados transformados com labels
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=[f'Componente Principal {i+1}' for i in range(n_components)]
    )
    pca_df['Type'] = y
    
    sns.scatterplot(
        x='Componente Principal 1',
        y='Componente Principal 2',
        hue='Type',
        data=pca_df,
        palette='viridis',
        s=100,
        alpha=0.8
    )
    plt.title('PCA 2 Componentes para Dados de Vidro, Colorido por Tipo')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    
    return pca_df


def plot_confusion_matrix_analysis(cm, class_labels, dataset):
    """
    Analisa e imprime informações sobre a matriz de confusão
    
    Args:
        cm: Matriz de confusão
        class_labels (list): Labels das classes
        dataset: Dataset original (para acessar tipos únicos)
    """
    print("\nAnálise da Matriz de Confusão:")
    print("Classes com maior taxa de erro:")
    
    for i, class_label in enumerate(sorted(dataset["Type"].unique())):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        print(f"Classe {class_label}: {correct}/{total} corretos ({accuracy*100:.1f}%)")
        
        if total > 0:
            errors = [(j, cm[i, j]) for j in range(len(cm[i])) if i != j and cm[i, j] > 0]
            if errors:
                errors.sort(key=lambda x: x[1], reverse=True)
                print(f"  Principais confusões: ", end="")
                for j, count in errors[:2]:
                    print(f"Classe {sorted(dataset['Type'].unique())[j]} ({count}x) ", end="")
                print()


def compare_models(results_dict):
    """
    Compara resultados de múltiplos modelos
    
    Args:
        results_dict: Dicionário com resultados de cada modelo
        
    Returns:
        DataFrame: Comparação de acurácia dos modelos
    """
    comparison = []
    for model_name, results in results_dict.items():
        comparison.append({
            'Modelo': model_name,
            'Treino': results['acc_train'],
            'Validação': results['acc_val'],
            'Teste': results['acc_test']
        })
    
    df_comparison = pd.DataFrame(comparison).sort_values('Teste', ascending=False)
    return df_comparison


def plot_model_comparison(df_comparison):
    """
    Plota comparação de acurácia entre modelos
    
    Args:
        df_comparison: DataFrame com resultados
    """
    plt.figure(figsize=(10, 5))
    colors = ["tomato" if acc == df_comparison["Teste"].max() else "steelblue"
              for acc in df_comparison["Teste"]]
    
    plt.barh(df_comparison["Modelo"], df_comparison["Teste"], color=colors)
    plt.xlabel("Acurácia (Teste)")
    plt.title("Comparação de Acurácia dos Modelos")
    plt.xlim(0.5, 1.0)
    plt.gca().invert_yaxis()
    
    for i, v in enumerate(df_comparison["Teste"]):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")
    
    plt.tight_layout()
    plt.show()
