# Classificação de Vidro - Glass Classification Expert

## Descrição do Projeto

Sistema inteligente para classificação de tipos de vidro baseado em análise química usando Machine Learning. Este projeto implementa e compara três algoritmos de classificação: **KNN**, **SVM** e **Naive Bayes**.

**Autores**: Vitoria Ayres (2086138) e Leticia Ribeiro (2034293)

---

## Estrutura do Projeto

```
vidro-classificacao/
├── app.py                          # Aplicação Streamlit principal
├── requirements.txt                # Dependências do projeto
├── README.md                       # Este arquivo
│
├── src/                           # Módulos de código-fonte
│   ├── data_loader.py             # Carregamento e processamento de dados
│   ├── models.py                  # Treinamento e avaliação de modelos
│   └── utils.py                   # Funções auxiliares e visualizações
│
├── notebooks/                     # Jupyter Notebooks
│   └── analise_exploratoria.ipynb # Notebook revisado da análise
│
├── model/                         # Modelos salvos treinados
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   └── nb_model.pkl
│
├── reports/                       # Relatórios e análises
│   └── relatorio_atualizado.pdf   # Relatório final
│
└── data/                          # Datasets
    └── glass.csv                  # Dataset utilizado
```

---
## Dataset

Este é um conjunto de dados de identificação de vidro da UCI. Contém 10 atributos, incluindo o ID. A resposta é o tipo de vidro (7 valores).




### Tipos de Vidro Classificáveis

1. **Tipo 1**: Janela (Flutuação)
2. **Tipo 2**: Janela (Não-Flutuação)
3. **Tipo 3**: Veículo
4. **Tipo 5**: Recipiente
5. **Tipo 6**: Farol
6. **Tipo 7**: Mesa

*Nota: Tipo 4 não existe no dataset*



### Features (Características Químicas)

O modelo utiliza 9 atributos químicos para classificação:

- **RI**: Índice de Refração
- **Na**: Sódio
- **Mg**: Magnésio
- **Al**: Alumínio
- **Si**: Silício
- **K**: Potássio
- **Ca**: Cálcio
- **Ba**: Bário
- **Fe**: Ferro

---
## Tipo do Problema de Machine Learning

Esse programa é um problema de *aprendizado supervisionado de classificação multiclasse*, no qual o modelo aprende a associar as características químicas de uma amostra de vidro a uma das seis classes de vidros presentes no dataset
	
---

## Instalação e execução

### Pré-requisitos
- Python 3.7+
- pip ou conda

### Passo 1: Clonar/Baixar o Projeto
```bash
cd vidro-classificacao
```

### Passo 2: Instalar Dependências
```bash
pip install -r requirements.txt
```

### Passo 3: Preparar Dados
- Coloque o arquivo `glass.csv` na pasta `data/`
- O arquivo deve ter a seguinte estrutura:
```
RI,Na,Mg,Al,Si,K,Ca,Ba,Fe,Type
1.52101,13.64,4.65,1.95,71.99,0.16,8.01,1.63,0.36,1
...
```

### Passo 4: Executar a Aplicação
```bash
streamlit run app.py
```

A aplicação abrirá em `http://localhost:8501`

---

## Arquitetura do Código

### Módulos Principais

#### **data_loader.py**
Responsável pelo carregamento e processamento de dados:
- `DataProcessor`: Classe para gerenciar o ciclo de vida dos dados
  - `load_data()`: Carrega o dataset
  - `split_and_scale()`: Divide em treino/validação/teste e escala
  - `get_dataset_info()`: Retorna informações sobre o dataset

#### **models.py**
Treinamento e avaliação de modelos:
- `ModelTrainer`: Classe para gerenciar treinamento de modelos
  - `train_knn()`: Treina KNN
  - `train_svm()`: Treina SVM
  - `train_naive_bayes()`: Treina Naive Bayes
  - `evaluate_model()`: Avalia desempenho
  - `cross_validation()`: Executa validação cruzada
  - `get_roc_auc_scores()`: Calcula scores ROC/AUC

#### **utils.py**
Funções auxiliares e visualizações:
- `detect_outliers_iqr()`: Detecção de outliers
- `analyze_outliers()`: Análise de outliers
- `plot_pca()`: Visualização PCA
- `compare_models()`: Comparação de modelos
- `plot_model_comparison()`: Gráfico comparativo

### Aplicação Streamlit

**app.py**: Interface web interativa com 3 abas:

1. **Classificação**: Faz predições em tempo real
   - Entrada de parâmetros químicos
   - Resultados dos 3 modelos
   - Legenda de tipos

2. **Análise de Dados**: Exploração do dataset
   - Estatísticas gerais
   - Distribuição de classes
   - Estatísticas descritivas

3. **Desempenho**: Comparação de modelos
   - Métricas de acurácia
   - Gráficos comparativos
   - Relatórios detalhados

---

## Metodologia

### Divisão de Dados
- **Treino**: 60% dos dados
- **Validação**: 20% dos dados
- **Teste**: 20% dos dados

Utiliza `StratifiedKFold` para garantir distribuição equilibrada de classes.

### Preprocessamento
- **Escalonamento**: StandardScaler normaliza as features
- **Detecção de Outliers**: Método IQR (Interquartil Range)
- **Redução de Dimensionalidade**: PCA para visualização

### Modelos Implementados

#### **KNN (K-Nearest Neighbors)**
- K = 3 (otimizado via grid search)
- Métrica: Euclidiana
- Validação cruzada: 5-fold

#### **SVM (Support Vector Machine)**
- Kernel: RBF (Radial Basis Function)
- C = 10 (parâmetro de regularização)
- Probability: True (para predict_proba)
- Validação cruzada: 5-fold estratificada

#### **Naive Bayes (Gaussiano)**
- Assume distribuição normal das features
- Rápido e interpretável
- Validação cruzada: 5-fold estratificada

### Métricas de Avaliação
- **Acurácia**: Proporção de predições corretas
- **Matriz de Confusão**: Análise de erros por classe
- **ROC/AUC**: Curva ROC para análise One-vs-Rest
- **Classification Report**: Precisão, Recall, F1-Score

### Modelo Final Escolhido
**SVM (Support Vector Machine)**
- Kernel: RBF (Radial Basis Function)
- C = 10 (parâmetro de regularização)
- Probability: True (para predict_proba)
- Validação cruzada: 5-fold estratificada

---

## Resultados Esperados

Os modelos devem atingir:
- **Acurácia de Treino**: ~95-98%
- **Acurácia de Validação**: ~80-90%
- **Acurácia de Teste**: ~75-85%

*Os valores exatos dependem da configuração e aleatoriedade*

---

## Análise Detalhada

### Exploração de Dados
- Distribuição de classes em histogramas
- Scatter plots para detecção de outliers
- Análise estatística (média, desvio padrão, quartis)

### Validação Cruzada
Cada modelo é avaliado usando **Stratified K-Fold** (5 splits):
- Mantém proporção de classes em cada fold
- Fornece estimativa robusta de desempenho
- Reduz variância de estimação

### Curvas ROC/AUC
- Análise One-vs-Rest para classificação multiclasse
- Avalia trade-off entre Taxa de Verdadeiro Positivo e Falso Positivo
- AUC próximo a 1.0 indica excelente desempenho

---

## Tecnologias Utilizadas

- **Linguagem**: Python 3.7+
- **Machine Learning**: Scikit-learn
- **Visualização**: Matplotlib, Seaborn
- **Interface Web**: Streamlit
- **Processamento de Dados**: Pandas, NumPy
- **Serialização**: Joblib

---

## Como Usar a Aplicação

### 1. Classificação em Tempo Real
1. Abra a aba "Classificação"
2. Ajuste os valores dos parâmetros químicos na barra lateral
3. Clique em "Realizar Classificação"
4. Compare os resultados dos três modelos

### 2. Análise Exploratória
1. Vá para "Análise de Dados"
2. Visualize estatísticas e distribuição do dataset
3. Identifique padrões nas features

### 3. Avaliação de Desempenho
1. Acesse "Desempenho dos Modelos"
2. Compare acurácia em treino, validação e teste
3. Revise relatórios detalhados por modelo

---

## Salvando Modelos Treinados

Para salvar modelos treinados:

```python
import joblib
from src.models import ModelTrainer

trainer = ModelTrainer()
# ... treinar modelos ...

# Salvar
joblib.dump(trainer.models['knn'], 'model/knn_model.pkl')
joblib.dump(trainer.models['svm'], 'model/svm_model.pkl')
joblib.dump(trainer.models['naive_bayes'], 'model/nb_model.pkl')
```

---

## Observações Importantes

1. **Dataset**: Use `glass.csv` com exatamente as colunas especificadas
2. **Reproduzibilidade**: Use `random_state=42` para resultados consistentes
3. **Escalamento**: Sempre escale os dados antes de usar os modelos
4. **Cache**: A aplicação Streamlit usa cache para performance
5. **Validação**: Sempre faça predições com dados escalados corretamente

---

## Solução de Problemas

### Erro: "FileNotFoundError: glass.csv"
- Certifique-se de que `glass.csv` está na pasta `data/`

### Erro: "ModuleNotFoundError"
- Reinstale as dependências: `pip install -r requirements.txt`
- Verifique se o Python está no PATH

### Streamlit não abre
- Verifique porta 8501: `streamlit run app.py --server.port=8502`
- Limpe cache: `streamlit cache clear`

---
## Conclusão

Este estudo desenvolveu uma inteligência artificial que identifica tipos de vidro com base em sua composição química, visando acelerar a triagem na reciclagem industrial e apoiar investigações policiais. Utilizando uma base de dados com 214 amostras, o projeto testou os algoritmos KNN, SVM e Naive Bayes, identificando que o Sódio, o Magnésio e o Alumínio são os elementos mais importantes para a diferenciação. O modelo SVM apresentou o melhor desempenho, atingindo 72% de acerto geral, e o principal desafio apontado foi a falta de dados sobre vidros mais raros, o que limita a precisão do sistema nas categorias menos comuns.


---


## Link do APP Publicado:

**Link:** 


---


## Limitações
- Uma restrição mapeada na base de dados é a ausência total de amostras da Classe 4, limitando o escopo preditivo às demais categorias. 
- Dataset pequeno
- Generalização limitada
- Classes Mais Confundidas: As maiores taxas de erro ocorrem de maneira cruzada e sistêmica entre as fronteiras da Classe 1 e da Classe 2. 
- Gargalo da Classe 3: O vidro pertencente ao "Tipo 3" é frequentemente classificado de forma errônea como pertencente às Classes 1 e 2, gerando um baixíssimo índice de Recall. 
	- Causa Raiz: Esse fenômeno é justificado diretamente pelo desbalanceamento crítico mapeado na EDA. Como a máquina tem uma quantidade massiva de exemplos das classes 1 e 2 para estudar, ela herda um viés estatístico estrutural, tendendo a ignorar os padrões da classe minoritária (Tipo 3) nas zonas de alta sobreposição química. 
- Classifica apenas os seguintes tipos de vidro: 
  - Janela (Flutuação)
  - Janela (Não-Flutuação)
  - Veículo (Flutuação)
  - Veículo (Não-Flutuação)
  - Recipiente
  - Farol
  - Mesa

---


## Referências

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Glass Dataset](https://archive.ics.uci.edu/ml/datasets/glass+identification)

---

## Licença

Projeto educacional desenvolvido para fins acadêmicos.

---

**Última atualização**: Junho 2024


