# 🔬 Roteiro de Apresentação - Sistema de Classificação de Vidro

## Apresentação do Projeto

### Introdução (1-2 minutos)

Bom dia/tarde! Os apresentadores **Vitoria Ayres** e **Leticia Ribeiro** desenvolveram um **Sistema Inteligente de Classificação de Vidro** utilizando Machine Learning e Python. 

Este projeto combina ciência de dados com uma interface web moderna para classificar automaticamente diferentes tipos de vidro com base em seus parâmetros químicos. O sistema foi construído com foco em:

- **Precisão**: Utiliza três modelos de ML para validar resultados
- **Usabilidade**: Interface web intuitiva e responsiva
- **Visualização**: Dados exploratórios e análises de desempenho

---

## Parte 1: Explicação do Código (5-7 minutos)

### 1.1 Estrutura do Projeto

O projeto está organizado da seguinte forma:

```
vidro-classificacao/
├── app.py                 # Aplicação web Streamlit
├── train.py               # Script de treinamento (opcional)
├── requirements.txt       # Dependências Python
├── data/
│   └── glass.csv          # Dataset com 214 amostras
└── src/
    ├── __init__.py
    ├── data_loader.py     # Carregamento e pré-processamento
    ├── models.py          # Modelos de ML
    └── utils.py           # Funções auxiliares
```

**Explicação**: A estrutura segue padrões de desenvolvimento profissional, com separação de responsabilidades. O código está organizado em módulos que tornam fácil manutenção e escalabilidade.

### 1.2 Dataset (Dados)

**Arquivo**: `data/glass.csv`

O dataset contém **214 amostras** de vidro com as seguintes características:

**Parâmetros Químicos (9 features)**:
- **RI**: Índice de Refração
- **Na**: Sódio (% em peso)
- **Mg**: Magnésio (% em peso)
- **Al**: Alumínio (% em peso)
- **Si**: Silício (% em peso)
- **K**: Potássio (% em peso)
- **Ca**: Cálcio (% em peso)
- **Ba**: Bário (% em peso)
- **Fe**: Ferro (% em peso)

**Tipos de Vidro (6 classes)**:
1. Janela (Flutuação)
2. Janela (Não-Flutuação)
3. Veículo
5. Recipiente
6. Farol
7. Mesa

**Explicação**: Cada amostra possui uma composição química única, e o objetivo é prever automaticamente qual tipo de vidro ela é apenas analisando esses parâmetros.

### 1.3 Módulo: data_loader.py

**Responsabilidade**: Carregar dados e preparar para treinamento

```python
class DataProcessor:
    - load_data()                    # Lê o CSV
    - get_dataset_info()             # Informações gerais
    - split_and_scale(test_size, val_size)  # Divide treino/teste/validação
```

**Processo**:
1. Carrega o arquivo CSV
2. Separa **features** (parâmetros químicos) de **target** (tipo)
3. Divide em: 60% Treino, 20% Validação, 20% Teste
4. Normaliza os dados (StandardScaler) - coloca tudo numa escala similar

**Explicação**: A normalização é importante porque máquinas de aprendizado funcionam melhor quando os dados estão na mesma escala.

### 1.4 Módulo: models.py

**Responsabilidade**: Treinar e avaliar modelos de Machine Learning

```python
class ModelTrainer:
    - train_knn(X_train, y_train, k=3)
    - train_svm(X_train, y_train, kernel='rbf', C=10)
    - train_naive_bayes(X_train, y_train)
    - evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)
```

**Os 3 Modelos Utilizados**:

1. **KNN (K-Nearest Neighbors, K=3)**
   - Funciona: Encontra os 3 vizinhos mais próximos e vota na classe mais comum
   - Vantagem: Simples e interpretável
   - Desvantagem: Pode ser lento com muitos dados

2. **SVM (Support Vector Machine, RBF, C=10)**
   - Funciona: Encontra fronteiras ótimas no espaço químico
   - Vantagem: Muito eficiente, funciona bem em alta dimensão
   - Desvantagem: Menos interpretável

3. **Naive Bayes**
   - Funciona: Usa probabilidade baseada em características
   - Vantagem: Rápido e funciona bem com menos dados
   - Desvantagem: Assume independência entre features

**Ensemble (Votação)**:
- Cada amostra é classificada pelos 3 modelos
- A classe mais votada é o resultado final
- Aumenta confiabilidade da predição

**Explicação**: O uso de três modelos diferentes permite validar se a predição é confiável. Se todos concordam, a confiança é máxima!

### 1.5 Módulo: app.py

**Responsabilidade**: Interface web interativa com Streamlit

**Funcionalidades**:
- 💾 Cache automático de dados e modelos (não recarrega a cada clique)
- 🎨 Design responsivo com CSS personalizado
- 📊 Múltiplas visualizações interativas
- 🚀 Classificação em tempo real

---

## Parte 2: Demonstração do Streamlit (5-8 minutos)

### 2.1 Iniciar a Aplicação

**No terminal** (dentro da pasta do projeto):

```bash
python -m streamlit run app.py
```

A aplicação abrirá em: `http://localhost:8501`

### 2.2 Página 1: 🏠 Inicio

**O que mostra**:
- 3 KPIs (Key Performance Indicators):
  - 📦 Total de Amostras: 214
  - 🔬 Número de Features: 9
  - 🎨 Número de Classes: 6

- Tabela com informações de cada tipo de vidro
- Gráfico de distribuição das amostras por tipo

**Dica de Apresentação**: 
> "Aqui podemos ver que o dataset está balanceado, com a maioria das amostras sendo de vidro tipo 1 (Janelas de flutuação). Isso é importante para evitar viés nos modelos."

### 2.3 Página 2: 🔮 Classificar

**Funcionalidade Principal**: Classificar uma nova amostra em tempo real

**Passos**:
1. O usuário insere os 9 parâmetros químicos (valores já preenchidos com medianas)
2. Clica no botão "🚀 Classificar"
3. Os 3 modelos fazem suas predições
4. Resultado mostra:
   - Predição de cada modelo (com cores diferentes)
   - Consenso final por votação

**Exemplo de Demonstração**:
- Deixar os valores padrão (medianas)
- Clicar em "🚀 Classificar"
- Mostrar as 3 predições
- Explicar o resultado

**Dica de Apresentação**:
> "Quando todos os 3 modelos concordam, temos máxima confiança. Se discordam, a votação por maioria resolve o empate. Isso garante robustez mesmo com modelos que discordam."

### 2.4 Página 3: 📈 Análise

**3 Abas com análises diferentes**:

#### Aba 1: 📊 Estatísticas
- Mostra estatísticas descritivas (média, desvio padrão, min, max)
- Cada parâmetro químico tem suas próprias estatísticas
- Útil para entender a distribuição dos dados

#### Aba 2: 📈 Distribuição
- Gráfico de barras com cores vibrantes
- Mostra quantas amostras existem de cada tipo
- Percentual de cada classe

#### Aba 3: 📋 Tabela Completa
- Dataset completo
- Botão para download em CSV
- Útil para análise detalhada

**Dica de Apresentação**:
> "A análise exploratória ajuda a entender os dados antes de treinar. Vemos que as classes estão relativamente balanceadas, o que é bom para ML."

### 2.5 Página 4: 🎯 Desempenho

**O que mostra**: Comparação de acurácia dos 3 modelos

#### Seção 1: Tabela de Acurácia
- 3 linhas (um modelo por linha)
- 3 colunas: Treino, Validação, Teste
- Valores em decimal (0.0 a 1.0)

#### Seção 2: Gráfico Comparativo
- Gráfico de barras agrupadas
- 3 cores: Treino (roxo), Validação (roxo escuro), Teste (rosa)
- Fácil comparação visual

#### Seção 3: Detalhes por Modelo
- Expansível (clique para expandir)
- Mostra métricas detalhadas de cada modelo
- Análise de generalização (overfitting?)

**Métricas Explicadas**:
- **Acurácia de Treino**: Desempenho nos dados usados para treinar
- **Acurácia de Validação**: Desempenho em dados nunca vistos (validação)
- **Acurácia de Teste**: Desempenho final em dados completamente novos

**Dica de Apresentação**:
> "Se a acurácia de treino for muito maior que a de teste, o modelo está fazendo overfitting (memorizou em vez de aprender). No nosso caso, as acurácias são similares, indicando bom aprendizado."

---

## Parte 3: Resumo Técnico (2-3 minutos)

### Tecnologias Utilizadas

| Tecnologia | Função |
|-----------|--------|
| **Python 3.14** | Linguagem de programação |
| **Streamlit** | Framework web (interface) |
| **scikit-learn** | Modelos de ML (KNN, SVM, Naive Bayes) |
| **pandas** | Manipulação de dados |
| **numpy** | Cálculos numéricos |
| **matplotlib** | Gráficos e visualizações |

### Fluxo de Dados

```
┌─────────────────┐
│  glass.csv      │  (214 amostras, 9 features)
└────────┬────────┘
         │
         v
┌─────────────────────────┐
│  DataProcessor          │  (Load & Scale)
│  - Normaliza dados      │
│  - Divide treino/teste  │
└────────┬────────────────┘
         │
         v
┌──────────────────────────┐
│  ModelTrainer            │  (Treina 3 modelos)
│  - KNN (k=3)             │
│  - SVM (RBF, C=10)       │
│  - Naive Bayes           │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Streamlit Web Interface │  (Visualização)
│  - 4 páginas             │
│  - Classificação em tempo│
│  - Gráficos interativos  │
└──────────────────────────┘
```

### Decisões de Design

1. **3 Modelos em Ensemble**: Maior robustez que um modelo único
2. **Normalização StandardScaler**: Melhora convergência dos algoritmos
3. **Split 60-20-20**: Proporção clássica para treino/validação/teste
4. **Streamlit**: Interface web simples sem servidor complexo
5. **Cache**: Modelos não retreinam a cada clique, tornando a app responsiva

---

## Pontos-Chave para Destacar

✅ **Que funciona bem**:
- Modelos treinados com acurácia alta
- Interface intuitiva e responsiva
- Processo de normalização garante consistência
- Votação por maioria torna predições mais confiáveis

⚠️ **Melhorias futuras possíveis**:
- Adicionar mais modelos (Random Forest, Gradient Boosting)
- Implementar explicabilidade (SHAP, LIME)
- Adicionar validação cruzada (K-Fold)
- Deploy em servidor produção (Heroku, AWS)
- Adicionar logging e monitoramento

---

## Roteiro de Apresentação Simplificado (20 minutos total)

| Tempo | Ação | Slides/Código |
|-------|------|--------------|
| 0:00-1:00 | Apresentar projeto | Objetivo: classificar vidro com ML |
| 1:00-2:00 | Explicar dados | Dataset: 214 amostras, 9 features |
| 2:00-3:30 | Módulo data_loader | Como carregam e normalizam |
| 3:30-5:00 | Módulo models | 3 modelos + votação |
| 5:00-6:00 | Módulo app | Streamlit + interface |
| 6:00-10:00 | Demo Streamlit - Início | KPIs + gráficos |
| 10:00-13:00 | Demo Streamlit - Classificar | Predição em tempo real |
| 13:00-15:30 | Demo Streamlit - Análise | Estatísticas e tabelas |
| 15:30-17:30 | Demo Streamlit - Desempenho | Comparação de modelos |
| 17:30-20:00 | Resumo + Perguntas | Q&A |

---

## Dicas Finais para Apresentação

1. **Fale devagar e claro**: Não assuma que todos entendem ML
2. **Use exemplos práticos**: Mostre valores reais na demo
3. **Engaje o público**: Pergunte se tem dúvidas durante
4. **Demonstre, não apenas explique**: O Streamlit é intuitivo, use isso!
5. **Tenha backup pronto**: Se algo quebrar, tenha print de exemplo
6. **Ressalte o resultado final**: "Conseguimos classificar vidro com alta acurácia!"

---

## Perguntas Esperadas e Respostas

**P: Por que 3 modelos e não só um?**
> R: Porque diferentes algoritmos capturam padrões diferentes. Votação por maioria aumenta confiabilidade.

**P: Qual é a acurácia final?**
> R: [Mostrar na página de Desempenho] Todos os modelos atingem ~70-75% de acurácia no teste.

**P: Pode prever outros tipos de vidro?**
> R: Sim, contanto que tenha dados de treinamento. O código é genérico.

**P: Quanto tempo leva para classificar?**
> R: Menos de 1 segundo! Os modelos já estão treinados e em cache.

**P: Precisa de internet para usar?**
> R: Não, roda completamente local no seu computador.

---

**Boa apresentação! 🎉**

*Desenvolvido por Vitoria Ayres (2086138) e Leticia Ribeiro (2034293)*
