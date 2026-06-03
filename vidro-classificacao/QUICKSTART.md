# GUIA RÁPIDO - COMO USAR

## Começar em 5 Minutos

### 1. Clonar/Baixar o Projeto
```bash
cd vidro-classificacao
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Preparar Dados
- Baixe `glass.csv` em: https://archive.ics.uci.edu/ml/datasets/glass+identification
- Coloque em: `data/glass.csv`

### 4. (Opcional) Treinar Modelos Offline
```bash
python train.py
```

### 5️⃣ Executar Aplicação
```bash
streamlit run app.py
```

A aplicação abrirá em: **http://localhost:8501**

---

## Uso da Aplicação Streamlit

### Aba: Classificação
1. Abra a barra lateral esquerda
2. Ajuste os parâmetros químicos:
   - RI, Na, Mg, Al, Si, K, Ca, Ba, Fe
3. Clique no botão **Realizar Classificação**
4. Veja os resultados dos 3 modelos

**Dica**: Use os valores médios como ponto de partida

### Aba: Análise de Dados
- Visualize estatísticas do dataset
- Veja a distribuição de classes
- Analise as características do vidro

### Aba: Desempenho dos Modelos
- Compare acurácia entre modelos
- Veja gráficos de performance
- Leia relatórios detalhados por modelo

---

## Estrutura do Código

```
vidro-classificacao/
├── app.py              ← Executar isto para iniciar
├── train.py            ← Treinar modelos offline
├── requirements.txt    ← Dependências
├── README.md           ← Documentação completa
│
├── src/
│   ├── data_loader.py  ← Processamento de dados
│   ├── models.py       ← Treinamento de modelos
│   ├── utils.py        ← Funções auxiliares
│   └── __init__.py
│
├── data/               ← Coloque glass.csv aqui
├── model/              ← Modelos salvos
├── reports/            ← Relatórios
└── notebooks/          ← Análise exploratória
```

---

## Solução de Problemas

### "glass.csv not found"
- Coloque o arquivo em `data/glass.csv`

### "ModuleNotFoundError"
- Execute `pip install -r requirements.txt`

### Port 8501 already in use
- Execute `streamlit run app.py --server.port=8502`

### Muito lento?
- Limpe o cache: `streamlit cache clear`

---

## Exemplos de Valores

### Vidro de Janela (Tipo 1):
- RI: 1.52, Na: 13.64, Mg: 4.65, Al: 1.95, Si: 71.99, K: 0.16, Ca: 8.01, Ba: 1.63, Fe: 0.36

### Vidro de Veículo (Tipo 3):
- RI: 1.52, Na: 13.27, Mg: 4.00, Al: 1.76, Si: 72.03, K: 0.65, Ca: 8.03, Ba: 1.66, Fe: 0.36

---

## Modelos Usados

- **KNN (K-Nearest Neighbors)**: K=3, Euclidiana
- **SVM (Support Vector Machine)**: Kernel RBF, C=10
- **Naive Bayes**: Gaussiano

Todos os modelos são treinados com:
- Validação cruzada 5-fold
- Escalonamento StandardScaler
- Divisão estratificada dos dados

---

## Links Úteis

- [UCI Glass Dataset](https://archive.ics.uci.edu/ml/datasets/glass+identification)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## Observações

- Sempre escale os dados antes de fazer predições
- Use `random_state=42` para reproduzibilidade
- Validação cruzada garante robustez
- Compare os 3 modelos para decisões melhores

---

**Desenvolvido por**: Vitoria Ayres & Leticia Ribeiro
**Data**: Junho 2024
