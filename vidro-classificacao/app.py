"""
Aplicação Streamlit para Classificação de Vidro
Autores: Vitoria Ayres (2086138) e Leticia Ribeiro (2034293)
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataProcessor
from models import ModelTrainer
from utils import compare_models, plot_model_comparison

st.set_page_config(
    page_title="Classificação de Vidro Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3em; 
        background-color: #007bff; 
        color: white;
        font-weight: bold;
    }
    .predict-card { 
        padding: 20px; 
        border-radius: 10px; 
        background-color: white; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        margin-bottom: 10px; 
        border-left: 5px solid #007bff; 
    }
    .type-id { 
        font-size: 24px; 
        font-weight: bold; 
        color: #007bff; 
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title('Classificação de Vidro')
st.write("Insira os parâmetros químicos abaixo para identificar a categoria do vidro.")

with st.sidebar:
    st.header("Configuração")
    option = st.radio(
        "Selecione uma opção:",
        ["Classificação", "Análise de Dados", "Desempenho dos Modelos"]
    )

@st.cache_resource
def load_data_and_train():
    """Carrega dados e treina modelos"""
    data_path = "data/glass.csv"
    
    if not os.path.exists(data_path):
        st.error(f"Arquivo não encontrado: {data_path}")
        st.info("Coloque o arquivo 'glass.csv' na pasta 'data/'")
        return None, None, None, None
    
    processor = DataProcessor(data_path)
    processor.load_data()
    
    data_split = processor.split_and_scale()
    X_train = data_split['X_train']
    X_val = data_split['X_val']
    X_test = data_split['X_test']
    y_train = data_split['y_train']
    y_val = data_split['y_val']
    y_test = data_split['y_test']
    scaler = data_split['scaler']
    
    # Treina modelos
    trainer = ModelTrainer()
    
    knn = trainer.train_knn(X_train, y_train, k=3)
    svm = trainer.train_svm(X_train, y_train, kernel='rbf', C=10)
    nb = trainer.train_naive_bayes(X_train, y_train)
    
    # Avalia modelos
    trainer.evaluate_model(knn, X_train, y_train, X_val, y_val, X_test, y_test, "KNN (K=3)")
    trainer.evaluate_model(svm, X_train, y_train, X_val, y_val, X_test, y_test, "SVM (RBF, C=10)")
    trainer.evaluate_model(nb, X_train, y_train, X_val, y_val, X_test, y_test, "Naive Bayes")
    
    return {
        'processor': processor,
        'trainer': trainer,
        'knn': knn,
        'svm': svm,
        'nb': nb,
        'scaler': scaler,
        'X': processor.X,
        'y': processor.y
    }

models_data = load_data_and_train()

if models_data is None:
    st.stop()

processor = models_data['processor']
trainer = models_data['trainer']
knn = models_data['knn']
svm = models_data['svm']
nb = models_data['nb']
scaler = models_data['scaler']
X = models_data['X']
y = models_data['y']

if option == "Classificação":
    st.header("Classificação de Vidro")
    
    with st.sidebar:
        st.header("Parâmetros Químicos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ri = st.number_input('RI', value=float(X['RI'].mean()), step=0.01)
            na = st.number_input('Na', value=float(X['Na'].mean()), step=0.01)
            mg = st.number_input('Mg', value=float(X['Mg'].mean()), step=0.01)
            al = st.number_input('Al', value=float(X['Al'].mean()), step=0.01)
            si = st.number_input('Si', value=float(X['Si'].mean()), step=0.01)
        
        with col2:
            k = st.number_input('K', value=float(X['K'].mean()), step=0.01)
            ca = st.number_input('Ca', value=float(X['Ca'].mean()), step=0.01)
            ba = st.number_input('Ba', value=float(X['Ba'].mean()), step=0.01)
            fe = st.number_input('Fe', value=float(X['Fe'].mean()), step=0.01)
        
        btn_predict = st.button("Realizar Classificação", use_container_width=True)
    
    st.markdown("""
    ### Legenda de Tipos
    - **1**: Janela (Flutuação)
    - **2**: Janela (Não-Flutuação)
    - **3**: Veículo
    - **5**: Recipiente
    - **6**: Farol
    - **7**: Mesa
    """)
    
    if btn_predict:
        input_data = pd.DataFrame(
            [[ri, na, mg, al, si, k, ca, ba, fe]],
            columns=X.columns
        )
        scaled_input = scaler.transform(input_data)
        
        st.subheader("Resultados da Predição")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            res = knn.predict(scaled_input)[0]
            st.markdown(f'<div class="predict-card"><b>KNN</b><br><span class="type-id">Tipo {int(res)}</span></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            res = svm.predict(scaled_input)[0]
            st.markdown(f'<div class="predict-card"><b>SVM</b><br><span class="type-id">Tipo {int(res)}</span></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            res = nb.predict(scaled_input)[0]
            st.markdown(f'<div class="predict-card"><b>Naive Bayes</b><br><span class="type-id">Tipo {int(res)}</span></div>', 
                       unsafe_allow_html=True)
        
        st.info("Compare os resultados dos três modelos para uma classificação mais robusta!")

elif option == "Análise de Dados":
    st.header("Análise do Dataset")
    
    dataset = processor.dataset
    info = processor.get_dataset_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Amostras", info['shape'][0])
    with col2:
        st.metric("Número de Features", info['shape'][1] - 1)
    with col3:
        st.metric("Número de Classes", len(info['classes']))
    
    st.subheader("Distribuição de Classes")
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(dataset['Type'].value_counts().sort_index())
    
    with col2:
        st.write(dataset['Type'].value_counts())
    
    st.subheader("Estatísticas Descritivas")
    st.dataframe(dataset.describe(), use_container_width=True)

elif option == "Desempenho dos Modelos":
    st.header("Análise de Desempenho dos Modelos")
    
    results = trainer.get_summary_results()
    df_comparison = compare_models(results)
    
    st.subheader("Comparação de Acurácia")
    st.dataframe(df_comparison, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df_comparison.set_index('Modelo')[['Treino', 'Validação', 'Teste']]
    df_plot.plot(kind='barh', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xlabel("Acurácia")
    ax.set_title("Comparação de Acurácia dos Modelos")
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc='lower right')
    st.pyplot(fig)
    
    st.subheader("Relatório Detalhado por Modelo")
    for model_name, model_results in results.items():
        with st.expander(f"{model_name}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Acurácia Treino", f"{model_results['acc_train']:.4f}")
            with col2:
                st.metric("Acurácia Validação", f"{model_results['acc_val']:.4f}")
            with col3:
                st.metric("Acurácia Teste", f"{model_results['acc_test']:.4f}")

st.markdown("""
---
**Projeto de Classificação de Vidro** | Desenvolvido por: Vitoria Ayres e Leticia Ribeiro
""")
