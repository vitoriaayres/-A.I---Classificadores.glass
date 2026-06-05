"""
Aplicação Streamlit para Classificação de Vidro
Autores: Vitoria Ayres (2086138) e Leticia Ribeiro (2034293)
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataProcessor
from models import ModelTrainer

# ═════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO STREAMLIT
# ═════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Classificação de Vidro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .section-title {
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        font-size: 26px;
        font-weight: bold;
        color: #667eea;
        margin: 20px 0;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 20px 0;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .stats-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .feature-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #333;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ═════════════════════════════════════════════════════════════════

@st.cache_data
def load_dataset(path):
    """Carrega o dataset com cache"""
    if not os.path.exists(path):
        return None
    processor = DataProcessor(path)
    processor.load_data()
    return processor

@st.cache_resource
def train_models(_processor):
    """Treina os modelos com cache de resource"""
    data_split = _processor.split_and_scale()
    trainer = ModelTrainer()
    
    # Treina os três modelos
    knn = trainer.train_knn(data_split['X_train'], data_split['y_train'], k=3)
    svm = trainer.train_svm(data_split['X_train'], data_split['y_train'], kernel='rbf', C=10)
    nb = trainer.train_naive_bayes(data_split['X_train'], data_split['y_train'])
    
    # Avalia os modelos
    trainer.evaluate_model(knn, data_split['X_train'], data_split['y_train'], 
                          data_split['X_val'], data_split['y_val'], 
                          data_split['X_test'], data_split['y_test'], "KNN (K=3)")
    trainer.evaluate_model(svm, data_split['X_train'], data_split['y_train'], 
                          data_split['X_val'], data_split['y_val'], 
                          data_split['X_test'], data_split['y_test'], "SVM (RBF, C=10)")
    trainer.evaluate_model(nb, data_split['X_train'], data_split['y_train'], 
                          data_split['X_val'], data_split['y_val'], 
                          data_split['X_test'], data_split['y_test'], "Naive Bayes")
    
    return {
        'trainer': trainer,
        'knn': knn,
        'svm': svm,
        'nb': nb,
        'scaler': data_split['scaler'],
        'data_split': data_split
    }

# ═════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ═════════════════════════════════════════════════════════════════

st.title('🔬 Classificação de Vidro')
st.markdown("Sistema inteligente de classificação de tipos de vidro usando Machine Learning")

DATA_PATH = "data/glass.csv"
processor = load_dataset(DATA_PATH)

if processor is None:
    st.error(f"❌ Arquivo não encontrado: {DATA_PATH}")
    st.info("📁 Coloque o arquivo 'glass.csv' na pasta 'data/' e recarregue a página.")
    st.stop()

# ═════════════════════════════════════════════════════════════════
# SIDEBAR - NAVEGAÇÃO
# ═════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("---")
    st.header("📊 Navegação")
    page = st.radio("Escolha uma seção:", 
                    ["🏠 Inicio", "🔮 Classificar", "📈 Análise", "🎯 Desempenho"])
    st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# PÁGINA: INICIO
# ═════════════════════════════════════════════════════════════════

if page == "🏠 Inicio":
    st.markdown('<div class="header-title">📊 Bem-vindo ao Sistema</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    dataset = processor.dataset
    info = processor.get_dataset_info()
    
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <h3>📦 Amostras</h3>
            <h2>{info['shape'][0]:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stats-box">
            <h3>🔬 Features</h3>
            <h2>{info['shape'][1] - 1}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stats-box">
            <h3>🎨 Classes</h3>
            <h2>{len(info['classes'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<h3 class="section-title">📋 Tipos de Vidro</h3>', unsafe_allow_html=True)
    
    glass_types = {
        1: ("Janela (Flutuação)", "🪟"),
        2: ("Janela (Não-Flutuação)", "🪟"),
        3: ("Veículo", "🚗"),
        5: ("Recipiente", "🥃"),
        6: ("Farol", "💡"),
        7: ("Mesa", "🪑")
    }
    
    cols = st.columns(3)
    for idx, (type_id, (type_name, emoji)) in enumerate(glass_types.items()):
        with cols[idx % 3]:
            count = len(dataset[dataset['Type'] == type_id])
            percentage = (count / len(dataset)) * 100
            st.markdown(f"""
            <div class="info-card">
                <h4>{emoji} Tipo {type_id}</h4>
                <p>{type_name}</p>
                <h3>{count} amostras ({percentage:.1f}%)</h3>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<h3 class="section-title">📈 Distribuição</h3>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    dist = dataset['Type'].value_counts().sort_index()
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b']
    ax.bar(dist.index, dist.values, color=colors[:len(dist)], edgecolor='black', linewidth=2)
    ax.set_xlabel("Tipo de Vidro", fontsize=13, fontweight='bold')
    ax.set_ylabel("Contagem", fontsize=13, fontweight='bold')
    ax.set_title("Distribuição de Amostras", fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    st.pyplot(fig)

# ═════════════════════════════════════════════════════════════════
# PÁGINA: CLASSIFICAÇÃO
# ═════════════════════════════════════════════════════════════════

elif page == "🔮 Classificar":
    st.markdown('<div class="header-title">🔮 Classificar Amostra</div>', unsafe_allow_html=True)
    st.write("Insira os parâmetros químicos para classificar o vidro")
    
    with st.spinner("⏳ Carregando modelos..."):
        models_data = train_models(processor)
    
    X = processor.X
    defaults = {col: float(X[col].median()) for col in X.columns}
    
    st.markdown('<h3 class="section-title">🧪 Parâmetros</h3>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    inputs = {}
    
    for idx, col in enumerate(X.columns):
        with cols[idx % 3]:
            inputs[col] = st.number_input(
                f"{col} ({X[col].min():.2f} - {X[col].max():.2f})",
                min_value=float(X[col].min()),
                max_value=float(X[col].max()),
                value=defaults[col],
                step=0.01,
                format="%.4f"
            )
    
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn2:
        classify_btn = st.button("🚀 Classificar", use_container_width=True)
    
    if classify_btn:
        input_df = pd.DataFrame([inputs])
        scaled_input = models_data['scaler'].transform(input_df)
        
        pred_knn = models_data['knn'].predict(scaled_input)[0]
        pred_svm = models_data['svm'].predict(scaled_input)[0]
        pred_nb = models_data['nb'].predict(scaled_input)[0]
        
        st.success("✅ Classificação concluída!")
        st.markdown('<h3 class="section-title">📊 Resultados</h3>', unsafe_allow_html=True)
        
        res_cols = st.columns(3)
        
        with res_cols[0]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; text-align: center;">
                <h3>KNN</h3>
                <h1 style="font-size: 3.5em;">Tipo {int(pred_knn)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with res_cols[1]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 15px; text-align: center;">
                <h3>SVM</h3>
                <h1 style="font-size: 3.5em;">Tipo {int(pred_svm)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with res_cols[2]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 25px; border-radius: 15px; text-align: center;">
                <h3>Naive Bayes</h3>
                <h1 style="font-size: 3.5em;">Tipo {int(pred_nb)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        votes = [int(pred_knn), int(pred_svm), int(pred_nb)]
        majority_class = max(set(votes), key=votes.count)
        consensus_count = votes.count(majority_class)
        
        glass_names = {1: "Janela (Flutuação)", 2: "Janela (Não-Flutuação)", 3: "Veículo", 
                      5: "Recipiente", 6: "Farol", 7: "Mesa"}
        
        if consensus_count == 3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 20px; border-radius: 12px; text-align: center;">
                <h2>🎯 CONSENSO: Tipo {majority_class}</h2>
                <p>{glass_names.get(int(majority_class))}</p>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# PÁGINA: ANÁLISE
# ═════════════════════════════════════════════════════════════════

elif page == "📈 Análise":
    st.markdown('<div class="header-title">📈 Análise do Dataset</div>', unsafe_allow_html=True)
    
    dataset = processor.dataset
    
    tab1, tab2, tab3 = st.tabs(["📊 Estatísticas", "📈 Distribuição", "📋 Tabela"])
    
    with tab1:
        st.markdown('<h3 class="section-title">Estatísticas Descritivas</h3>', unsafe_allow_html=True)
        st.dataframe(dataset.describe(), use_container_width=True)
    
    with tab2:
        st.markdown('<h3 class="section-title">Distribuição de Classes</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        dist = dataset['Type'].value_counts().sort_index()
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b']
        ax.bar(dist.index, dist.values, color=colors[:len(dist)], edgecolor='black', linewidth=2)
        ax.set_xlabel("Tipo de Vidro", fontsize=13, fontweight='bold')
        ax.set_ylabel("Contagem", fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with tab3:
        st.markdown('<h3 class="section-title">Tabela Completa</h3>', unsafe_allow_html=True)
        st.dataframe(dataset, use_container_width=True)
        csv = dataset.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="glass_dataset.csv", mime="text/csv")

# ═════════════════════════════════════════════════════════════════
# PÁGINA: DESEMPENHO
# ═════════════════════════════════════════════════════════════════

elif page == "🎯 Desempenho":
    st.markdown('<div class="header-title">🎯 Desempenho dos Modelos</div>', unsafe_allow_html=True)
    
    with st.spinner("⏳ Treinando modelos..."):
        models_data = train_models(processor)
        trainer = models_data['trainer']
    
    st.success("✅ Modelos treinados!")
    
    st.markdown('<h3 class="section-title">📊 Comparação de Acurácia</h3>', unsafe_allow_html=True)
    
    results = trainer.results
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Modelo': model_name,
            'Treino': f"{metrics['acc_train']:.4f}",
            'Validação': f"{metrics['acc_val']:.4f}",
            'Teste': f"{metrics['acc_test']:.4f}"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    st.markdown("---")
    st.markdown('<h3 class="section-title">📈 Gráfico Comparativo</h3>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = [m['Modelo'] for m in comparison_data]
    train_acc = [float(m['Treino']) for m in comparison_data]
    val_acc = [float(m['Validação']) for m in comparison_data]
    test_acc = [float(m['Teste']) for m in comparison_data]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax.bar(x - width, train_acc, width, label='Treino', color='#667eea')
    ax.bar(x, val_acc, width, label='Validação', color='#764ba2')
    ax.bar(x + width, test_acc, width, label='Teste', color='#f093fb')
    
    ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acurácia', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Acurácia', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    st.pyplot(fig)
    
    st.markdown("---")
    st.markdown('<h3 class="section-title">🔍 Detalhes por Modelo</h3>', unsafe_allow_html=True)
    
    for model_name, metrics in results.items():
        with st.expander(f"📌 {model_name}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Treino", f"{metrics['acc_train']:.4f}")
            with col2:
                st.metric("Validação", f"{metrics['acc_val']:.4f}")
            with col3:
                st.metric("Teste", f"{metrics['acc_test']:.4f}")

st.markdown("---")
st.markdown("<center><small>🔬 Classificação de Vidro | Vitoria Ayres & Leticia Ribeiro</small></center>", unsafe_allow_html=True)
