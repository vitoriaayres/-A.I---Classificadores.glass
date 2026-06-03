
"""
Script de treinamento offline - Treina e salva modelos
Útil para gerar modelos que podem ser carregados depois
"""

import sys
import os
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataProcessor
from src.models import ModelTrainer

def main():
    """Função principal para treinamento offline"""
    
    print("=" * 60)
    print("TREINAMENTO OFFLINE - CLASSIFICAÇÃO DE VIDRO")
    print("=" * 60)
    
    print("\nCarregando dados...")
    data_path = "data/glass.csv"
    
    if not os.path.exists(data_path):
            print(f"Erro: Arquivo {data_path} não encontrado!")
            print("Certifique-se de colocar glass.csv na pasta 'data/'")
    return
    
    processor = DataProcessor(data_path)
    dataset = processor.load_data()
    print(f"Dataset carregado: {dataset.shape[0]} amostras, {dataset.shape[1]} colunas")
    
    print("\nProcessando dados...")
    data_split = processor.split_and_scale()
    print(f"Dados divididos e escalados")
    print(f"   - Treino: {data_split['X_train'].shape[0]} amostras")
    print(f"   - Validação: {data_split['X_val'].shape[0]} amostras")
    print(f"   - Teste: {data_split['X_test'].shape[0]} amostras")
    
    print("\nTreinando modelos...")
    trainer = ModelTrainer()
    
    print("   → Treinando KNN (K=3)...")
    knn = trainer.train_knn(
        data_split['X_train'], 
        data_split['y_train'], 
        k=3
    )
    print("     KNN treinado")
    
    print("   → Treinando SVM (RBF, C=10)...")
    svm = trainer.train_svm(
        data_split['X_train'], 
        data_split['y_train'], 
        kernel='rbf', 
        C=10
    )
    print("     SVM treinado")
    
    print("   → Treinando Naive Bayes...")
    nb = trainer.train_naive_bayes(
        data_split['X_train'], 
        data_split['y_train']
    )
    print("     Naive Bayes treinado")
    
    print("\nAvaliando modelos...")

    trainer.evaluate_model(
        knn,
        data_split['X_train'], data_split['y_train'],
        data_split['X_val'], data_split['y_val'],
        data_split['X_test'], data_split['y_test'],
        "KNN (K=3)"
    )
    
    trainer.evaluate_model(
        svm,
        data_split['X_train'], data_split['y_train'],
        data_split['X_val'], data_split['y_val'],
        data_split['X_test'], data_split['y_test'],
        "SVM (RBF, C=10)"
    )
    
    trainer.evaluate_model(
        nb,
        data_split['X_train'], data_split['y_train'],
        data_split['X_val'], data_split['y_val'],
        data_split['X_test'], data_split['y_test'],
        "Naive Bayes"
    )
    
    print("\n" + "=" * 60)
    print("RESULTADOS DOS MODELOS")
    print("=" * 60)
    
    results = trainer.get_summary_results()
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Acurácia Treino: {metrics['acc_train']:.4f}")
        print(f"  Acurácia Validação: {metrics['acc_val']:.4f}")
        print(f"  Acurácia Teste: {metrics['acc_test']:.4f}")
    
        print("\nValidação Cruzada (5-fold):")
    
    cv_results = trainer.cross_validation(
        knn, 
        data_split['X_train'], 
        data_split['y_train']
    )
    print(f"  KNN: {cv_results['mean']:.4f} (+/- {cv_results['ci']:.4f})")
    
    cv_results = trainer.cross_validation(
        svm, 
        data_split['X_train'], 
        data_split['y_train']
    )
    print(f"  SVM: {cv_results['mean']:.4f} (+/- {cv_results['ci']:.4f})")
    
    cv_results = trainer.cross_validation(
        nb, 
        data_split['X_train'], 
        data_split['y_train']
    )
    print(f"  Naive Bayes: {cv_results['mean']:.4f} (+/- {cv_results['ci']:.4f})")
    
    print("\nSalvando modelos...")
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(knn, 'model/knn_model.pkl')
    print("  KNN salvo em model/knn_model.pkl")
    
    joblib.dump(svm, 'model/svm_model.pkl')
    print("  SVM salvo em model/svm_model.pkl")
    
    joblib.dump(nb, 'model/nb_model.pkl')
    print("  Naive Bayes salvo em model/nb_model.pkl")
    
    joblib.dump(data_split['scaler'], 'model/scaler.pkl')
    print("  Scaler salvo em model/scaler.pkl")
    
    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print("\nVocê pode agora executar: streamlit run app.py")


if __name__ == "__main__":
    main()
