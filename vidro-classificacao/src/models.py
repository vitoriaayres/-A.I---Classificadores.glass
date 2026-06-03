# -*- coding: utf-8 -*-
"""
Módulo para treinamento e avaliação de modelos de classificação de vidro
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


class ModelTrainer:
    """Classe responsável pelo treinamento e avaliação de modelos"""
    
    def __init__(self):
        """Inicializa o treinador de modelos"""
        self.models = {}
        self.results = {}
        
    def train_knn(self, X_train, y_train, k=3):
        """
        Treina modelo KNN
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            k (int): Número de vizinhos
            
        Returns:
            model: Modelo KNN treinado
        """
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        self.models['knn'] = model
        return model
    
    def train_svm(self, X_train, y_train, kernel='rbf', C=10):
        """
        Treina modelo SVM
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            kernel (str): Tipo de kernel
            C (float): Parâmetro de regularização
            
        Returns:
            model: Modelo SVM treinado
        """
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        model.fit(X_train, y_train)
        self.models['svm'] = model
        return model
    
    def train_naive_bayes(self, X_train, y_train):
        """
        Treina modelo Naive Bayes
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            
        Returns:
            model: Modelo Naive Bayes treinado
        """
        model = GaussianNB()
        model.fit(X_train, y_train)
        self.models['naive_bayes'] = model
        return model
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
        """
        Avalia um modelo em treino, validação e teste
        
        Args:
            model: Modelo a avaliar
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validação
            X_test, y_test: Dados de teste
            model_name (str): Nome do modelo
            
        Returns:
            dict: Dicionário com métricas de avaliação
        """
        acc_train = accuracy_score(y_train, model.predict(X_train))
        acc_val = accuracy_score(y_val, model.predict(X_val))
        acc_test = accuracy_score(y_test, model.predict(X_test))
        
        y_pred_test = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        
        results = {
            'model_name': model_name,
            'acc_train': acc_train,
            'acc_val': acc_val,
            'acc_test': acc_test,
            'confusion_matrix': cm,
            'predictions': y_pred_test,
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
        }
        
        self.results[model_name] = results
        return results
    
    def cross_validation(self, model, X, y, cv_splits=5):
        """
        Realiza validação cruzada
        
        Args:
            model: Modelo a avaliar
            X: Features
            y: Labels
            cv_splits (int): Número de folds
            
        Returns:
            dict: Scores de validação cruzada
        """
        skfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'ci': scores.std() * 2
        }
    
    def get_roc_auc_scores(self, model, X_val, y_val, classes):
        """
        Calcula scores ROC/AUC usando One-vs-Rest
        
        Args:
            model: Modelo com predict_proba
            X_val: Features de validação
            y_val: Labels de validação
            classes: Classes únicas
            
        Returns:
            dict: Scores AUC por classe
        """
        y_proba = model.predict_proba(X_val)
        auc_scores = {}
        
        for i, class_label in enumerate(classes):
            y_true_binary = (y_val == class_label).astype(int)
            
            if class_label in model.classes_:
                class_idx = list(model.classes_).index(class_label)
                y_score_class = y_proba[:, class_idx]
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_score_class)
                roc_auc = auc(fpr, tpr)
                auc_scores[str(class_label)] = roc_auc
        
        return auc_scores
    
    def get_summary_results(self):
        """Retorna resumo de todos os resultados de avaliação"""
        return self.results
