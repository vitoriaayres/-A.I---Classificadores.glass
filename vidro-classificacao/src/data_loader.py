# -*- coding: utf-8 -*-
"""
Módulo para carregamento e processamento de dados do Glass Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Classe responsável pelo carregamento e processamento de dados"""
    
    def __init__(self, data_path):
        """
        Inicializa o processador de dados
        
        Args:
            data_path (str): Caminho para o arquivo CSV do dataset
        """
        self.data_path = data_path
        self.dataset = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Carrega o dataset"""
        self.dataset = pd.read_csv(self.data_path)
        self.X = self.dataset.drop(columns="Type")
        self.y = self.dataset["Type"]
        return self.dataset
    
    def get_dataset_info(self):
        """Retorna informações sobre o dataset"""
        if self.dataset is None:
            self.load_data()
        
        return {
            "shape": self.dataset.shape,
            "columns": list(self.dataset.columns),
            "classes": sorted(self.dataset["Type"].unique()),
            "class_distribution": self.dataset["Type"].value_counts().to_dict()
        }
    
    def split_and_scale(self, test_size=0.4, val_size=0.5, random_state=42):
        """
        Divide os dados em treino, validação e teste, e escala os features
        
        Args:
            test_size (float): Proporção de dados para teste
            val_size (float): Proporção dos dados temp para validação
            random_state (int): Seed para reproducibilidade
            
        Returns:
            dict: Dicionário com X_train, X_val, X_test, y_train, y_val, y_test escalados
        """
        if self.X is None:
            self.load_data()
        
        # Primeira divisão: treino vs temp (validação + teste)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Segunda divisão: temp em validação e teste
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=random_state
        )
        
        # Escalonamento
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)
        X_test_sc = self.scaler.transform(X_test)
        
        return {
            "X_train": X_train_sc,
            "X_val": X_val_sc,
            "X_test": X_test_sc,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "scaler": self.scaler
        }
    
    def scale_input(self, input_data):
        """
        Escala dados de entrada usando o scaler já treinado
        
        Args:
            input_data: Dados a serem escalados
            
        Returns:
            array: Dados escalados
        """
        return self.scaler.transform(input_data)
