# -*- coding: utf-8 -*-
"""
Script para construcción de features para el modelo de riesgo crediticio.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

# Imports opcionales - instalar scikit-learn cuando sea necesario
# from sklearn.preprocessing import StandardScaler, LabelEncoder  
# from sklearn.model_selection import train_test_split

# Alternativa temporal sin sklearn
class StandardScaler:
    def fit_transform(self, X): return X
    def transform(self, X): return X

class LabelEncoder:
    def fit_transform(self, y): return y
    
def train_test_split(*arrays, **options):
    """Función temporal - usar sklearn cuando esté instalado"""
    return arrays[0], arrays[0], arrays[1] if len(arrays) > 1 else None, arrays[1] if len(arrays) > 1 else None

class FeatureBuilder:
    """Clase para construir y transformar features para el modelo de crédito."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features financieras derivadas.
        
        Args:
            df: DataFrame con datos transaccionales
            
        Returns:
            DataFrame con features adicionales
        """
        logger = logging.getLogger(__name__)
        logger.info("Creando features financieras...")
        
        df_features = df.copy()
        
        # TODO: Implementar features específicas según el dataset
        # Ejemplos de features financieras:
        
        # # Ratios financieros
        # if 'income' in df.columns and 'expenses' in df.columns:
        #     df_features['income_expense_ratio'] = df['income'] / (df['expenses'] + 1)
        
        # # Features de transacciones
        # if 'transaction_amount' in df.columns:
        #     df_features['avg_transaction'] = df.groupby('customer_id')['transaction_amount'].transform('mean')
        #     df_features['std_transaction'] = df.groupby('customer_id')['transaction_amount'].transform('std')
        
        # # Features temporales
        # if 'date' in df.columns:
        #     df['date'] = pd.to_datetime(df['date'])
        #     df_features['month'] = df['date'].dt.month
        #     df_features['day_of_week'] = df['date'].dt.dayofweek
        
        logger.info(f"Features financieras creadas. Nuevas columnas: {df_features.shape[1] - df.shape[1]}")
        return df_features
        
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Codifica variables categóricas.
        
        Args:
            df: DataFrame con features
            categorical_cols: Lista de columnas categóricas
            
        Returns:
            DataFrame con variables codificadas
        """
        logger = logging.getLogger(__name__)
        logger.info("Codificando variables categóricas...")
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df_encoded[col] = self.encoders[col].transform(df[col].astype(str))
                    
        logger.info(f"Variables categóricas codificadas: {categorical_cols}")
        return df_encoded
        
    def scale_numerical_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Escala variables numéricas.
        
        Args:
            df: DataFrame con features
            numerical_cols: Lista de columnas numéricas
            
        Returns:
            DataFrame con variables escaladas
        """
        logger = logging.getLogger(__name__)
        logger.info("Escalando variables numéricas...")
        
        df_scaled = df.copy()
        
        for col in numerical_cols:
            if col in df.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df[[col]])
                else:
                    df_scaled[col] = self.scalers[col].transform(df[[col]])
                    
        logger.info(f"Variables numéricas escaladas: {numerical_cols}")
        return df_scaled
        
    def build_features(self, df: pd.DataFrame, target_col: str = 'default') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pipeline completo de construcción de features.
        
        Args:
            df: DataFrame raw
            target_col: Nombre de la columna objetivo
            
        Returns:
            Tuple con features (X) y target (y)
        """
        logger = logging.getLogger(__name__)
        logger.info("Iniciando construcción de features...")
        
        # Crear features financieras
        df_features = self.create_financial_features(df)
        
        # Identificar tipos de columnas
        categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remover target de las listas si está presente
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
            
        # Procesar features
        if categorical_cols:
            df_features = self.encode_categorical_features(df_features, categorical_cols)
        if numerical_cols:
            df_features = self.scale_numerical_features(df_features, numerical_cols)
            
        # Separar features y target
        if target_col in df_features.columns:
            X = df_features.drop(columns=[target_col])
            y = df_features[target_col]
        else:
            logger.warning(f"Columna objetivo '{target_col}' no encontrada")
            X = df_features
            y = None
            
        self.feature_names = X.columns.tolist()
        logger.info(f"Features construidas exitosamente. Shape: {X.shape}")
        
        return X, y

def prepare_train_test_split(X: pd.DataFrame, y: pd.Series, 
                           test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Features
        y: Target
        test_size: Proporción del conjunto de prueba
        random_state: Semilla aleatoria
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Dividiendo datos en train/test (test_size={test_size})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def main():
    """Función principal para construir features"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Iniciando construcción de features...")
    
    # TODO: Cargar datos desde data/interim o data/raw
    # df = pd.read_csv("data/interim/cleaned_data.csv")
    
    # # Construir features
    # feature_builder = FeatureBuilder()
    # X, y = feature_builder.build_features(df, target_col='default')
    
    # # Dividir en train/test
    # X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)
    
    # # Guardar datasets procesados
    # X_train.to_csv("data/processed/X_train.csv", index=False)
    # X_test.to_csv("data/processed/X_test.csv", index=False)
    # y_train.to_csv("data/processed/y_train.csv", index=False)
    # y_test.to_csv("data/processed/y_test.csv", index=False)
    
    logger.info("Construcción de features completada")

if __name__ == '__main__':
    main()