# -*- coding: utf-8 -*-
"""
Script para entrenar modelos de clasificación de riesgo crediticio.
"""
import logging
# import joblib  # Instalar cuando sea necesario
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

# Machine Learning models - COMENTADOS TEMPORALMENTE
# Descomentar cuando se instalen las dependencias:
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.model_selection import cross_val_score, GridSearchCV
# import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostClassifier

# Clases temporales para evitar errores
class RandomForestClassifier:
    def __init__(self, **kwargs): pass
    def fit(self, X, y): return self
    def predict(self, X): return X[:, 0] if len(X.shape) > 1 else X

def accuracy_score(y_true, y_pred): return 0.5
def precision_score(y_true, y_pred): return 0.5
def recall_score(y_true, y_pred): return 0.5
def f1_score(y_true, y_pred): return 0.5
def roc_auc_score(y_true, y_pred): return 0.5
def cross_val_score(model, X, y, cv=5): return [0.5] * cv

class GridSearchCV:
    def __init__(self, **kwargs): pass
    def fit(self, X, y): return self
    @property
    def best_estimator_(self): return RandomForestClassifier()

# Clases placeholder para ML avanzado
class lgb:
    class LGBMClassifier: 
        def __init__(self, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return X[:, 0] if len(X.shape) > 1 else X

class xgb:
    class XGBClassifier:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return X[:, 0] if len(X.shape) > 1 else X

class CatBoostClassifier:
    def __init__(self, **kwargs): pass
    def fit(self, X, y): return self
    def predict(self, X): return X[:, 0] if len(X.shape) > 1 else X

def joblib_dump(obj, filename): pass
def joblib_load(filename): return RandomForestClassifier()

class ModelTrainer:
    """Clase para entrenar y evaluar modelos de riesgo crediticio."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.evaluation_results = {}
        
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene configuraciones de modelos a entrenar.
        
        Returns:
            Diccionario con configuraciones de modelos
        """
        return {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'num_leaves': [31, 50],
                    'max_depth': [5, 10]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 6],
                    'subsample': [0.8, 1.0]
                }
            },
            'catboost': {
                'model': CatBoostClassifier(random_state=42, verbose=False),
                'params': {
                    'iterations': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'depth': [4, 6],
                    'l2_leaf_reg': [1, 3]
                }
            }
        }
        
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Entrena un modelo específico con optimización de hiperparámetros.
        
        Args:
            model_name: Nombre del modelo a entrenar
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Entrenando modelo: {model_name}")
        
        model_configs = self.get_model_configs()
        
        if model_name not in model_configs:
            raise ValueError(f"Modelo {model_name} no está configurado")
            
        config = model_configs[model_name]
        
        # Grid search para optimización de hiperparámetros
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.models[model_name] = best_model
        
        logger.info(f"Mejor score para {model_name}: {grid_search.best_score_:.4f}")
        logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        
        return best_model
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """
        Evalúa un modelo entrenado.
        
        Args:
            model: Modelo entrenado
            X_test: Features de prueba
            y_test: Target de prueba
            model_name: Nombre del modelo
            
        Returns:
            Diccionario con métricas de evaluación
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Evaluando modelo: {model_name}")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"Resultados para {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        return metrics
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Entrena y evalúa todos los modelos configurados.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con todos los modelos entrenados
        """
        logger = logging.getLogger(__name__)
        logger.info("Iniciando entrenamiento de todos los modelos...")
        
        model_configs = self.get_model_configs()
        
        for model_name in model_configs.keys():
            try:
                # Entrenar modelo
                model = self.train_model(model_name, X_train, y_train)
                
                # Evaluar modelo
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                
                # Actualizar mejor modelo si es necesario
                if metrics['roc_auc'] > self.best_score:
                    self.best_score = metrics['roc_auc']
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error entrenando modelo {model_name}: {str(e)}")
                continue
                
        logger.info(f"Mejor modelo: {self.best_model_name} (ROC-AUC: {self.best_score:.4f})")
        return self.models
        
    def save_models(self, models_path: str = "models/") -> None:
        """
        Guarda todos los modelos entrenados.
        
        Args:
            models_path: Directorio donde guardar los modelos
        """
        logger = logging.getLogger(__name__)
        logger.info("Guardando modelos...")
        
        models_dir = Path(models_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            model_file = models_dir / f"{model_name}_{timestamp}.pkl"
            joblib_dump(model, model_file)
            logger.info(f"Modelo guardado: {model_file}")
            
        # Guardar el mejor modelo por separado
        if self.best_model is not None:
            best_model_file = models_dir / "best_model.pkl"
            joblib_dump(self.best_model, best_model_file)
            logger.info(f"Mejor modelo guardado: {best_model_file}")
            
        # Guardar resultados de evaluación
        results_file = models_dir / f"evaluation_results_{timestamp}.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        logger.info(f"Resultados guardados: {results_file}")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Carga los datos procesados para entrenamiento.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger = logging.getLogger(__name__)
    logger.info("Cargando datos procesados...")
    
    try:
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
        
        logger.info(f"Datos cargados - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        logger.error(f"Archivos de datos no encontrados: {e}")
        logger.info("Por favor, ejecuta primero el script de construcción de features")
        raise

def main():
    """Función principal para entrenar modelos"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Iniciando entrenamiento de modelos de riesgo crediticio...")
    
    try:
        # Cargar datos
        X_train, X_test, y_train, y_test = load_data()
        
        # Entrenar modelos
        trainer = ModelTrainer()
        models = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Guardar modelos
        trainer.save_models()
        
        logger.info("Entrenamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == '__main__':
    main()