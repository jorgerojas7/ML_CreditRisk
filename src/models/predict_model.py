# -*- coding: utf-8 -*-
"""
Script para hacer predicciones con el modelo entrenado de riesgo crediticio.
"""
import logging
# import joblib  # Instalar cuando sea necesario
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any

# Función temporal para joblib
def joblib_load(filename):
    """Función temporal - reemplazar con joblib.load cuando esté instalado"""
    return None  # Retorna modelo dummy

class CreditRiskPredictor:
    """Clase para hacer predicciones de riesgo crediticio."""
    
    def __init__(self, model_path: str = "models/best_model.pkl"):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Ruta al modelo entrenado
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = []
        self.load_model()
        
    def load_model(self) -> None:
        """Carga el modelo entrenado desde disco."""
        logger = logging.getLogger(__name__)
        
        try:
            self.model = joblib_load(self.model_path)
            logger.info(f"Modelo cargado desde: {self.model_path}")
            
            # Intentar obtener nombres de features si está disponible
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
            
        except FileNotFoundError:
            logger.error(f"Modelo no encontrado en: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            raise
            
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hace predicción para un único perfil.
        
        Args:
            features: Diccionario con features del perfil
            
        Returns:
            Diccionario con predicción y probabilidades
        """
        logger = logging.getLogger(__name__)
        
        # Convertir a DataFrame
        df = pd.DataFrame([features])
        
        # Hacer predicción
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        result = {
            'prediction': int(prediction),
            'risk_score': float(probabilities[1]),  # Probabilidad de default
            'risk_level': self._get_risk_level(probabilities[1]),
            'confidence': float(max(probabilities))
        }
        
        logger.info(f"Predicción realizada: {result}")
        return result
        
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hace predicciones para un lote de perfiles.
        
        Args:
            df: DataFrame con perfiles a evaluar
            
        Returns:
            DataFrame con predicciones agregadas
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Realizando predicciones en lote para {len(df)} perfiles")
        
        # Hacer predicciones
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)
        
        # Crear DataFrame de resultados
        results_df = df.copy()
        results_df['prediction'] = predictions
        results_df['risk_score'] = probabilities[:, 1]
        results_df['risk_level'] = [self._get_risk_level(score) for score in probabilities[:, 1]]
        results_df['confidence'] = np.max(probabilities, axis=1)
        
        logger.info("Predicciones en lote completadas")
        return results_df
        
    def _get_risk_level(self, risk_score: float) -> str:
        """
        Convierte score numérico a nivel de riesgo categórico.
        
        Args:
            risk_score: Score de riesgo (0-1)
            
        Returns:
            Nivel de riesgo categórico
        """
        if risk_score < 0.3:
            return "BAJO"
        elif risk_score < 0.7:
            return "MEDIO"
        else:
            return "ALTO"
            
    def simulate_credit_decisions(self, df: pd.DataFrame, 
                                credit_amount_col: str = 'credit_amount',
                                profit_margin: float = 0.05) -> Dict[str, Any]:
        """
        Simula decisiones de crédito y calcula rentabilidad esperada.
        
        Args:
            df: DataFrame con perfiles y montos de crédito
            credit_amount_col: Nombre de columna con monto de crédito
            profit_margin: Margen de ganancia esperado en créditos exitosos
            
        Returns:
            Diccionario con resultados de simulación
        """
        logger = logging.getLogger(__name__)
        logger.info("Iniciando simulación de decisiones de crédito...")
        
        # Hacer predicciones
        results_df = self.predict_batch(df)
        
        # Definir umbral de decisión (puede ser optimizable)
        decision_threshold = 0.5
        
        # Decisiones de crédito
        results_df['credit_approved'] = results_df['risk_score'] < decision_threshold
        
        # Calcular métricas financieras simuladas
        if credit_amount_col in df.columns:
            approved_credits = results_df[results_df['credit_approved']]
            rejected_credits = results_df[~results_df['credit_approved']]
            
            # Simular resultados (esto sería reemplazado por datos reales)
            total_approved_amount = approved_credits[credit_amount_col].sum()
            expected_defaults = (approved_credits['risk_score'] * approved_credits[credit_amount_col]).sum()
            expected_profit = total_approved_amount * profit_margin
            expected_loss = expected_defaults
            net_profit = expected_profit - expected_loss
            
            simulation_results = {
                'total_applications': len(df),
                'approved_applications': len(approved_credits),
                'rejection_rate': len(rejected_credits) / len(df),
                'total_approved_amount': float(total_approved_amount),
                'expected_profit': float(expected_profit),
                'expected_loss': float(expected_loss),
                'net_profit': float(net_profit),
                'roi': float(net_profit / total_approved_amount) if total_approved_amount > 0 else 0
            }
        else:
            simulation_results = {
                'total_applications': len(df),
                'approved_applications': len(results_df[results_df['credit_approved']]),
                'rejection_rate': len(results_df[~results_df['credit_approved']]) / len(df)
            }
            
        logger.info("Simulación completada")
        logger.info(f"Tasa de aprobación: {(1 - simulation_results['rejection_rate']) * 100:.1f}%")
        
        return {
            'simulation_results': simulation_results,
            'detailed_predictions': results_df
        }

def load_test_data() -> pd.DataFrame:
    """
    Carga datos de prueba para predicciones.
    
    Returns:
        DataFrame con datos de prueba
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Intentar cargar datos de prueba reales
        test_data = pd.read_csv("data/processed/X_test.csv")
        logger.info(f"Datos de prueba cargados: {test_data.shape}")
        return test_data
        
    except FileNotFoundError:
        logger.warning("Datos de prueba no encontrados, generando datos sintéticos")
        
        # Generar datos sintéticos para demostración
        n_samples = 100
        np.random.seed(42)
        
        synthetic_data = pd.DataFrame({
            'income': np.random.normal(50000, 15000, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'credit_amount': np.random.uniform(1000, 50000, n_samples),
            'employment_length': np.random.randint(0, 30, n_samples),
            'debt_ratio': np.random.uniform(0, 1, n_samples)
        })
        
        # Asegurar valores positivos
        synthetic_data = synthetic_data.abs()
        
        logger.info(f"Datos sintéticos generados: {synthetic_data.shape}")
        return synthetic_data

def main():
    """Función principal para hacer predicciones"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Iniciando predicciones de riesgo crediticio...")
    
    try:
        # Cargar predictor
        predictor = CreditRiskPredictor()
        
        # Cargar datos de prueba
        test_data = load_test_data()
        
        # Hacer predicciones en lote
        results = predictor.predict_batch(test_data)
        
        # Simular decisiones de crédito
        simulation = predictor.simulate_credit_decisions(test_data)
        
        # Guardar resultados
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        results.to_csv(output_dir / "predictions.csv", index=False)
        
        # Mostrar resumen
        print("\n=== RESUMEN DE PREDICCIONES ===")
        print(f"Total de perfiles evaluados: {len(results)}")
        print(f"Riesgo promedio: {results['risk_score'].mean():.3f}")
        print(f"Distribución de riesgo:")
        print(results['risk_level'].value_counts())
        
        print("\n=== SIMULACIÓN DE DECISIONES ===")
        sim_results = simulation['simulation_results']
        for key, value in sim_results.items():
            print(f"{key}: {value}")
            
        logger.info("Predicciones completadas exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante las predicciones: {str(e)}")
        raise

if __name__ == '__main__':
    main()