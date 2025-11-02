# -*- coding: utf-8 -*-
"""
Predicción de riesgo crediticio: carga de artefactos y scoring.

Soporta dos formas de despliegue:
1) Pipeline único serializado (recomendado): MODEL_PATH apunta a un .joblib/.pkl
    que contiene un sklearn.Pipeline con preprocesamiento + modelo.
2) Artefactos separados (opcional): PREPROCESSOR_PATH y MODEL_PATH; se arma
    un Pipeline en memoria con ('preprocessor', ...) y ('model', ...).
"""
import os
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline

class CreditRiskPredictor:
    """Clase para hacer predicciones de riesgo crediticio."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
    ):
        """
        Inicializa el predictor.

        Args:
            model_path: Ruta al modelo o pipeline serializado
            preprocessor_path: Ruta al preprocesador (si no está incluido en el pipeline)
        """
        self.model: Optional[Pipeline] = None
        self.model_path = model_path or os.getenv("MODEL_PATH", "models/best_model.pkl")
        self.preprocessor_path = preprocessor_path or os.getenv("PREPROCESSOR_PATH")
        self.feature_names = []
        self.load_model()
        
    def load_model(self) -> None:
        """Carga el modelo/pipeline desde disco; arma pipeline si hay preprocesador separado."""
        logger = logging.getLogger(__name__)

        # Verificar existencia
        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.error(f"Modelo no encontrado en: {model_path}")
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        # Cargar artefactos
        model_obj = joblib.load(model_path)

        if self.preprocessor_path:
            prep_path = Path(self.preprocessor_path)
            if not prep_path.exists():
                logger.error(f"Preprocesador no encontrado en: {prep_path}")
                raise FileNotFoundError(f"Preprocesador no encontrado: {prep_path}")
            preprocessor = joblib.load(prep_path)
            self.model = Pipeline(steps=[("preprocessor", preprocessor), ("model", model_obj)])
            logger.info(f"Pipeline armado en memoria con preprocesador + modelo.")
        else:
            # Si ya es un Pipeline, usarlo directamente; si es un estimador, igualmente se usará (asume features ya preprocesadas)
            self.model = model_obj
            logger.info(f"Modelo/pipeline cargado desde: {model_path}")

        # Intentar obtener nombres de features si está disponible
        try:
            if hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)
        except Exception:
            pass
            
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
        # Algunos modelos pueden no tener predict_proba
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df)[0]
            p1 = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            conf = float(max(probabilities))
        else:
            # Fallback: usar decisión como score binario
            p1 = float(prediction)
            conf = 1.0
        
        result = {
            'prediction': int(prediction),
            'risk_score': p1,  # Probabilidad de default (o aproximación)
            'risk_level': self._get_risk_level(p1),
            'confidence': conf
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
        has_proba = hasattr(self.model, "predict_proba")
        probabilities = self.model.predict_proba(df) if has_proba else None

        # Crear DataFrame de resultados
        results_df = df.copy()
        results_df['prediction'] = predictions
        if has_proba and probabilities is not None:
            p1 = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            results_df['risk_score'] = p1
            results_df['risk_level'] = [self._get_risk_level(score) for score in p1]
            results_df['confidence'] = np.max(probabilities, axis=1)
        else:
            results_df['risk_score'] = predictions.astype(float)
            results_df['risk_level'] = [self._get_risk_level(float(p)) for p in results_df['risk_score']]
            results_df['confidence'] = 1.0

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
                                  profit_margin: float = 0.05,
                                  decision_threshold: float = 0.5) -> Dict[str, Any]:
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

        # Decisiones de crédito (aprobación cuando el riesgo es menor o igual al umbral)
        results_df['credit_approved'] = results_df['risk_score'] <= decision_threshold
        
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