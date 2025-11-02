"""
API principal para el servicio de predicción de riesgo crediticio.
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path
import sys

# Agregar src al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.predict_model import CreditRiskPredictor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Credit Risk Analysis API",
    description="API para predicción de riesgo crediticio usando ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (opcional)
security = HTTPBearer(auto_error=False)

# Modelos Pydantic para request/response
class CreditProfile(BaseModel):
    """Modelo para perfil crediticio de entrada."""
    income: float
    age: int
    credit_amount: float
    employment_length: int
    debt_ratio: float
    # Agregar más campos según el dataset específico
    
    class Config:
        schema_extra = {
            "example": {
                "income": 50000.0,
                "age": 35,
                "credit_amount": 15000.0,
                "employment_length": 5,
                "debt_ratio": 0.3
            }
        }

class CreditPrediction(BaseModel):
    """Modelo para respuesta de predicción."""
    prediction: int  # 0: no default, 1: default
    risk_score: float  # Probabilidad de default (0-1)
    risk_level: str  # BAJO, MEDIO, ALTO
    confidence: float  # Confianza de la predicción
    recommendation: str  # APROBAR, RECHAZAR, REVISAR

class BatchPredictionRequest(BaseModel):
    """Modelo para predicciones en lote."""
    profiles: List[CreditProfile]

class BatchPredictionResponse(BaseModel):
    """Respuesta para predicciones en lote."""
    predictions: List[CreditPrediction]
    summary: Dict[str, Any]

class SimulationRequest(BaseModel):
    """Modelo para simulación de decisiones."""
    profiles: List[CreditProfile]
    decision_threshold: Optional[float] = 0.5
    profit_margin: Optional[float] = 0.05

class SimulationResponse(BaseModel):
    """Respuesta de simulación."""
    simulation_results: Dict[str, Any]
    recommendations: List[str]

# Variable global para el predictor
predictor: Optional[CreditRiskPredictor] = None

def get_predictor():
    """Dependency para obtener el predictor."""
    global predictor
    if predictor is None:
        try:
            # Permitir configurar rutas vía variables de entorno
            import os
            model_path = os.getenv("MODEL_PATH")
            preprocessor_path = os.getenv("PREPROCESSOR_PATH")
            predictor = CreditRiskPredictor(model_path=model_path, preprocessor_path=preprocessor_path)
            logger.info("Predictor inicializado exitosamente")
        except Exception as e:
            logger.error(f"Error inicializando predictor: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error inicializando el modelo de predicción"
            )
    return predictor

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificación de token opcional (para implementar autenticación)."""
    # TODO: Implementar lógica de verificación de tokens
    # if credentials:
    #     token = credentials.credentials
    #     # Verificar token aquí
    #     pass
    return True

@app.on_event("startup")
async def startup_event():
    """Eventos al iniciar la aplicación."""
    logger.info("Iniciando Credit Risk Analysis API...")
    
    # Pre-cargar el modelo
    try:
        get_predictor()
        logger.info("Modelo cargado exitosamente al inicio")
    except Exception as e:
        logger.warning(f"No se pudo cargar el modelo al inicio: {e}")

@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "Credit Risk Analysis API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Endpoint de health check."""
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": predictor is not None,
            "timestamp": "2024-01-01T00:00:00Z"  # En producción usar datetime real
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/predict", response_model=CreditPrediction)
async def predict_credit_risk(
    profile: CreditProfile,
    predictor: CreditRiskPredictor = Depends(get_predictor),
    _: bool = Depends(verify_token)
):
    """
    Predice el riesgo crediticio para un perfil individual.
    """
    try:
        logger.info(f"Realizando predicción para perfil: {profile.dict()}")
        
        # Convertir a diccionario
        profile_dict = profile.dict()
        
        # Hacer predicción
        result = predictor.predict_single(profile_dict)
        
        # Agregar recomendación basada en risk_score
        if result['risk_score'] < 0.3:
            recommendation = "APROBAR"
        elif result['risk_score'] < 0.7:
            recommendation = "REVISAR"
        else:
            recommendation = "RECHAZAR"
            
        return CreditPrediction(
            prediction=result['prediction'],
            risk_score=result['risk_score'],
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando predicción: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor),
    _: bool = Depends(verify_token)
):
    """
    Predice el riesgo crediticio para múltiples perfiles.
    """
    try:
        logger.info(f"Realizando predicción en lote para {len(request.profiles)} perfiles")
        
        # Convertir perfiles a DataFrame
        import pandas as pd
        profiles_data = [profile.dict() for profile in request.profiles]
        df = pd.DataFrame(profiles_data)
        
        # Hacer predicciones
        results_df = predictor.predict_batch(df)
        
        # Convertir resultados a lista de predicciones
        predictions = []
        for _, row in results_df.iterrows():
            # Determinar recomendación
            if row['risk_score'] < 0.3:
                recommendation = "APROBAR"
            elif row['risk_score'] < 0.7:
                recommendation = "REVISAR"
            else:
                recommendation = "RECHAZAR"
                
            predictions.append(CreditPrediction(
                prediction=int(row['prediction']),
                risk_score=float(row['risk_score']),
                risk_level=row['risk_level'],
                confidence=float(row['confidence']),
                recommendation=recommendation
            ))
        
        # Calcular resumen
        summary = {
            "total_profiles": len(predictions),
            "average_risk_score": float(results_df['risk_score'].mean()),
            "risk_distribution": results_df['risk_level'].value_counts().to_dict(),
            "approval_rate": len([p for p in predictions if p.recommendation == "APROBAR"]) / len(predictions)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando predicción batch: {str(e)}"
        )

@app.post("/simulate", response_model=SimulationResponse)
async def simulate_credit_decisions(
    request: SimulationRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor),
    _: bool = Depends(verify_token)
):
    """
    Simula decisiones de crédito y calcula métricas de rentabilidad.
    """
    try:
        logger.info(f"Ejecutando simulación para {len(request.profiles)} perfiles")
        
        # Convertir perfiles a DataFrame
        import pandas as pd
        profiles_data = [profile.dict() for profile in request.profiles]
        df = pd.DataFrame(profiles_data)
        
        # Ejecutar simulación
        simulation_result = predictor.simulate_credit_decisions(
            df, 
            profit_margin=request.profit_margin
        )
        
        # Generar recomendaciones basadas en resultados
        sim_results = simulation_result['simulation_results']
        recommendations = []
        
        if sim_results.get('roi', 0) > 0.02:  # ROI mayor a 2%
            recommendations.append("Modelo muestra rentabilidad positiva")
        else:
            recommendations.append("Revisar criterios de aprobación")
            
        if sim_results.get('rejection_rate', 0) > 0.8:  # Tasa de rechazo muy alta
            recommendations.append("Considerar relajar criterios de aprobación")
        elif sim_results.get('rejection_rate', 0) < 0.3:  # Tasa de rechazo muy baja
            recommendations.append("Considerar endurecer criterios de aprobación")
            
        return SimulationResponse(
            simulation_results=sim_results,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error en simulación: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ejecutando simulación: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info(
    predictor: CreditRiskPredictor = Depends(get_predictor),
    _: bool = Depends(verify_token)
):
    """
    Obtiene información sobre el modelo cargado.
    """
    try:
        return {
            "model_path": predictor.model_path,
            "model_type": type(predictor.model).__name__,
            "feature_names": predictor.feature_names,
            "status": "loaded"
        }
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo información del modelo: {str(e)}"
        )

# Manejo de errores globales
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones."""
    logger.error(f"Error no manejado: {exc}")
    return {
        "error": "Error interno del servidor",
        "detail": "Por favor contacte al administrador"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_DEBUG", "false").lower() == "true"
    )