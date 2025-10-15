"""
Modelos Pydantic para validación de request/response de la API.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime

class CreditProfile(BaseModel):
    """Modelo base para perfil crediticio."""
    income: float = Field(..., gt=0, description="Ingresos anuales")
    age: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    credit_amount: float = Field(..., gt=0, description="Monto de crédito solicitado")
    employment_length: int = Field(..., ge=0, description="Años de empleo")
    debt_ratio: float = Field(..., ge=0, le=1, description="Ratio de deuda")
    
    @validator('income')
    def validate_income(cls, v):
        if v < 0:
            raise ValueError('El ingreso debe ser positivo')
        return v
    
    @validator('debt_ratio')
    def validate_debt_ratio(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('El ratio de deuda debe estar entre 0 y 1')
        return v
    
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

class ExtendedCreditProfile(CreditProfile):
    """Perfil crediticio extendido con campos adicionales."""
    # Agregar campos adicionales según el dataset específico
    education_level: Optional[str] = Field(None, description="Nivel de educación")
    marital_status: Optional[str] = Field(None, description="Estado civil")
    number_of_dependents: Optional[int] = Field(None, ge=0, description="Número de dependientes")
    property_ownership: Optional[str] = Field(None, description="Tipo de propiedad")
    years_at_current_address: Optional[int] = Field(None, ge=0, description="Años en dirección actual")

class CreditPrediction(BaseModel):
    """Respuesta de predicción de riesgo crediticio."""
    prediction: int = Field(..., description="0: no default, 1: default")
    risk_score: float = Field(..., ge=0, le=1, description="Score de riesgo (0-1)")
    risk_level: str = Field(..., description="Nivel de riesgo: BAJO, MEDIO, ALTO")
    confidence: float = Field(..., ge=0, le=1, description="Confianza de la predicción")
    recommendation: str = Field(..., description="Recomendación: APROBAR, RECHAZAR, REVISAR")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp de la predicción")

class BatchPredictionRequest(BaseModel):
    """Request para predicciones en lote."""
    profiles: List[CreditProfile] = Field(..., min_items=1, max_items=1000)
    
    @validator('profiles')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Máximo 1000 perfiles por lote')
        return v

class BatchPredictionResponse(BaseModel):
    """Respuesta para predicciones en lote."""
    predictions: List[CreditPrediction]
    summary: Dict[str, Any]
    total_processed: int
    processing_time_seconds: Optional[float] = None

class SimulationRequest(BaseModel):
    """Request para simulación de decisiones crediticias."""
    profiles: List[CreditProfile] = Field(..., min_items=1)
    decision_threshold: Optional[float] = Field(0.5, ge=0, le=1, description="Umbral de decisión")
    profit_margin: Optional[float] = Field(0.05, ge=0, le=1, description="Margen de ganancia")
    loss_given_default: Optional[float] = Field(0.5, ge=0, le=1, description="Pérdida en caso de default")

class SimulationResponse(BaseModel):
    """Respuesta de simulación."""
    simulation_results: Dict[str, Any]
    recommendations: List[str]
    profitability_analysis: Dict[str, float]

class ModelInfo(BaseModel):
    """Información del modelo cargado."""
    model_type: str
    model_path: str
    feature_names: List[str]
    model_version: Optional[str] = None
    training_date: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    """Respuesta de health check."""
    status: str = Field(..., description="healthy o unhealthy")
    model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None

class ErrorResponse(BaseModel):
    """Respuesta de error estándar."""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

class RetrainingRequest(BaseModel):
    """Request para re-entrenar modelo (funcionalidad opcional)."""
    new_data: List[Dict[str, Any]] = Field(..., min_items=1)
    retrain_immediately: bool = Field(False, description="Entrenar inmediatamente o programar")
    model_name: Optional[str] = Field(None, description="Nombre del nuevo modelo")

class APIUsageStats(BaseModel):
    """Estadísticas de uso de la API."""
    total_predictions: int
    predictions_today: int
    average_response_time: float
    most_common_risk_level: str
    uptime_percentage: float