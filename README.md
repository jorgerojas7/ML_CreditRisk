# ML Credit Risk Analysis

Este proyecto genera un servicio API respaldado por modelos de machine learning que puede predecir puntuaciones de riesgo crediticio basándose en perfiles financieros.

## Objetivo del Proyecto

Crear un servicio capaz de predecir las puntuaciones crediticias de las personas basándose en información de transacciones financieras, incluyendo simulaciones para evaluar la rentabilidad del modelo en un entorno real.

## Arquitectura del Sistema

El proyecto utiliza una arquitectura modular con dos componentes principales:

### 🔧 **Backend - FastAPI**
- **API REST** para predicciones de riesgo crediticio
- **Endpoints especializados** para análisis individual, por lotes y simulaciones
- **Validación de datos** con Pydantic
- **Documentación automática** con OpenAPI/Swagger

### 🎨 **Frontend - Streamlit**
- **Dashboard interactivo** para análisis exploratorio de datos
- **Interfaz de predicción** para casos individuales y por lotes
- **Visualizaciones dinámicas** con Plotly
- **Simulación de escenarios** de negocio

### 📊 **Dataset**
- **PAKDD 2010 Credit Risk Competition** - Datos reales de riesgo crediticio
- **Variables financieras** y demográficas de clientes
- **Target binario** para clasificación de riesgo

## Estructura del Proyecto

```
├── README.md          <- Descripción principal del proyecto
├── data/
│   ├── external/      <- Datos de fuentes externas
│   ├── interim/       <- Datos intermedios transformados
│   ├── processed/     <- Conjuntos de datos finales y canónicos
│   └── raw/           <- Datos originales sin modificar
│
├── docs/              <- Documentación del proyecto
│
├── models/            <- Modelos entrenados y serializados, predicciones
│
├── notebooks/         <- Jupyter notebooks para EDA y experimentación
│
├── references/        <- Diccionarios de datos, manuales y materiales explicativos
│
├── reports/           <- Análisis generados como HTML, PDF, LaTeX, etc.
│   └── figures/       <- Gráficos y figuras para usar en reportes
│
├── requirements.txt   <- Dependencias para reproducir el entorno de análisis
│
├── setup.py          <- Hace que el proyecto sea instalable con pip (pip install -e .)
│
├── src/              <- Código fuente para uso en este proyecto
│   ├── __init__.py   <- Hace que src sea un módulo Python
│   │
│   ├── data/         <- Scripts para descargar o generar datos
│   │   └── make_dataset.py
│   │
│   ├── features/     <- Scripts para convertir datos raw en features para modeling
│   │   └── build_features.py
│   │
│   ├── models/       <- Scripts para entrenar modelos y hacer predicciones
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization/ <- Scripts para crear visualizaciones exploratorias y de resultados
│       └── visualize.py
│
├── api/              <- API FastAPI para el servicio backend
│   ├── main.py       <- Aplicación principal de la API
│   ├── models.py     <- Modelos Pydantic para request/response
│   └── routers/      <- Endpoints organizados por funcionalidad
│
├── frontend/         <- Aplicación Streamlit para interfaz web interactiva
│   ├── streamlit_app.py <- Aplicación principal de Streamlit
│   └── utils.py      <- Utilidades y funciones auxiliares para el frontend
│
├── tests/            <- Tests unitarios y de integración
│   └── test_api.py   <- Tests para los endpoints de la API
│
└── deployment/       <- Archivos Docker y configuración para deployment
    ├── Dockerfile
    └── docker-compose.yml
```

## Entregables Principales

1. **Análisis exploratorio de datos (EDA)** - Notebooks Jupyter y datasets
2. **Scripts de preprocesamiento** - Para preparación de datos
3. **Scripts de entrenamiento y modelos entrenados** - Con documentación de reproducibilidad
4. **Modelo de predicción de puntuación crediticia**
5. **Simulación del modelo** - Con documentación de resultados y proceso
6. **API con interfaz de usuario** - Para demostraciones
7. **Dockerización completa** - Lista para deployment

## Entregables Opcionales

- Autenticación basada en tokens
- Re-entrenamiento online con nuevos datos
- Tests adicionales de API

## Configuración del Entorno

**Requisitos:** Python 3.10 o superior

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias básicas (ya instaladas ✅)
pip install -r requirements.txt

# Instalar el proyecto en modo desarrollo
pip install -e .
```

### 📦 **Dependencias Actuales**
- ✅ **Instaladas**: pandas, numpy, plotly, fastapi, uvicorn, streamlit, requests
- 🔄 **Opcionales**: scikit-learn, matplotlib, seaborn (se instalarán según necesidad)
- 🚀 **ML Avanzado**: lightgbm, xgboost, catboost (para modelos avanzados)

### 🗂️ **Datos del Proyecto**
- **Dataset**: PAKDD 2010 Credit Risk Competition
- **Ubicación**: `data/raw/` (ya copiados ✅)
- **Formato**: Archivos .txt con datos tabulares

## Uso Rápido

### 📈 **Análisis Exploratorio**
```bash
# Abrir notebook de EDA
jupyter notebook notebooks/01_EDA_PAKDD2010.ipynb
```

### 🤖 **Entrenamiento del Modelo**
```bash
# Entrenar modelos (cuando sklearn esté instalado)
python src/models/train_model.py
```

### 🚀 **Ejecutar Servicios**

**Backend API (FastAPI):**
```bash
uvicorn api.main:app --reload --port 8000
# Documentación: http://localhost:8000/docs
```

**Frontend Dashboard (Streamlit):**
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
# Aplicación: http://localhost:8501
```

**Docker (Servicios completos):**
```bash
docker-compose up --build
```

## Hitos del Proyecto

- [x] Configurar repositorio y estructura
- [ ] Descarga y evaluación del dataset
- [ ] Normalización de datos y EDA
- [ ] Creación de dataset de entrenamiento
- [ ] Entrenamiento de modelos clasificadores
- [ ] Evaluación y selección del mejor modelo
- [ ] Configuración de API
- [ ] Integración de UI básica
- [ ] Ajuste de modelos adicionales
- [ ] Tests de API (opcional)
- [ ] Presentación final

## Contribución

Por favor, revisa las guías de contribución en `docs/` antes de hacer cambios.

## Licencia

Este proyecto está bajo la licencia MIT - ver el archivo LICENSE para detalles.
test