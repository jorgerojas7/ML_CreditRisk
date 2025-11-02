# ML Credit Risk Analysis

> Predicción de riesgo crediticio con pipelines reproducibles, importancia de variables y flujo de scoring/visualización. Incluye API/Frontend opcional para demo.

## 📦 Componentes principales

- `ml_creditrisk/` (paquete Python):
    - `feature_grouping.py`: utilidades para agrupar variables y auditar Missing%.
    - `preprocessing.py`: ColumnTransformer desde grupos (imputación, winsor, OHE, target-encoding) y discretizador robusto por cuantiles.
    - `importance.py`: entrenamiento XGBoost + agregado de importancias a variables originales, filtrado por umbral y plotting.
    - `models.py`: modelos base (RF, XGBoost, LightGBM, CatBoost) + evaluador y pipeline “GB leaves → OneHot → LR”.
- `notebooks/02_Feature_Engineering_Modelado.ipynb`: orquesta el flujo E2E (carga → grupos → preprocesamiento → importancia → modelos → predicciones → gráficos).
- `api/` y `frontend/`: demo opcional con FastAPI/Streamlit (no requerida para el notebook).

## 🧰 Requisitos y entorno

- Python 3.10 recomendado (Windows soportado)
- Instalar dependencias:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Notas:
- Para leer el archivo .XLS del dataset se fija xlrd==1.2.0 (las versiones ≥2.0 ya no soportan .xls).
- LightGBM y CatBoost están incluidos en requirements y se usan si están instalados.

## 🗂️ Datos

- Ubicación: `data/raw/`
- Archivos esperados:
    - `PAKDD2010_VariablesList.XLS` (nombres de columnas)
    - `PAKDD2010_Modeling_Data.txt` (modelado)
    - `PAKDD2010_Prediction_Data.txt` (scoring)

## 🧪 Uso del notebook principal

1) Abrir `notebooks/02_Feature_Engineering_Modelado.ipynb` y ejecutar en orden:
     - Celda 1: carga de datos.
     - Celda 2: agrupación de variables, exclusiones y DataFrame FINAL (auditable).
     - Celda 3: construcción del preprocesador y resumen de columnas generadas.
     - Celda 4: importancia con XGBoost; umbral configurable; crea `preprocessor_filtered` con variables ≥ umbral (por defecto 0.02 en el cuaderno; se puede ajustar).
     - Celda 5: entrenamiento y evaluación de modelos activos (RF, XGBoost, LightGBM, CatBoost).
     - Celda 6: predicciones sobre `Prediction_Data.txt` y columnas `score_*` en `df_pred`.
     - Celda 7: histogramas de scores por modelo.

2) Búsqueda de hiperparámetros (RandomizedSearchCV):
     - La celda de HPO incluye un flag `HPO_ENABLED = False` para evitar ejecuciones largas. Cambiar a `True` para activar.
     - Los mejores pipelines quedan en `tuned_models`.
     - En la celda de predicciones, `USE_TUNED_MODELS = False` por defecto. Cambiar a `True` para usar `tuned_models` si existen.

## 🤖 Modelos incluidos

- Random Forest (scikit-learn)
- XGBoost (xgboost)
- LightGBM (lightgbm) – opcional si instalado
- CatBoost (catboost) – opcional si instalado
- (Opcional) GB leaves → OneHot → LR (útil para calibración y capturar interacciones de árboles)

## 🧩 Diseño del preprocesamiento

- Numéricas reales: imputación (−1) → winsor (cuantiles) → robust scaler
- Numéricas con prioridad (AGE, MONTHS_IN_THE_JOB): discretización por cuantiles (robusta)
- Categóricas baja cardinalidad y binarias: OneHotEncoder
- Categóricas alta cardinalidad: TargetEncoder (smoothing=0.3)
- Todas las decisiones dependen de `df_groups_final` como “single source of truth”.

## � Importancia y filtrado

- Importancias por feature output se agregan a variables raw originales.
- Se construye `preprocessor_filtered` con variables ≥ umbral.
- Tabla de variables eliminadas incluida para auditoría.

## ▶️ API / Frontend (opcional)

Para demo rápida (cuando quieras mostrar un servicio):

```powershell
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs

streamlit run frontend/streamlit_app.py --server.port 8501
# App: http://localhost:8501
```

## ✅ Checklist de reproducibilidad

- [x] requirements.txt actualizado (incluye xlrd==1.2.0, sklearn, xgboost, lightgbm, catboost, scipy, etc.)
- [x] Paquete `ml_creditrisk` con docstrings y funciones reutilizables
- [x] Notebook principal orquestando el flujo E2E
- [x] Flags para activar/desactivar HPO y usar modelos tuneados

## � Guardar y usar el preprocesador (.joblib)

En la última sección del notebook `02_Feature_Engineering_Modelado.ipynb` se incluye una celda para entrenar el preprocesador activo y guardarlo como artefacto reutilizable.

- Qué guarda:
    - `models/preprocessor_active_<timestamp>.joblib`: el `ColumnTransformer`/`Pipeline` final ajustado sobre el set de entrenamiento.
    - `models/preprocessor_active_<timestamp>.json`: metadatos (umbral de importancia, tamaños, fecha, hash de columnas, etc.).

- Cómo cargarlo y usarlo en otro script o sesión:
  
    Ejemplo mínimo en Python:
  
    1) Cargar el artefacto
    2) Aplicar `transform` sobre un DataFrame con el mismo esquema de columnas que el de entrenamiento

    Notas:
    - El artefacto espera las mismas columnas “raw” de entrada que se usaron en el entrenamiento (mismo `df_groups_final`).
    - Si cambian los grupos o el umbral de importancia, se debe volver a entrenar y guardar un nuevo artefacto.

- Versionado y .gitignore:
    - Por defecto, `models/*.joblib` y `models/*.json` están ignorados en `.gitignore` para evitar subir artefactos pesados.
    - Si necesitas versionar un artefacto concreto, puedes:
        - Forzar el agregado con `git add -f models/preprocessor_active_YYYYMMDD_HHMMSS.joblib` y su `.json`, o
        - Quitar/ajustar la regla de `.gitignore` para `models/*`.

Sugerencia: mantén un naming consistente y documenta en el metadato el dataset y los flags utilizados (por ejemplo, `IMPORTANCE_THRESHOLD`).

## �📄 Licencia

MIT (ver archivo LICENSE si aplica).