"""
Aplicación Streamlit para interfaz de usuario del sistema de análisis de riesgo crediticio.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import json
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de la API
API_BASE_URL = "http://localhost:8000"

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 10px 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Verifica si la API está disponible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_single_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Envía predicción individual a la API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=profile_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error conectando con la API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return None

def predict_batch_profiles(profiles_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Envía predicciones en lote a la API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"profiles": profiles_data},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error conectando con la API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return None

def simulate_credit_decisions(profiles_data: List[Dict[str, Any]], 
                            decision_threshold: float = 0.5,
                            profit_margin: float = 0.05) -> Dict[str, Any]:
    """Envía solicitud de simulación a la API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/simulate",
            json={
                "profiles": profiles_data,
                "decision_threshold": decision_threshold,
                "profit_margin": profit_margin
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error conectando con la API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return None

def display_risk_result(prediction: Dict[str, Any]):
    """Muestra el resultado de predicción de riesgo."""
    risk_level = prediction.get('risk_level', 'DESCONOCIDO')
    risk_score = prediction.get('risk_score', 0)
    recommendation = prediction.get('recommendation', 'REVISAR')
    confidence = prediction.get('confidence', 0)
    
    # Determinar clase CSS según el nivel de riesgo
    if risk_level == 'ALTO':
        css_class = "risk-high"
        color = "#f44336"
    elif risk_level == 'MEDIO':
        css_class = "risk-medium" 
        color = "#ff9800"
    else:
        css_class = "risk-low"
        color = "#4caf50"
    
    # Mostrar resultado principal
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Nivel de Riesgo",
            value=risk_level,
            delta=f"Score: {risk_score:.3f}"
        )
    
    with col2:
        st.metric(
            label="Recomendación",
            value=recommendation,
            delta=f"Confianza: {confidence:.3f}"
        )
    
    with col3:
        # Gauge chart para risk score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "#e8f5e8"},
                    {'range': [30, 70], 'color': "#fff3e0"},
                    {'range': [70, 100], 'color': "#ffebee"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Mostrar detalles adicionales
        st.markdown(f"""
        <div class="{css_class}">
            <h4>Detalles de la Predicción</h4>
            <p><strong>Probabilidad de Default:</strong> {risk_score:.1%}</p>
            <p><strong>Nivel de Confianza:</strong> {confidence:.1%}</p>
            <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

def create_profile_input_form() -> Dict[str, Any]:
    """Crea formulario para capturar datos del perfil crediticio."""
    st.subheader("📋 Datos del Perfil Crediticio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input(
            "Ingresos Anuales ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=50000.0,
            step=1000.0,
            help="Ingresos anuales del solicitante"
        )
        
        age = st.slider(
            "Edad",
            min_value=18,
            max_value=100,
            value=35,
            help="Edad del solicitante"
        )
        
        employment_length = st.slider(
            "Años de Empleo",
            min_value=0,
            max_value=50,
            value=5,
            help="Años en el empleo actual"
        )
    
    with col2:
        credit_amount = st.number_input(
            "Monto de Crédito Solicitado ($)",
            min_value=0.0,
            max_value=500000.0,
            value=15000.0,
            step=500.0,
            help="Monto del crédito solicitado"
        )
        
        debt_ratio = st.slider(
            "Ratio de Deuda",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Ratio de deuda actual (deuda/ingresos)"
        )
    
    return {
        "income": float(income),
        "age": int(age),
        "credit_amount": float(credit_amount),
        "employment_length": int(employment_length),
        "debt_ratio": float(debt_ratio)
    }

def main():
    """Función principal de la aplicación."""
    
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Risk Analysis System</h1>', unsafe_allow_html=True)
    
    # Verificar conexión con API
    with st.spinner("Verificando conexión con API..."):
        api_status = check_api_health()
    
    if not api_status:
        st.error("⚠️ No se puede conectar con la API. Asegúrate de que el servidor FastAPI esté ejecutándose en http://localhost:8000")
        st.info("Para iniciar la API, ejecuta: `uvicorn api.main:app --reload`")
        st.stop()
    
    st.success("✅ Conexión con API establecida")
    
    # Sidebar para navegación
    st.sidebar.title("📊 Navegación")
    page = st.sidebar.selectbox(
        "Selecciona una opción:",
        [
            "🔍 Análisis Individual",
            "📊 Análisis en Lote", 
            "🎯 Simulación de Decisiones",
            "📈 Dashboard de Métricas"
        ]
    )
    
    if page == "🔍 Análisis Individual":
        st.header("🔍 Análisis de Riesgo Individual")
        
        # Formulario de entrada
        profile_data = create_profile_input_form()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Predecir Riesgo Crediticio", type="primary", use_container_width=True):
                with st.spinner("Analizando perfil crediticio..."):
                    result = predict_single_profile(profile_data)
                
                if result:
                    st.success("✅ Análisis completado")
                    display_risk_result(result)
                else:
                    st.error("❌ Error al procesar la predicción")
    
    elif page == "📊 Análisis en Lote":
        st.header("📊 Análisis de Riesgo en Lote")
        
        # Opción 1: Upload CSV
        st.subheader("📁 Cargar archivo CSV")
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV con los perfiles crediticios",
            type=['csv'],
            help="El archivo debe contener columnas: income, age, credit_amount, employment_length, debt_ratio"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Archivo cargado: {len(df)} perfiles encontrados")
                
                # Mostrar preview
                st.subheader("👀 Vista previa de los datos")
                st.dataframe(df.head())
                
                # Validar columnas requeridas
                required_cols = ['income', 'age', 'credit_amount', 'employment_length', 'debt_ratio']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Faltan las siguientes columnas: {missing_cols}")
                else:
                    if st.button("🔮 Analizar Todos los Perfiles", type="primary"):
                        profiles_list = df[required_cols].to_dict('records')
                        
                        with st.spinner(f"Analizando {len(profiles_list)} perfiles..."):
                            batch_result = predict_batch_profiles(profiles_list)
                        
                        if batch_result:
                            st.success("✅ Análisis en lote completado")
                            
                            # Mostrar resumen
                            summary = batch_result.get('summary', {})
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Perfiles", summary.get('total_profiles', 0))
                            with col2:
                                st.metric("Risk Score Promedio", f"{summary.get('average_risk_score', 0):.3f}")
                            with col3:
                                st.metric("Tasa de Aprobación", f"{summary.get('approval_rate', 0):.1%}")
                            with col4:
                                risk_dist = summary.get('risk_distribution', {})
                                most_common = max(risk_dist.keys(), key=lambda x: risk_dist[x]) if risk_dist else "N/A"
                                st.metric("Nivel Más Común", most_common)
                            
                            # Gráficos de distribución
                            st.subheader("📈 Análisis de Resultados")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Distribución de niveles de riesgo
                                risk_dist = summary.get('risk_distribution', {})
                                if risk_dist:
                                    fig_pie = px.pie(
                                        values=list(risk_dist.values()),
                                        names=list(risk_dist.keys()),
                                        title="Distribución de Niveles de Riesgo",
                                        color_discrete_map={
                                            'BAJO': '#4caf50',
                                            'MEDIO': '#ff9800', 
                                            'ALTO': '#f44336'
                                        }
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                # Histograma de risk scores
                                predictions = batch_result.get('predictions', [])
                                if predictions:
                                    risk_scores = [p['risk_score'] for p in predictions]
                                    fig_hist = px.histogram(
                                        x=risk_scores,
                                        title="Distribución de Risk Scores",
                                        labels={'x': 'Risk Score', 'y': 'Frecuencia'}
                                    )
                                    st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Tabla detallada
                            st.subheader("📋 Resultados Detallados")
                            if predictions:
                                results_df = pd.DataFrame([
                                    {
                                        'Risk Score': p['risk_score'],
                                        'Risk Level': p['risk_level'],
                                        'Recommendation': p['recommendation'],
                                        'Confidence': p['confidence']
                                    } for p in predictions
                                ])
                                
                                # Combinar con datos originales
                                combined_df = pd.concat([df, results_df], axis=1)
                                st.dataframe(combined_df)
                                
                                # Opción de descarga
                                csv_download = combined_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Descargar Resultados (CSV)",
                                    data=csv_download,
                                    file_name=f"credit_risk_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
            
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")
        
        # Opción 2: Generar datos sintéticos para prueba
        st.subheader("🧪 Generar Datos de Prueba")
        num_samples = st.slider("Número de perfiles sintéticos", 10, 1000, 100)
        
        if st.button("🎲 Generar y Analizar Datos Sintéticos"):
            # Generar datos sintéticos
            import numpy as np
            np.random.seed(42)
            
            synthetic_profiles = []
            for _ in range(num_samples):
                synthetic_profiles.append({
                    "income": float(np.random.normal(50000, 15000)),
                    "age": int(np.random.randint(18, 80)),
                    "credit_amount": float(np.random.uniform(1000, 50000)),
                    "employment_length": int(np.random.randint(0, 30)),
                    "debt_ratio": float(np.random.uniform(0, 1))
                })
            
            # Asegurar valores positivos
            for profile in synthetic_profiles:
                profile["income"] = abs(profile["income"])
                profile["credit_amount"] = abs(profile["credit_amount"])
            
            with st.spinner(f"Analizando {num_samples} perfiles sintéticos..."):
                batch_result = predict_batch_profiles(synthetic_profiles)
            
            if batch_result:
                st.success("✅ Análisis de datos sintéticos completado")
                # Mostrar mismos gráficos que arriba...
    
    elif page == "🎯 Simulación de Decisiones":
        st.header("🎯 Simulación de Decisiones Crediticias")
        
        st.info("💡 Esta funcionalidad simula el impacto financiero de las decisiones crediticias basadas en el modelo.")
        
        # Parámetros de simulación
        st.subheader("⚙️ Parámetros de Simulación")
        
        col1, col2 = st.columns(2)
        with col1:
            decision_threshold = st.slider(
                "Umbral de Decisión",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Risk score por debajo del cual se aprueba el crédito"
            )
        
        with col2:
            profit_margin = st.slider(
                "Margen de Ganancia",
                min_value=0.0,
                max_value=0.5,
                value=0.05,
                step=0.01,
                help="Margen de ganancia esperado en créditos exitosos"
            )
        
        # Generar datos para simulación
        if st.button("🚀 Ejecutar Simulación"):
            # Generar datos sintéticos para la simulación
            import numpy as np
            np.random.seed(42)
            
            simulation_profiles = []
            for _ in range(500):  # Simulación con 500 perfiles
                simulation_profiles.append({
                    "income": float(abs(np.random.normal(50000, 20000))),
                    "age": int(np.random.randint(18, 80)),
                    "credit_amount": float(np.random.uniform(5000, 100000)),
                    "employment_length": int(np.random.randint(0, 30)),
                    "debt_ratio": float(np.random.uniform(0, 0.8))
                })
            
            with st.spinner("Ejecutando simulación..."):
                simulation_result = simulate_credit_decisions(
                    simulation_profiles,
                    decision_threshold,
                    profit_margin
                )
            
            if simulation_result:
                st.success("✅ Simulación completada")
                
                # Mostrar resultados de simulación
                sim_results = simulation_result.get('simulation_results', {})
                recommendations = simulation_result.get('recommendations', [])
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Solicitudes Totales",
                        sim_results.get('total_applications', 0)
                    )
                
                with col2:
                    st.metric(
                        "Solicitudes Aprobadas", 
                        sim_results.get('approved_applications', 0)
                    )
                
                with col3:
                    rejection_rate = sim_results.get('rejection_rate', 0)
                    st.metric(
                        "Tasa de Rechazo",
                        f"{rejection_rate:.1%}"
                    )
                
                with col4:
                    roi = sim_results.get('roi', 0)
                    st.metric(
                        "ROI Esperado",
                        f"{roi:.2%}",
                        delta=f"{'Positivo' if roi > 0 else 'Negativo'}"
                    )
                
                # Métricas financieras detalladas
                if 'total_approved_amount' in sim_results:
                    st.subheader("💰 Análisis Financiero")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Monto Total Aprobado",
                            f"${sim_results.get('total_approved_amount', 0):,.0f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Ganancia Esperada",
                            f"${sim_results.get('expected_profit', 0):,.0f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Pérdida Esperada", 
                            f"${sim_results.get('expected_loss', 0):,.0f}"
                        )
                
                # Recomendaciones
                st.subheader("💡 Recomendaciones")
                for rec in recommendations:
                    st.info(f"• {rec}")
    
    elif page == "📈 Dashboard de Métricas":
        st.header("📈 Dashboard de Métricas del Sistema")
        
        # Información del modelo
        try:
            model_info_response = requests.get(f"{API_BASE_URL}/model/info")
            if model_info_response.status_code == 200:
                model_info = model_info_response.json()
                
                st.subheader("🤖 Información del Modelo")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Tipo de Modelo:** {model_info.get('model_type', 'N/A')}")
                    st.info(f"**Ruta del Modelo:** {model_info.get('model_path', 'N/A')}")
                
                with col2:
                    features = model_info.get('feature_names', [])
                    st.info(f"**Número de Features:** {len(features)}")
                    if features:
                        st.info(f"**Features:** {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
            
        except:
            st.warning("⚠️ No se pudo obtener información del modelo")
        
        # Métricas del sistema (simuladas)
        st.subheader("📊 Métricas del Sistema")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Predicciones Hoy",
                "1,234",
                delta="12%"
            )
        
        with col2:
            st.metric(
                "Tiempo Promedio de Respuesta",
                "145ms",
                delta="-5ms"
            )
        
        with col3:
            st.metric(
                "Uptime",
                "99.9%",
                delta="0.1%"
            )
        
        with col4:
            st.metric(
                "Precisión del Modelo",
                "87.3%",
                delta="1.2%"
            )

if __name__ == "__main__":
    main()