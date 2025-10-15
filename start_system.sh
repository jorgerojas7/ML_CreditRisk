#!/bin/bash

# Script para iniciar tanto la API FastAPI como la aplicación Streamlit (Linux/Mac)

echo "========================================"
echo " Credit Risk Analysis System - Startup"
echo "========================================"
echo

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 no está instalado"
    exit 1
fi

# Verificar si las dependencias están instaladas
echo "Verificando dependencias..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "Instalando dependencias..."
    pip3 install -r requirements.txt
fi

# Crear directorios necesarios
mkdir -p data/{raw,interim,processed,external}
mkdir -p models
mkdir -p logs
mkdir -p reports/figures

echo
echo "Iniciando servicios..."
echo

# Iniciar FastAPI en segundo plano
echo "[1/2] Iniciando API FastAPI en puerto 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Esperar un momento para que la API inicie
sleep 5

# Verificar que la API esté funcionando
echo "Verificando que la API esté funcionando..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Esperando a que la API inicie..."
    sleep 10
fi

# Iniciar Streamlit
echo "[2/2] Iniciando interfaz Streamlit en puerto 8501..."
echo
echo "========================================"
echo " URLs del sistema:"
echo " - API FastAPI: http://localhost:8000"
echo " - Docs API: http://localhost:8000/docs"
echo " - Streamlit UI: http://localhost:8501"
echo "========================================"
echo

cd frontend
streamlit run streamlit_app.py &
STREAMLIT_PID=$!

# Función para limpiar procesos al salir
cleanup() {
    echo "Cerrando servicios..."
    kill $API_PID 2>/dev/null
    kill $STREAMLIT_PID 2>/dev/null
    exit 0
}

# Configurar trap para limpieza
trap cleanup SIGINT SIGTERM

# Mantener el script corriendo
wait