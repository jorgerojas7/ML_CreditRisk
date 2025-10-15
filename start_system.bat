@echo off
REM Script para iniciar tanto la API FastAPI como la aplicación Streamlit

echo ========================================
echo  Credit Risk Analysis System - Startup
echo ========================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python no está instalado o no está en el PATH
    pause
    exit /b 1
)

REM Verificar si las dependencias están instaladas
echo Verificando dependencias...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo Instalando dependencias...
    pip install -r requirements.txt
)

REM Crear directorios necesarios
if not exist "data\raw" mkdir data\raw
if not exist "data\interim" mkdir data\interim
if not exist "data\processed" mkdir data\processed
if not exist "data\external" mkdir data\external
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "reports\figures" mkdir reports\figures

echo.
echo Iniciando servicios...
echo.

REM Iniciar FastAPI en segundo plano
echo [1/2] Iniciando API FastAPI en puerto 8000...
start "FastAPI Server" cmd /c "uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

REM Esperar un momento para que la API inicie
timeout /t 5 /nobreak >nul

REM Verificar que la API esté funcionando
echo Verificando que la API esté funcionando...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo Esperando a que la API inicie...
    timeout /t 10 /nobreak >nul
)

REM Iniciar Streamlit
echo [2/2] Iniciando interfaz Streamlit en puerto 8501...
echo.
echo ========================================
echo  URLs del sistema:
echo  - API FastAPI: http://localhost:8000
echo  - Docs API: http://localhost:8000/docs  
echo  - Streamlit UI: http://localhost:8501
echo ========================================
echo.

cd frontend
streamlit run streamlit_app.py

pause