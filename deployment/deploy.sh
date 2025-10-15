#!/bin/bash

# Script de deployment para producción
# Uso: ./deploy.sh [production|staging|development]

set -e

ENVIRONMENT=${1:-development}
PROJECT_NAME="ml-credit-risk"
DOCKER_COMPOSE_FILE="deployment/docker-compose.yml"

echo "🚀 Desplegando $PROJECT_NAME en entorno: $ENVIRONMENT"

# Verificar que Docker esté ejecutándose
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker no está ejecutándose"
    exit 1
fi

# Verificar que docker-compose esté instalado
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose no está instalado"
    exit 1
fi

# Crear directorios necesarios si no existen
echo "📁 Creando directorios necesarios..."
mkdir -p data/{raw,interim,processed,external}
mkdir -p models
mkdir -p logs
mkdir -p reports/figures

# Configurar variables de entorno según el ambiente
case $ENVIRONMENT in
    production)
        echo "🔧 Configurando para producción..."
        export API_DEBUG=false
        export LOG_LEVEL=WARNING
        ;;
    staging)
        echo "🔧 Configurando para staging..."
        export API_DEBUG=false
        export LOG_LEVEL=INFO
        ;;
    development)
        echo "🔧 Configurando para desarrollo..."
        export API_DEBUG=true
        export LOG_LEVEL=DEBUG
        ;;
    *)
        echo "❌ Entorno no válido: $ENVIRONMENT"
        echo "Uso: $0 [production|staging|development]"
        exit 1
        ;;
esac

# Construir imágenes
echo "🔨 Construyendo imágenes Docker..."
docker-compose -f $DOCKER_COMPOSE_FILE build

# Detener contenedores existentes
echo "🛑 Deteniendo contenedores existentes..."
docker-compose -f $DOCKER_COMPOSE_FILE down

# Iniciar servicios
echo "▶️ Iniciando servicios..."
docker-compose -f $DOCKER_COMPOSE_FILE up -d

# Esperar a que los servicios estén listos
echo "⏳ Esperando a que los servicios estén listos..."
sleep 30

# Verificar que la API esté funcionando
echo "🔍 Verificando estado de la API..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API está funcionando correctamente"
else
    echo "❌ Error: API no responde"
    echo "📋 Logs del contenedor:"
    docker-compose -f $DOCKER_COMPOSE_FILE logs credit-risk-api
    exit 1
fi

# Mostrar estado de los contenedores
echo "📊 Estado de los contenedores:"
docker-compose -f $DOCKER_COMPOSE_FILE ps

echo ""
echo "🎉 Deployment completado exitosamente!"
echo "📍 API disponible en: http://localhost:8000"
echo "📚 Documentación: http://localhost:8000/docs"
echo "🔍 Health check: http://localhost:8000/health"
echo ""
echo "📋 Comandos útiles:"
echo "  Ver logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
echo "  Detener: docker-compose -f $DOCKER_COMPOSE_FILE down"
echo "  Reiniciar: docker-compose -f $DOCKER_COMPOSE_FILE restart"