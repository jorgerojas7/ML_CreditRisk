# -*- coding: utf-8 -*-
"""
Script para descargar y procesar datos raw para el análisis de riesgo crediticio.
"""
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

def download_data(source_url: Optional[str] = None, output_path: str = "data/raw/") -> None:
    """
    Descarga datos desde una fuente externa.
    
    Args:
        source_url: URL de los datos (si aplicable)
        output_path: Directorio donde guardar los datos raw
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando descarga de datos...")
    
    # TODO: Implementar descarga de datos desde fuente específica
    # Ejemplo:
    # df = pd.read_csv(source_url)
    # df.to_csv(os.path.join(output_path, "raw_data.csv"), index=False)
    
    logger.info("Descarga completada")

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Carga datos raw desde archivo.
    
    Args:
        file_path: Ruta al archivo de datos
        
    Returns:
        DataFrame con los datos cargados
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Cargando datos desde {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        raise

def validate_data(df: pd.DataFrame) -> bool:
    """
    Valida la integridad básica de los datos.
    
    Args:
        df: DataFrame a validar
        
    Returns:
        True si los datos son válidos, False en caso contrario
    """
    logger = logging.getLogger(__name__)
    
    # Validaciones básicas
    if df.empty:
        logger.error("DataFrame está vacío")
        return False
        
    # TODO: Agregar validaciones específicas para el dataset de crédito
    # Ejemplos:
    # - Verificar columnas requeridas
    # - Validar rangos de valores
    # - Verificar tipos de datos
    
    logger.info("Validación de datos completada exitosamente")
    return True

def main():
    """Función principal para crear el dataset"""
    logger = logging.getLogger(__name__)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Definir rutas
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_path = project_dir / "data" / "raw"
    
    # Crear directorio si no existe
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Iniciando proceso de creación de dataset")
    
    # TODO: Implementar lógica específica de descarga y procesamiento
    # download_data(output_path=str(raw_data_path))
    
    logger.info("Dataset creado exitosamente")

if __name__ == '__main__':
    main()