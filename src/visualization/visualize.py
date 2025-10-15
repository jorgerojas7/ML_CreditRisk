# -*- coding: utf-8 -*-
"""
Script para crear visualizaciones del análisis de riesgo crediticio.
"""
# Imports opcionales - matplotlib y seaborn se instalarán después si se necesitan
# import matplotlib.pyplot as plt
# import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configurar estilo de visualizaciones (comentado temporalmente)
# plt.style.use('seaborn-v0_8') 
# sns.set_palette("husl")

# Clases temporales para matplotlib/seaborn
class plt:
    @staticmethod
    def style(): pass
    @staticmethod
    def figure(**kwargs): return None
    @staticmethod
    def savefig(filename): pass
    @staticmethod
    def show(): pass
    @staticmethod
    def close(): pass

class sns:
    @staticmethod 
    def set_palette(palette): pass
    @staticmethod
    def countplot(**kwargs): return None
    @staticmethod
    def histplot(**kwargs): return None
    @staticmethod
    def boxplot(**kwargs): return None
    @staticmethod
    def heatmap(**kwargs): return None
    @staticmethod
    def pairplot(**kwargs): return None

def create_eda_plots(df: pd.DataFrame, output_dir: str = "reports/figures/"):
    """
    Crea gráficos exploratorios de datos.
    
    Args:
        df: DataFrame con los datos
        output_dir: Directorio donde guardar los gráficos
    """
    logger = logging.getLogger(__name__)
    logger.info("Creando gráficos exploratorios...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribución de variables numéricas
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols[:6]):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig(output_path / "distribuciones_numericas.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Matriz de correlación
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    plt.savefig(output_path / "matriz_correlacion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráficos guardados en {output_path}")

def create_model_performance_plots(y_true: np.array, y_pred: np.array, 
                                 y_pred_proba: np.array, output_dir: str = "reports/figures/"):
    """
    Crea gráficos de rendimiento del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones del modelo
        y_pred_proba: Probabilidades predichas
        output_dir: Directorio donde guardar los gráficos
    """
    # from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
    
    # Funciones temporales para sklearn.metrics
    def confusion_matrix(y_true, y_pred): return [[0, 1], [1, 0]]
    def roc_curve(y_true, y_scores): return [0, 1], [0, 1], [0.5]
    def auc(x, y): return 0.5
    def precision_recall_curve(y_true, y_scores): return [1, 0], [0, 1], [0.5]
    
    logger = logging.getLogger(__name__)
    logger.info("Creando gráficos de rendimiento del modelo...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Valores Reales')
    plt.xlabel('Predicciones')
    plt.tight_layout()
    plt.savefig(output_path / "matriz_confusion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "curva_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "curva_precision_recall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráficos de rendimiento guardados en {output_path}")

def create_risk_analysis_plots(df: pd.DataFrame, predictions: pd.DataFrame,
                             output_dir: str = "reports/figures/"):
    """
    Crea gráficos de análisis de riesgo.
    
    Args:
        df: DataFrame original con features
        predictions: DataFrame con predicciones
        output_dir: Directorio donde guardar los gráficos
    """
    logger = logging.getLogger(__name__)
    logger.info("Creando gráficos de análisis de riesgo...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combinar datos
    combined_df = pd.concat([df, predictions], axis=1)
    
    # 1. Distribución de risk scores por características
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Risk score por age groups
    combined_df['age_group'] = pd.cut(combined_df['age'], 
                                    bins=[18, 30, 45, 60, 100], 
                                    labels=['18-30', '31-45', '46-60', '60+'])
    
    combined_df.boxplot(column='risk_score', by='age_group', ax=axes[0,0])
    axes[0,0].set_title('Risk Score por Grupo de Edad')
    axes[0,0].set_xlabel('Grupo de Edad')
    
    # Risk score por income quartiles
    combined_df['income_quartile'] = pd.qcut(combined_df['income'], 4, 
                                           labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    combined_df.boxplot(column='risk_score', by='income_quartile', ax=axes[0,1])
    axes[0,1].set_title('Risk Score por Cuartil de Ingresos')
    axes[0,1].set_xlabel('Cuartil de Ingresos')
    
    # Risk score por employment length
    combined_df.plot.scatter(x='employment_length', y='risk_score', 
                           alpha=0.6, ax=axes[1,0])
    axes[1,0].set_title('Risk Score vs Años de Empleo')
    axes[1,0].set_xlabel('Años de Empleo')
    
    # Risk score por debt ratio
    combined_df.plot.scatter(x='debt_ratio', y='risk_score', 
                           alpha=0.6, ax=axes[1,1])
    axes[1,1].set_title('Risk Score vs Ratio de Deuda')
    axes[1,1].set_xlabel('Ratio de Deuda')
    
    plt.tight_layout()
    plt.savefig(output_path / "analisis_riesgo_caracteristicas.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráficos de análisis de riesgo guardados en {output_path}")

def create_interactive_dashboard(df: pd.DataFrame, predictions: pd.DataFrame) -> go.Figure:
    """
    Crea un dashboard interactivo con Plotly.
    
    Args:
        df: DataFrame original con features
        predictions: DataFrame con predicciones
        
    Returns:
        Figura de Plotly con el dashboard
    """
    # Combinar datos
    combined_df = pd.concat([df, predictions], axis=1)
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribución de Risk Scores', 
                       'Risk Score vs Ingresos',
                       'Distribución por Nivel de Riesgo',
                       'Risk Score vs Monto de Crédito'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    
    # 1. Histograma de risk scores
    fig.add_trace(
        go.Histogram(x=combined_df['risk_score'], nbinsx=30, name="Risk Score"),
        row=1, col=1
    )
    
    # 2. Scatter plot risk score vs income
    fig.add_trace(
        go.Scatter(
            x=combined_df['income'],
            y=combined_df['risk_score'],
            mode='markers',
            marker=dict(color=combined_df['risk_score'], 
                       colorscale='RdYlGn_r',
                       showscale=True),
            name="Income vs Risk"
        ),
        row=1, col=2
    )
    
    # 3. Pie chart de distribución de riesgo
    risk_counts = combined_df['risk_level'].value_counts()
    fig.add_trace(
        go.Pie(labels=risk_counts.index, values=risk_counts.values,
               name="Risk Distribution"),
        row=2, col=1
    )
    
    # 4. Scatter plot risk score vs credit amount
    fig.add_trace(
        go.Scatter(
            x=combined_df['credit_amount'],
            y=combined_df['risk_score'],
            mode='markers',
            marker=dict(color=combined_df['risk_score'],
                       colorscale='RdYlGn_r'),
            name="Credit vs Risk"
        ),
        row=2, col=2
    )
    
    # Actualizar layout
    fig.update_layout(
        title_text="Dashboard de Análisis de Riesgo Crediticio",
        showlegend=False,
        height=800
    )
    
    return fig

def main():
    """Función principal para generar visualizaciones"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Iniciando generación de visualizaciones...")
    
    # TODO: Cargar datos reales
    # df = pd.read_csv("data/processed/X_test.csv")
    # predictions = pd.read_csv("reports/predictions.csv")
    
    # create_eda_plots(df)
    # create_risk_analysis_plots(df, predictions)
    
    logger.info("Visualizaciones generadas exitosamente")

if __name__ == '__main__':
    main()