# 06_Analisis_Estadistico.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg

def main():
    print("--- EJECUTANDO SCRIPT: 06_Analisis_Estadistico ---")
    print("Objetivo: Agrupar 'metrics.csv' de todos los videos y generar conclusiones globales.")

    # 1. Recolectar todos los metrics.csv
    all_metrics = []
    root_dir = cfg.CSV_SPECIFIC_DIR
    
    if not os.path.exists(root_dir):
        print(f"Error: No existe el directorio {root_dir}. Ejecuta primero el script 05.")
        return

    print(f"Buscando archivos en: {root_dir}...")
    
    # Recorrer carpetas: csv_specific/VideoID/ArmName/metrics.csv
    for video_id in os.listdir(root_dir):
        video_path = os.path.join(root_dir, video_id)
        if not os.path.isdir(video_path): continue
        
        for arm_name in os.listdir(video_path):
            arm_path = os.path.join(video_path, arm_name)
            if not os.path.isdir(arm_path): continue
            
            metrics_file = os.path.join(arm_path, 'metrics.csv')
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    # Añadir columnas de contexto
                    df['Video'] = video_id
                    df['Arm'] = arm_name
                    
                    # Intentar deducir Tarea/Skill del ID (ej. Suturing_G002)
                    parts = video_id.split('_')
                    if len(parts) >= 2:
                        df['Task'] = parts[0] # Suturing
                        # Skill suele ser la letra del ID (G002 -> G -> Novato/Intermedio/Experto según JIGSAWS)
                        # Pero para simplificar, usar el ID completo
                    
                    all_metrics.append(df)
                except Exception as e:
                    print(f"Error leyendo {metrics_file}: {e}")

    if not all_metrics:
        print("No se encontraron archivos metrics.csv. ¿Ejecutaste el script 05?")
        return

    # 2. Crear DataFrame Maestro
    master_df = pd.concat(all_metrics, ignore_index=True)
    print(f"\nDatos recolectados: {len(master_df)} filas totales.")
    
    # Guardar tabla maestra
    master_csv_path = os.path.join(cfg.CSV_OUT_DIR, 'MASTER_METRICS_ALL_VIDEOS.csv')
    master_df.to_csv(master_csv_path, index=False)
    print(f"Tabla Maestra guardada en: {master_csv_path}")

    # 3. Análisis Estadístico (Promedio y Desviación por Modelo y Brazo)
    # Agrupar por Brazo y Modelo para ver quién gana en cada mano
    summary = master_df.groupby(['Arm', 'Model'])[['RMSE', 'MAE', 'MSJ']].agg(['mean', 'std', 'min', 'max'])
    
    # Aplanar nombres de columnas para exportar
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    summary_path = os.path.join(cfg.CSV_OUT_DIR, 'STATISTICAL_SUMMARY.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nResumen Estadístico guardado en: {summary_path}")
    
    print("\n--- TOP 3 MODELOS POR BRAZO (Menor RMSE Promedio) ---")
    for arm in master_df['Arm'].unique():
        print(f"\n[{arm}]")
        arm_data = summary[summary['Arm'] == arm].sort_values('RMSE_mean')
        print(arm_data[['Model', 'RMSE_mean', 'RMSE_std', 'MSJ_mean']].head(3))

    # 4. Generación de Gráficos (Boxplots)
    
    sns.set_style("whitegrid")
    
    # A. Boxplot RMSE (Precisión)
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Model', y='RMSE', hue='Arm', data=master_df, palette="Set3", showfliers=False)
    plt.title('Distribución del Error de Posición (RMSE) por Modelo y Brazo')
    plt.ylabel('RMSE (mm) - Menor es mejor')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOT_DIR, 'Boxplot_RMSE_Global.png'))
    print("Gráfico generado: Boxplot_RMSE_Global.png")
    
    # B. Boxplot MSJ (Suavidad)
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Model', y='MSJ', hue='Arm', data=master_df, palette="Set2", showfliers=False)
    plt.title('Distribución de la Suavidad (MSJ) por Modelo y Brazo')
    plt.ylabel('MSJ (Jerk) - Menor es más suave')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOT_DIR, 'Boxplot_MSJ_Global.png'))
    print("Gráfico generado: Boxplot_MSJ_Global.png")

    print("\n--- 06_Analisis_Estadistico COMPLETO ---")

if __name__ == "__main__":
    main()