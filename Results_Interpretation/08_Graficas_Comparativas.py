# 08_Graficas_Comparativas.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config as cfg

def main():
    print("--- EJECUTANDO: 08_Graficas_Comparativas ---")
    
    # 1. Cargar los datos maestros (generados por el script 06)
    master_file = os.path.join(cfg.CSV_OUT_DIR, 'MASTER_METRICS_ALL_VIDEOS.csv')
    
    if not os.path.exists(master_file):
        print(f"ERROR: No se encontró {master_file}")
        print("Por favor, ejecuta primero '06_Analisis_Estadistico.py' para recolectar los datos.")
        return

    df = pd.read_csv(master_file)
    print(f"Datos cargados: {len(df)} registros.")

    # Configuración de estilo para Tesis (Fondo blanco, letras grandes)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    
    # Colores distintivos por modelo (consistentes con tus videos)
    # Ajusta según tus preferencias
    palette_colors = {
        'LSTM': 'red', 
        'GRU': 'blue', 
        'CNN': 'gold', 
        'KALMAN': 'magenta', 
        'UKF_Baseline': 'orange',
        'LSTM-KF_Hybrid': 'cyan',
        'AR_Model (MA)': 'gray'
    }
    # Fallback si hay nombres diferentes
    palette = sns.color_palette("bright") 

    # --- GRÁFICO 1: PRECISIÓN (RMSE) ---
    plt.figure(figsize=(12, 6))
    
    # Boxplot: Muestra la distribución del error
    sns.boxplot(
        data=df, 
        x='Model', 
        y='RMSE', 
        hue='Arm', 
        palette="Blues", # O usa palette_colors si ajustas los nombres exactos
        showfliers=False # Ocultar outliers extremos para ver mejor la caja
    )
    
    plt.title('Comparación de Precisión (RMSE) por Brazo', fontweight='bold')
    plt.ylabel('Error RMSE (mm)')
    plt.xlabel('Modelo')
    plt.xticks(rotation=15)
    plt.legend(title='Brazo')
    
    # Guardar
    out_rmse = os.path.join(cfg.PLOT_DIR, 'Tesis_Boxplot_RMSE.png')
    plt.tight_layout()
    plt.savefig(out_rmse, dpi=300)
    print(f"Gráfico guardado: {out_rmse}")
    plt.close()

    # --- GRÁFICO 2: SUAVIDAD (MSJ) ---
    plt.figure(figsize=(12, 6))
    
    # Filtramos AR_Model si distorsiona mucho el gráfico (suele tener 0 jerk o mucho error)
    df_msj = df[df['Model'] != 'AR_Model (MA)']

    sns.boxplot(
        data=df_msj, 
        x='Model', 
        y='MSJ', 
        hue='Arm', 
        palette="Greens", 
        showfliers=False
    )
    
    # Línea de referencia humana (aprox 523)
    plt.axhline(y=523, color='r', linestyle='--', label='Humano (Ref. ~523)')
    
    plt.title('Comparación de Suavidad (MSJ - Jerk)', fontweight='bold')
    plt.ylabel('MSJ (Adimensional)')
    plt.xlabel('Modelo')
    plt.xticks(rotation=15)
    plt.legend()
    
    # Guardar
    out_msj = os.path.join(cfg.PLOT_DIR, 'Tesis_Boxplot_MSJ.png')
    plt.tight_layout()
    plt.savefig(out_msj, dpi=300)
    print(f"Gráfico guardado: {out_msj}")
    plt.close()

    # --- GRÁFICO 3: RANKING DE MODELOS (Barplot Resumen) ---
    # Calcular promedios para un gráfico de barras simple
    summary = df.groupby(['Model', 'Arm'])['RMSE'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary,
        x='RMSE',
        y='Model',
        hue='Arm',
        palette="viridis"
    )
    
    plt.title('Ranking de Modelos (Menor Error es Mejor)', fontweight='bold')
    plt.xlabel('RMSE Promedio (mm)')
    plt.ylabel('')
    
    out_rank = os.path.join(cfg.PLOT_DIR, 'Tesis_Ranking_Modelos.png')
    plt.tight_layout()
    plt.savefig(out_rank, dpi=300)
    print(f"Gráfico guardado: {out_rank}")
    plt.close()

if __name__ == "__main__":
    main()