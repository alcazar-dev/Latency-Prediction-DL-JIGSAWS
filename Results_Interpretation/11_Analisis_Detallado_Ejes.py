import pandas as pd
import numpy as np
import os
import config as cfg

# Videos "Expertos" específicos para el análisis
EXPERT_VIDEOS = [
    'Suturing_E001',
    'Knot_Tying_E004',
    'Needle_Passing_D005'
]

ARMS = ['Right_PSM1', 'Left_PSM2']
AXES = ['X', 'Y', 'Z']

def calculate_axis_metrics(real, pred, dt):
    """
    Calcula métricas detalladas para un solo eje (1D array).
    """
    # 1. Diferencia de Distancia (Error de Posición)
    # MAE: Error Absoluto Medio (más interpretables que RMSE para "distancia promedio")
    pos_diff = np.abs(real - pred)
    mae_pos = np.mean(pos_diff)
    
    # 2. Razón de Cambio (Velocidad)
    # Usamos gradiente simple (diferencia finita)
    vel_real = np.gradient(real, dt)
    vel_pred = np.gradient(pred, dt)
    
    # Diferencia en Velocidad (Magnitud)
    vel_diff = np.abs(vel_real - vel_pred)
    mae_vel = np.mean(vel_diff)
    
    # Similitud en Velocidad (Correlación de Pearson)
    # Nos dice si la "forma" de la velocidad es igual (aceleran al mismo tiempo)
    # aunque la magnitud varíe un poco.
    if np.std(vel_real) == 0 or np.std(vel_pred) == 0:
        corr_vel = 0.0 # Evitar división por cero si no hay movimiento
    else:
        corr_vel = np.corrcoef(vel_real, vel_pred)[0, 1]
        
    return mae_pos, mae_vel, corr_vel

def main():
    print("--- ANÁLISIS DETALLADO POR EJE (Expertos) ---")
    print(f"Videos a analizar: {EXPERT_VIDEOS}")
    
    results = []

    for video_id in EXPERT_VIDEOS:
        print(f"\nProcesando Video: {video_id}...")
        
        for arm in ARMS:
            # Ruta al archivo de trayectorias generado en el script 05
            csv_path = os.path.join(cfg.CSV_SPECIFIC_DIR, video_id, arm, 'trajectories.csv')
            
            if not os.path.exists(csv_path):
                print(f"   [AVISO] No se encontró {csv_path}. Saltando.")
                continue
                
            print(f"   -> Analizando brazo: {arm}")
            df = pd.read_csv(csv_path)
            
            # Identificar modelos basados en las columnas
            # Las columnas suelen ser: Real_X, Real_Y, Real_Z, Modelo1_X, ...
            # Filtramos columnas que terminan en _X para sacar los nombres de modelos
            model_cols = [c.replace('_X', '') for c in df.columns if c.endswith('_X') and 'Real' not in c]
            
            for model in model_cols:
                for axis in AXES:
                    # Nombres de columnas en el CSV
                    real_col = f"Real_{axis}"
                    pred_col = f"{model}_{axis}"
                    
                    if real_col not in df.columns or pred_col not in df.columns:
                        continue
                        
                    real_data = df[real_col].values
                    pred_data = df[pred_col].values
                    
                    # Calcular métricas
                    mae_pos, mae_vel, corr_vel = calculate_axis_metrics(real_data, pred_data, cfg.DT)
                    
                    results.append({
                        'Video': video_id,
                        'Arm': arm,
                        'Model': model,
                        'Axis': axis,
                        'Dist_Diff_mm': mae_pos,   # Diferencia Promedio Posición
                        'Rate_Diff_mm_s': mae_vel, # Diferencia Promedio Velocidad
                        'Rate_Similarity': corr_vel # Correlación (1.0 es perfecto)
                    })

    # Crear DataFrame y Guardar
    if not results:
        print("No se generaron resultados.")
        return

    df_res = pd.DataFrame(results)
    
    # Guardar CSV crudo
    out_path = os.path.join(cfg.CSV_OUT_DIR, 'DETAILED_AXIS_ANALYSIS.csv')
    df_res.to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")
    
    # --- RESUMEN DE INTERPRETACIÓN ---
    # Vamos a imprimir un resumen promedio para ver qué eje es más difícil
    print("\n" + "="*60)
    print("RESUMEN PROMEDIO POR EJE (¿Dónde fallan más los modelos?)")
    print("="*60)
    
    # Agrupamos por Eje y Modelo para ver tendencias globales en estos expertos
    summary_axis = df_res.groupby(['Axis', 'Model'])[['Dist_Diff_mm', 'Rate_Similarity']].mean()
    print(summary_axis)
    
    print("\n" + "="*60)
    print("RESUMEN POR MODELO (Promedio de todos los ejes y videos)")
    print("="*60)
    summary_model = df_res.groupby(['Model'])[['Dist_Diff_mm', 'Rate_Diff_mm_s', 'Rate_Similarity']].mean().sort_values('Dist_Diff_mm')
    print(summary_model)

if __name__ == "__main__":
    main()