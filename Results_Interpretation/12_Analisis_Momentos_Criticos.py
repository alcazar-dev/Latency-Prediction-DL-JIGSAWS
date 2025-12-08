import pandas as pd
import numpy as np
import os
import config as cfg

# Configuración
WINDOW_SECONDS = 1.0  # Tamaño de la ventana para considerar un "momento" (en segundos)
EXPERT_VIDEOS = [
    'Suturing_E001',
    'Knot_Tying_E004',
    'Needle_Passing_D005'
]
ARMS = ['Right_PSM1', 'Left_PSM2']

def format_time(seconds):
    """Convierte segundos a formato MM:SS.ms"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 100)
    return f"{m:02d}:{s:02d}.{ms:02d}"

def analyze_timeframes(df, dt):
    """
    Analiza las trayectorias para encontrar los momentos de mejor y peor desempeño.
    """
    results = []
    
    # Identificar modelos (columnas que terminan en _X y no son Real)
    model_names = [c.replace('_X', '') for c in df.columns if c.endswith('_X') and 'Real' not in c]
    
    # Obtener trayectoria Real 3D
    real_traj = df[['Real_X', 'Real_Y', 'Real_Z']].values
    
    window_size = int(WINDOW_SECONDS / dt)
    
    for model in model_names:
        # Obtener trayectoria Predicha 3D
        pred_cols = [f"{model}_X", f"{model}_Y", f"{model}_Z"]
        # Verificar que existan las columnas
        if not all(col in df.columns for col in pred_cols):
            continue
            
        pred_traj = df[pred_cols].values
        
        # 1. Calcular Error Euclidiano Instantáneo (fotograma a fotograma)
        # Distancia = raiz((x1-x2)^2 + ...)
        instant_error = np.linalg.norm(real_traj - pred_traj, axis=1)
        
        # 2. Suavizar con media móvil para encontrar "Momentos" sostenidos
        # Usamos pandas rolling para facilitar la ventana
        rolling_error = pd.Series(instant_error).rolling(window=window_size, center=True).mean()
        
        # Descartar los bordes (NaN por la ventana)
        valid_indices = rolling_error.dropna().index
        
        if len(valid_indices) == 0:
            continue

        # 3. Encontrar índices de Mejor y Peor desempeño
        # idxmin devuelve el índice del valor mínimo
        best_idx = rolling_error.idxmin()
        worst_idx = rolling_error.idxmax()
        
        # Obtener los valores de error en esos momentos
        min_err_val = rolling_error[best_idx]
        max_err_val = rolling_error[worst_idx]
        
        # 4. Calcular Timeframes (Inicio - Fin de la ventana)
        # El índice devuelto es el centro de la ventana (por center=True) o el final
        # Ajustamos para tener el rango [start, end]
        half_win = window_size // 2
        
        # Tiempo central en segundos
        t_best_sec = best_idx * dt
        t_worst_sec = worst_idx * dt
        
        # Rango de frames
        f_best_start = max(0, best_idx - half_win)
        f_best_end = min(len(df), best_idx + half_win)
        
        f_worst_start = max(0, worst_idx - half_win)
        f_worst_end = min(len(df), worst_idx + half_win)
        
        results.append({
            'Model': model,
            'Best_Error_mm': min_err_val,
            'Best_Time_Center': format_time(t_best_sec),
            'Best_Frames': f"{f_best_start}-{f_best_end}",
            'Worst_Error_mm': max_err_val,
            'Worst_Time_Center': format_time(t_worst_sec),
            'Worst_Frames': f"{f_worst_start}-{f_worst_end}"
        })
        
    return pd.DataFrame(results)

def main():
    print("--- ANÁLISIS DE MOMENTOS CRÍTICOS (Timeframes) ---")
    print(f"Ventana de análisis: {WINDOW_SECONDS} segundos")
    
    all_results = []
    
    for video_id in EXPERT_VIDEOS:
        print(f"\nProcesando Video: {video_id}...")
        
        for arm in ARMS:
            csv_path = os.path.join(cfg.CSV_SPECIFIC_DIR, video_id, arm, 'trajectories.csv')
            
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            
            # Analizar este archivo
            df_moments = analyze_timeframes(df, cfg.DT)
            
            if not df_moments.empty:
                df_moments['Video'] = video_id
                df_moments['Arm'] = arm
                all_results.append(df_moments)
                
                # Imprimir un adelanto en consola
                print(f"   Brazo: {arm}")
                for _, row in df_moments.iterrows():
                    print(f"      [{row['Model']}]")
                    print(f"         MEJOR: {row['Best_Time_Center']} (Err: {row['Best_Error_mm']:.4f} mm)")
                    print(f"         PEOR:  {row['Worst_Time_Center']} (Err: {row['Worst_Error_mm']:.4f} mm)")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Reordenar columnas para que sea legible
        cols = ['Video', 'Arm', 'Model', 'Best_Time_Center', 'Best_Error_mm', 'Worst_Time_Center', 'Worst_Error_mm', 'Best_Frames', 'Worst_Frames']
        final_df = final_df[cols]
        
        out_path = os.path.join(cfg.CSV_OUT_DIR, 'CRITICAL_MOMENTS_TIMEFRAMES.csv')
        final_df.to_csv(out_path, index=False)
        print(f"\nReporte completo guardado en: {out_path}")
        print("Usa los tiempos 'Best_Time_Center' y 'Worst_Time_Center' para buscar los momentos en tus videos.")

if __name__ == "__main__":
    main()