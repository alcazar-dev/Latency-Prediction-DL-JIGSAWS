import numpy as np
import pandas as pd
import os
import glob
import joblib
import config as cfg

def calculate_msj(trajectory, dt):
    """
    Calcula el Mean Squared Jerk (MSJ) usando la tercera derivada discreta.
    Fórmula: Jerk = (p_{t+1} - 3p_t + 3p_{t-1} - p_{t-2}) / dt^3
    """
    # Usamos np.diff con n=3 para la tercera derivada
    # axis=0 asume que el tiempo avanza en las filas
    if len(trajectory) < 4:
        return 0.0
    
    # Diferencia de tercer orden (aceleración de la aceleración)
    # Jerk * dt^3
    diff_3 = np.diff(trajectory, n=3, axis=0)
    
    # Dividimos por dt^3 para obtener el Jerk real
    jerk = diff_3 / (dt ** 3)
    
    # Elevamos al cuadrado
    squared_jerk = jerk ** 2
    
    # Promediamos sobre el tiempo y luego sobre las dimensiones (x, y, z)
    msj = np.mean(squared_jerk)
    return msj

def load_expert_files(task):
    """Identifica los archivos de expertos basándose en el meta_file."""
    meta_file = cfg.TASK_PATHS[task]['meta_file']
    kinematics_dir = cfg.TASK_PATHS[task]['kinematics_dir']
    
    # Cargar puntajes
    try:
        # Asumiendo formato: ID score ...
        meta_df = pd.read_csv(meta_file, sep=r'\s+', header=None, usecols=[0, 2], names=['id', 'score'])
        # Calcular umbral de experto (Q3)
        threshold = meta_df['score'].quantile(0.75)
        experts = meta_df[meta_df['score'] >= threshold]['id'].tolist()
        print(f"   [INFO] Tarea {task}: Umbral Experto (Q3) >= {threshold:.2f}. Expertos encontrados: {len(experts)}")
    except Exception as e:
        print(f"   [ERROR] No se pudo leer meta_file para {task}: {e}")
        return []

    # Buscar los archivos correspondientes
    expert_files = []
    for video_id in experts:
        # El nombre del archivo suele ser VideoID.txt
        file_path = os.path.join(kinematics_dir, f"{video_id}.txt")
        if os.path.exists(file_path):
            expert_files.append(file_path)
    
    return expert_files

def main():
    print("--- CÁLCULO DEL MSJ BIOLÓGICO (Línea Base Humana) ---")
    print(f"Sampling Rate: {cfg.SAMPLING_RATE_HZ} Hz | dt: {cfg.DT:.4f} s")
    
    results = []

    for task in cfg.TASKS:
        print(f"\nProcesando Tarea: {task}...")
        
        # 1. Cargar el Scaler (¡CRÍTICO!)
        # Debemos usar el MISMO escalado que los modelos para que el Jerk sea comparable
        scaler_path = cfg.get_scaler_path(task)
        if not os.path.exists(scaler_path):
            print(f"   [SKIP] No se encontró scaler para {task}. Ejecuta 01 primero.")
            continue
            
        scaler = joblib.load(scaler_path)
        
        # 2. Obtener lista de archivos de EXPERTOS
        files = load_expert_files(task)
        if not files:
            continue

        msj_values = []
        
        for f in files:
            try:
                # Leer cinemática cruda
                df = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
                raw_data = df.iloc[:, cfg.FEATURE_COLUMNS_INDICES].values
                
                # Normalizar (Transformar al espacio latente del modelo [0, 1])
                scaled_data = scaler.transform(raw_data)
                
                # Extraer solo las columnas TARGET (Posiciones PSM)
                # Usamos TARGET_INDICES_IN_FEATURES porque scaled_data solo tiene las columnas features
                target_data = scaled_data[:, cfg.TARGET_INDICES_IN_FEATURES]
                
                # Calcular MSJ para este video
                val = calculate_msj(target_data, cfg.DT)
                msj_values.append(val)
                
            except Exception as e:
                print(f"   Error procesando {os.path.basename(f)}: {e}")

        if msj_values:
            mean_msj = np.mean(msj_values)
            std_msj = np.std(msj_values)
            print(f"   >>> MSJ PROMEDIO (Human Baseline): {mean_msj:.4f} (+/- {std_msj:.4f})")
            results.append({'Task': task, 'Baseline_MSJ': mean_msj, 'Std': std_msj})
        else:
            print("   No se pudieron calcular valores.")

    print("\n" + "="*50)
    print("RESUMEN FINAL (Human Baseline MSJ)")
    print("="*50)
    df_res = pd.DataFrame(results)
    print(df_res)
    
    # Guardar para referencia
    df_res.to_csv(os.path.join(cfg.CSV_OUT_DIR, 'HUMAN_BASELINE_MSJ.csv'), index=False)

if __name__ == "__main__":
    main()