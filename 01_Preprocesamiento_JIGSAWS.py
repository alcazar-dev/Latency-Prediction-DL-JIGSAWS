# 01_Preprocesamiento_JIGSAWS.py
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import joblib 

import config as cfg

def load_data_structure(kinematics_dir, meta_file_path, task_name):
    print(f"   -> Cargando metadatos desde: {meta_file_path}")
    
    # 1. Cargar Metafile (Puntajes)
    score_dict = {}
    try:
        meta_df = pd.read_csv(
            meta_file_path, 
            sep=r'\s+', 
            header=None, 
            usecols=[cfg.META_FILE_ID_COLUMN, cfg.META_FILE_SCORE_COLUMN],
            names=['id', 'score']
        )              
        meta_df['score'] = pd.to_numeric(meta_df['score'], errors='coerce')
        score_dict = meta_df.set_index('id')['score'].to_dict()
    except Exception as e:
        print(f"      [AVISO] No se pudo cargar metafile: {e}. Se asume NO Expertos.")
        cfg.USE_FINE_TUNING = False

    # 2. Listar archivos
    search_path = os.path.join(kinematics_dir, f"{task_name}_*.txt")
    all_files = glob.glob(search_path)
    all_files.sort()
    
    if not all_files:
        raise ValueError(f"No se encontraron archivos en: {search_path}")

    structured_data = []

    for f in all_files:
        try:
            # ID: 'Suturing_G002'
            file_id = os.path.basename(f).replace('.txt', '')
            parts = file_id.split('_')

            if len(parts) >= 2:
                subject_id = parts[-1][0] 
            else:
                subject_id = 'Unknown'

            # Leer CSV
            df = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
            
            # Recorte Debug
            if cfg.DEBUG_MODE and cfg.DEBUG_DATA_LIMIT:
                df = df.iloc[:cfg.DEBUG_DATA_LIMIT]

            # Determinar si es experto
            is_expert = False
            if file_id in score_dict:
                if score_dict[file_id] >= cfg.FINE_TUNING_SCORE_THRESHOLD:
                    is_expert = True
            
            structured_data.append({
                'id': file_id,
                'subject': subject_id,
                'is_expert': is_expert,
                'df': df
            })
            
        except Exception as e:
            print(f"      Error leyendo {f}: {e}")

    print(f"   -> Archivos cargados: {len(structured_data)}")
    return structured_data

def create_windowed_sequences(data, lookback, horizon, target_indices):
    # Ventana Deslizante
    if len(data) <= (lookback + horizon):
        return np.array([]), np.array([])
        
    X, y = [], []
    target_data = data[:, target_indices]
    
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
        y.append(target_data[i + lookback + horizon - 1, :])
        
    return np.array(X), np.array(y)

def process_video_batch(data_list, scaler):
    X_list, y_list = [], []
    
    for item in data_list:
        df = item['df']
        
        # 1. Extraer Features
        raw_values = df.iloc[:, cfg.FEATURE_COLUMNS_INDICES].values
        
        # 2. Transformar (Usando Scaler ya entrenado)
        scaled_values = scaler.transform(raw_values)
        
        # 3. Ventanear 
        X_vid, y_vid = create_windowed_sequences(
            scaled_values, 
            cfg.LOOKBACK_STEPS, 
            cfg.PREDICTION_STEPS, 
            cfg.TARGET_INDICES_IN_FEATURES
        )
        
        if len(X_vid) > 0:
            X_list.append(X_vid)
            y_list.append(y_vid)
            
    if not X_list:
        return np.array([]), np.array([])
        
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def main():
    print(f"--- 01_Preprocesamiento (CORREGIDO: Leave-One-User-Out & Windowing) ---")
    
    for task in cfg.TASKS:
        print(f"\n=== Procesando Tarea: {task} ===")
        
        try:
            # 1. Cargar Estructura de Datos (Sin concatenar aún)
            data_struct = load_data_structure(
                cfg.TASK_PATHS[task]['kinematics_dir'], 
                cfg.TASK_PATHS[task]['meta_file'], 
                task
            )
        except ValueError as e:
            print(e); continue

        # 2. DIVISIÓN POR SUJETOS (Leave-Subject-Out Strategy)
        all_subjects = sorted(list(set([d['subject'] for d in data_struct])))
        print(f"   -> Sujetos encontrados: {all_subjects}")
        
        # Split: 80% 20%
        split_idx = int(len(all_subjects) * (1 - cfg.TEST_SIZE))
        train_subjects = all_subjects[:split_idx]
        test_subjects = all_subjects[split_idx:]
        
        print(f"   -> Train Subjects: {train_subjects}")
        print(f"   -> Test Subjects:  {test_subjects}")
        
        train_list = [d for d in data_struct if d['subject'] in train_subjects]
        test_list = [d for d in data_struct if d['subject'] in test_subjects]
        
        # Separar Expertos (Solo del conjunto de entrenamiento para no hacer trampa)
        finetune_list = [d for d in train_list if d['is_expert']]
        
        print(f"   -> Archivos Train: {len(train_list)} | Archivos Test: {len(test_list)} | Expertos (en Train): {len(finetune_list)}")

        if len(train_list) == 0:
            print("Error: No hay datos de entrenamiento. Revisa los paths."); continue

        # 3. ENTRENAR SCALER (Solo con datos de Train)
        print("   -> Ajustando Scaler (Fit)...")
        temp_train_df = pd.concat([d['df'] for d in train_list], ignore_index=True)
        train_features_raw = temp_train_df.iloc[:, cfg.FEATURE_COLUMNS_INDICES].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_features_raw) # FIT solo en TRAIN
        
        joblib.dump(scaler, cfg.get_scaler_path(task))
        
        # 4. PROCESAR Y VENTANEAR (Batch processing)
        print("   -> Generando Tensores X, y...")
        
        # Train
        X_train, y_train = process_video_batch(train_list, scaler)
        
        # Test (Usando el scaler de train)
        X_test, y_test = process_video_batch(test_list, scaler)
        
        print(f"      Dimensiones Finales -> X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        # Guardar (Manteniendo formato para scripts siguientes)
        np.savez_compressed(
            cfg.get_processed_data_path(task),
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )

        # 5. PROCESAR EXPERTOS (Finetune)
        if cfg.USE_FINE_TUNING and finetune_list:
            print("   -> Generando Tensores Expertos (Finetune)...")
            X_ft, y_ft = process_video_batch(finetune_list, scaler)
            
            np.savez_compressed(
                cfg.get_processed_data_path(task, finetune=True),
                X_train_finetune=X_ft, y_train_finetune=y_ft
            )
            print(f"      Dimensiones Finetune -> X_ft: {X_ft.shape}")
        else:
            if cfg.USE_FINE_TUNING:
                print("      [AVISO] No hay expertos en el set de entrenamiento. Saltando Finetune.")

    print(f"\n--- PROCESO COMPLETADO EXITOSAMENTE ---")

if __name__ == "__main__":
    main()