# 07_Reporte_Hiperparametros.py

import os
import pandas as pd
import joblib
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import config as cfg

# --- 1. REPLICAR DEFINICIONES DE HIPERMODELOS (Necesario para leer Keras Tuner) ---
# Estas funciones deben ser IDÉNTICAS a las de 02_Modelado_y_Baseline.py
# para que Keras Tuner pueda recargar el proyecto correctamente.

def build_lstm_hypermodel(hp):
    model = Sequential()
    hp_units_1 = hp.Int('units_layer_1', min_value=32, max_value=128, step=32)
    model.add(LSTM(units=hp_units_1, input_shape=cfg.INPUT_SHAPE, return_sequences=True))
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_1))
    hp_units_2 = hp.Int('units_layer_2', min_value=16, max_value=64, step=16)
    model.add(LSTM(units=hp_units_2))
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_2))
    model.add(Dense(cfg.N_TARGETS))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse')
    return model

def build_gru_hypermodel(hp):
    model = Sequential()
    hp_units_1 = hp.Int('units_layer_1', min_value=32, max_value=128, step=32)
    model.add(GRU(units=hp_units_1, input_shape=cfg.INPUT_SHAPE, return_sequences=True))
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_1))
    hp_units_2 = hp.Int('units_layer_2', min_value=16, max_value=64, step=16)
    model.add(GRU(units=hp_units_2))
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_2))
    model.add(Dense(cfg.N_TARGETS))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse')
    return model

def build_cnn_hypermodel(hp):
    model = Sequential()
    hp_filters_1 = hp.Int('filters_layer_1', min_value=32, max_value=128, step=32)
    model.add(Conv1D(filters=hp_filters_1, kernel_size=3, activation='relu', input_shape=cfg.INPUT_SHAPE))
    hp_filters_2 = hp.Int('filters_layer_2', min_value=16, max_value=64, step=16)
    model.add(Conv1D(filters=hp_filters_2, kernel_size=3, activation='relu'))
    model.add(Flatten())
    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=100, step=16)
    model.add(Dense(units=hp_dense_units, activation='relu'))
    model.add(Dense(cfg.N_TARGETS))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse')
    return model

# --- 2. EXTRACCIÓN DE DATOS ---

def get_dl_best_params(task, model_name, build_fn):
    """Carga el Tuner existente y extrae los mejores hiperparámetros."""
    try:
        tuner = kt.BayesianOptimization(
            build_fn,
            objective='val_loss',
            max_trials=cfg.DL_TUNER_TRIALS,
            directory=cfg.KERAS_TUNER_DIR,
            project_name=f'{model_name.lower()}_{task}',
            overwrite=False # ¡IMPORTANTE! No sobrescribir, solo leer.
        )
        
        # Verificar si hay resultados
        if not os.path.exists(os.path.join(cfg.KERAS_TUNER_DIR, f'{model_name.lower()}_{task}')):
            return "No Tuning Data"

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hps.values # Retorna un diccionario
    except Exception as e:
        return f"Error: {str(e)}"

def get_filter_best_params(task, filter_type):
    """Carga el archivo .gz de Skopt."""
    filename = f'{filter_type.lower()}_best_params_{task}.gz'
    path = os.path.join(cfg.SKOPT_TUNER_DIR, filename)
    
    if os.path.exists(path):
        try:
            # Skopt guarda una lista/tupla [q_var, r_noise]
            params = joblib.load(path)
            return {'Q_var': params[0], 'R_noise': params[1]}
        except Exception as e:
            return f"Error lectura: {e}"
    else:
        return "No Tuning Data"

def main():
    print("--- GENERANDO REPORTE DE MEJORES HIPERPARÁMETROS ---")
    
    all_records = []

    for task in cfg.TASKS:
        print(f"Procesando tarea: {task}...")
        
        # 1. LSTM
        hps = get_dl_best_params(task, 'lstm', build_lstm_hypermodel)
        all_records.append({'Task': task, 'Model': 'LSTM', 'Hyperparameters': str(hps)})
        
        # 2. GRU
        hps = get_dl_best_params(task, 'gru', build_gru_hypermodel)
        all_records.append({'Task': task, 'Model': 'GRU', 'Hyperparameters': str(hps)})
        
        # 3. CNN
        hps = get_dl_best_params(task, 'cnn', build_cnn_hypermodel)
        all_records.append({'Task': task, 'Model': 'CNN', 'Hyperparameters': str(hps)})
        
        # 4. Kalman Filter
        hps = get_filter_best_params(task, 'kf')
        all_records.append({'Task': task, 'Model': 'Kalman Filter', 'Hyperparameters': str(hps)})
        
        # 5. UKF
        hps = get_filter_best_params(task, 'ukf')
        all_records.append({'Task': task, 'Model': 'UKF', 'Hyperparameters': str(hps)})

    # Crear DataFrame
    df = pd.DataFrame(all_records)
    
    # Formatear para visualización bonita en consola
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + "="*80)
    print("RESUMEN DE HIPERPARÁMETROS ÓPTIMOS")
    print("="*80)
    print(df)
    
    # Guardar CSV
    output_file = os.path.join(cfg.CSV_OUT_DIR, 'HYPERPARAMETERS_TABLE.csv')
    df.to_csv(output_file, index=False)
    print(f"\nTabla guardada exitosamente en: {output_file}")

if __name__ == "__main__":
    main()