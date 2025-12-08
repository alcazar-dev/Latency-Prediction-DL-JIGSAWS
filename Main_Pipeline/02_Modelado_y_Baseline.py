import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from statsmodels.tsa.ar_model import AutoReg
from math import sqrt

import config as cfg

# --- 1. Definición de HIPER-Modelos (para KerasTuner) ---

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

# --- 2. Definición de Modelos Baseline ---
def run_kalman_filter_predictor(X_data, horizon_steps, q_var, r_noise):
    observed_positions = X_data[:, -1, cfg.TARGET_INDICES_IN_FEATURES]
    n_samples, n_dims = observed_positions.shape
    predictions = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0., 0.]); kf.F = np.array([[1., cfg.DT], [0., 1.]]); kf.H = np.array([[1., 0.]])
        kf.R = np.eye(1) * r_noise; kf.Q = Q_discrete_white_noise(dim=2, dt=cfg.DT, var=q_var); kf.P = np.eye(2) * 1.
        dim_preds = []
        F_k = np.linalg.matrix_power(kf.F, horizon_steps)
        for i in range(n_samples):
            z = observed_positions[i, dim]
            kf.predict(); kf.update(z)
            x_predicted_future = F_k @ kf.x
            dim_preds.append(x_predicted_future[0])
        predictions[:, dim] = np.array(dim_preds)
    return predictions

def f_ca(x, dt): F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]]); return F @ x
def h_ca(x): return np.array([x[0]])

def run_ukf_predictor(X_data, horizon_steps, q_var, r_noise):
    observed_positions = X_data[:, -1, cfg.TARGET_INDICES_IN_FEATURES]
    n_samples, n_dims = observed_positions.shape
    predictions = np.zeros((n_samples, n_dims))
    points = MerweScaledSigmaPoints(n=3, alpha=cfg.UKF_ALPHA, beta=cfg.UKF_BETA, kappa=cfg.UKF_KAPPA)
    for dim in range(n_dims):
        ukf = UnscentedKalmanFilter(dim_x=3, dim_z=1, dt=cfg.DT, hx=h_ca, fx=f_ca, points=points)
        ukf.x = np.array([0., 0., 0.]); ukf.R = np.eye(1) * r_noise; ukf.Q = Q_discrete_white_noise(dim=3, dt=cfg.DT, var=q_var); ukf.P = np.eye(3) * 1.
        dim_preds = []
        F_k = np.linalg.matrix_power(np.array([[1, cfg.DT, 0.5*cfg.DT**2], [0, 1, cfg.DT], [0, 0, 1]]), horizon_steps)
        for i in range(n_samples):
            z = observed_positions[i, dim]
            ukf.predict(); ukf.update(z)
            x_predicted_future = F_k @ ukf.x
            dim_preds.append(x_predicted_future[0])
        predictions[:, dim] = np.array(dim_preds)
    return predictions

# --- 3. Funciones de Sintonización de Filtros ---
search_space_filters = [
    Real(cfg.FILTER_Q_SPACE[0], cfg.FILTER_Q_SPACE[1], cfg.FILTER_Q_SPACE[2], name='q_var'),
    Real(cfg.FILTER_R_SPACE[0], cfg.FILTER_R_SPACE[1], cfg.FILTER_R_SPACE[2], name='r_noise')
]
_X_val_global = None; _y_val_global = None

def objective_function_kf(params):
    q_var, r_noise = params[0], params[1]
    preds = run_kalman_filter_predictor(_X_val_global, cfg.PREDICTION_STEPS, q_var, r_noise)
    return sqrt(mean_squared_error(_y_val_global, preds))

def objective_function_ukf(params):
    q_var, r_noise = params[0], params[1]
    preds = run_ukf_predictor(_X_val_global, cfg.PREDICTION_STEPS, q_var, r_noise)
    return sqrt(mean_squared_error(_y_val_global, preds))

# --- 4. Modelo AR ---
def run_autoregression_model(X_train_data, X_test_data, lookback, horizon_steps):
    print("\n--- Evaluando: Regresión de Media Móvil (AR) ---")
    train_positions = X_train_data[:, -1, cfg.TARGET_INDICES_IN_FEATURES]
    test_positions = X_test_data[:, -1, cfg.TARGET_INDICES_IN_FEATURES]
    n_samples_test, n_dims = test_positions.shape
    predictions = np.zeros((n_samples_test, n_dims))
    for dim in range(n_dims):
        print(f"  Entrenando AR para dimensión {dim+1}/{n_dims}...")
        ar_model = AutoReg(train_positions[:, dim], lags=lookback)
        ar_fit = ar_model.fit()
        dim_preds = []
        for i in range(n_samples_test):
            history = X_test_data[i, :, dim]
            pred = ar_fit.predict(start=len(history), end=len(history) + horizon_steps - 1)
            dim_preds.append(pred[-1])
        predictions[:, dim] = np.array(dim_preds)
    return predictions

# 02_Modelado_y_Baseline.py
# --- 5. Ejecución Principal (AJUSTE FINO) ---
def main():
    print(f"--- EJECUTANDO SCRIPT: 02_Modelado_y_Baseline ---")
    
    global _X_val_global, _y_val_global
    callbacks_dl = [EarlyStopping(monitor='val_loss', patience=cfg.PATIENCE)]
    
    for task in cfg.TASKS:
        print(f"\n--- Procesando Tarea: {task} ---")
        
        # 1. Cargar datos de PRE-ENTRENAMIENTO (Todos)
        data_path = cfg.get_processed_data_path(task)
        try:
            data = np.load(data_path)
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        except FileNotFoundError:
            print(f"Error: No se encontró {data_path}. Ejecuta 01_Preprocesamiento.py primero.")
            continue
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Advertencia: No hay datos de entrenamiento/prueba para {task}. Saltando.")
            continue

        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=cfg.VALIDATION_SIZE, shuffle=False
        )
        _X_val_global = X_val; _y_val_global = y_val
        predictions = {}
        
        # 2. Cargar datos de AJUSTE FINO (Expertos)
        X_train_finetune, y_train_finetune = None, None
        if cfg.USE_FINE_TUNING:
            try:
                finetune_data_path = cfg.get_processed_data_path(task, finetune=True)
                data_finetune = np.load(finetune_data_path)
                X_train_finetune = data_finetune['X_train_finetune']
                y_train_finetune = data_finetune['y_train_finetune']
                if len(X_train_finetune) > 0:
                    print(f"Datos de Ajuste Fino cargados: {X_train_finetune.shape}")
                else:
                    print("Advertencia: Archivo de Ajuste Fino vacío. Desactivando.")
                    cfg.USE_FINE_TUNING = False
            except FileNotFoundError:
                print(f"Advertencia: No se encontró {finetune_data_path}. Desactivando Ajuste Fino para {task}.")
                cfg.USE_FINE_TUNING = False
        
        # --- 3. Sintonización de Filtros (usando datos 'Todos') ---
        print(f"\nSintonizando KF para {task} (usando datos 'Todos')...")
        result_kf = gp_minimize(
            func=objective_function_kf, 
            dimensions=search_space_filters, 
            n_calls=cfg.FILTER_TUNER_TRIALS, 
            n_initial_points=cfg.FILTER_TUNER_N_INITIAL_POINTS, 
            random_state=42, n_jobs=-1
        )
        best_q_kf, best_r_kf = result_kf.x
        print(f"Mejores HPs para KF: Q_var={best_q_kf:.6f}, R_noise={best_r_kf:.6f} (RMSE: {result_kf.fun:.6f})")
        joblib.dump(result_kf.x, os.path.join(cfg.SKOPT_TUNER_DIR, f'kf_best_params_{task}.gz'))

        print(f"\nSintonizando UKF para {task} (usando datos 'Todos')...")
        result_ukf = gp_minimize(
            func=objective_function_ukf, 
            dimensions=search_space_filters, 
            n_calls=cfg.FILTER_TUNER_TRIALS, 
            n_initial_points=cfg.FILTER_TUNER_N_INITIAL_POINTS, 
            random_state=42, n_jobs=-1
        )
        best_q_ukf, best_r_ukf = result_ukf.x
        print(f"Mejores HPs para UKF: Q_var={best_q_ukf:.6f}, R_noise={best_r_ukf:.6f} (RMSE: {result_ukf.fun:.6f})")
        joblib.dump(result_ukf.x, os.path.join(cfg.SKOPT_TUNER_DIR, f'ukf_best_params_{task}.gz'))

        # --- 4. Sintonización y Entrenamiento de Modelos DL ---
        # 1. Sintonizar y Entrenar LSTM
        print(f"\nSintonizando LSTM para {task} (usando datos 'Todos')...")
        tuner_lstm = kt.BayesianOptimization(
            build_lstm_hypermodel, objective='val_loss', max_trials=cfg.DL_TUNER_TRIALS,
            directory=cfg.KERAS_TUNER_DIR, project_name=f'lstm_{task}', overwrite=True
        )
        tuner_lstm.search(X_train_main, y_train_main, epochs=cfg.EPOCHS_TUNING, validation_data=(X_val, y_val), callbacks=callbacks_dl, verbose=1)
        
        best_hps_lstm = tuner_lstm.get_best_hyperparameters(num_trials=1)[0]
        print(f"Mejores HPs para LSTM: {best_hps_lstm.values}")
        
        print("--- Fase 1: Pre-entrenando modelo final de LSTM (Datos 'Todos')... ---")
        model_lstm = tuner_lstm.hypermodel.build(best_hps_lstm)
        model_lstm.fit(X_train, y_train, epochs=cfg.EPOCHS_FINAL, validation_split=0.2, callbacks=callbacks_dl, verbose=0)
        
        if cfg.USE_FINE_TUNING and X_train_finetune is not None:
            print("--- Fase 2: Ajustando modelo final de LSTM (Datos 'Expertos')... ---")
            # Reducir la tasa de aprendizaje para el ajuste fino
            old_lr = best_hps_lstm.get('learning_rate')
            new_lr = old_lr * cfg.FINE_TUNING_LEARNING_RATE_MULTIPLIER
            model_lstm.optimizer.learning_rate.assign(new_lr)
            print(f"  Tasa de aprendizaje reducida de {old_lr} a {new_lr}")
            
            # Entrenar de nuevo (finetune)
            model_lstm.fit(X_train_finetune, y_train_finetune, epochs=cfg.EPOCHS_FINETUNE, validation_split=0.2, callbacks=callbacks_dl, verbose=0)
        
        model_lstm.save(os.path.join(cfg.MODEL_DIR, f'lstm_model_{task}.h5'))
        print("Modelo LSTM final guardado.")
        
        # 2. Sintonizar y Entrenar GRU
        print(f"\nSintonizando GRU para {task} (usando datos 'Todos')...")
        tuner_gru = kt.BayesianOptimization(
            build_gru_hypermodel, objective='val_loss', max_trials=cfg.DL_TUNER_TRIALS,
            directory=cfg.KERAS_TUNER_DIR, project_name=f'gru_{task}', overwrite=True
        )
        tuner_gru.search(X_train_main, y_train_main, epochs=cfg.EPOCHS_TUNING, validation_data=(X_val, y_val), callbacks=callbacks_dl, verbose=1)
        
        best_hps_gru = tuner_gru.get_best_hyperparameters(num_trials=1)[0]
        print(f"Mejores HPs para GRU: {best_hps_gru.values}")

        print("--- Fase 1: Pre-entrenando modelo final de GRU (Datos 'Todos')... ---")
        model_gru = tuner_gru.hypermodel.build(best_hps_gru)
        model_gru.fit(X_train, y_train, epochs=cfg.EPOCHS_FINAL, validation_split=0.2, callbacks=callbacks_dl, verbose=0)
        
        if cfg.USE_FINE_TUNING and X_train_finetune is not None:
            print("--- Fase 2: Ajustando modelo final de GRU (Datos 'Expertos')... ---")
            old_lr = best_hps_gru.get('learning_rate')
            new_lr = old_lr * cfg.FINE_TUNING_LEARNING_RATE_MULTIPLIER
            model_gru.optimizer.learning_rate.assign(new_lr)
            print(f"  Tasa de aprendizaje reducida de {old_lr} a {new_lr}")
            model_gru.fit(X_train_finetune, y_train_finetune, epochs=cfg.EPOCHS_FINETUNE, validation_split=0.2, callbacks=callbacks_dl, verbose=0)
        
        model_gru.save(os.path.join(cfg.MODEL_DIR, f'gru_model_{task}.h5'))
        print("Modelo GRU final guardado.")

        # 3. Sintonizar y Entrenar CNN
        print(f"\nSintonizando CNN para {task} (usando datos 'Todos')...")
        tuner_cnn = kt.BayesianOptimization(
            build_cnn_hypermodel, objective='val_loss', max_trials=cfg.DL_TUNER_TRIALS,
            directory=cfg.KERAS_TUNER_DIR, project_name=f'cnn_{task}', overwrite=True
        )
        tuner_cnn.search(X_train_main, y_train_main, epochs=cfg.EPOCHS_TUNING, validation_data=(X_val, y_val), callbacks=callbacks_dl, verbose=1)
        
        best_hps_cnn = tuner_cnn.get_best_hyperparameters(num_trials=1)[0]
        print(f"Mejores HPs para CNN: {best_hps_cnn.values}")
        
        print("--- Fase 1: Pre-entrenando modelo final de CNN (Datos 'Todos')... ---")
        model_cnn = tuner_cnn.hypermodel.build(best_hps_cnn)
        model_cnn.fit(X_train, y_train, epochs=cfg.EPOCHS_FINAL, validation_split=0.2, callbacks=callbacks_dl, verbose=0)
        
        if cfg.USE_FINE_TUNING and X_train_finetune is not None:
            print("--- Fase 2: Ajustando modelo final de CNN (Datos 'Expertos')... ---")
            old_lr = best_hps_cnn.get('learning_rate')
            new_lr = old_lr * cfg.FINE_TUNING_LEARNING_RATE_MULTIPLIER
            model_cnn.optimizer.learning_rate.assign(new_lr)
            print(f"  Tasa de aprendizaje reducida de {old_lr} a {new_lr}")
            model_cnn.fit(X_train_finetune, y_train_finetune, epochs=cfg.EPOCHS_FINETUNE, validation_split=0.2, callbacks=callbacks_dl, verbose=0)

        model_cnn.save(os.path.join(cfg.MODEL_DIR, f'cnn_1d_model_{task}.h5'))
        print("Modelo CNN 1D final guardado.")
        
        # --- 5. Ejecución Final en Set de Pruebas (usando datos 'Todos') ---
        print("\nEjecutando predicciones finales sobre el conjunto de prueba (Test Set)...")
        predictions['preds_lstm'] = model_lstm.predict(X_test)
        predictions['preds_gru'] = model_gru.predict(X_test)
        predictions['preds_cnn'] = model_cnn.predict(X_test)
        
        predictions['preds_kalman'] = run_kalman_filter_predictor(X_test, cfg.PREDICTION_STEPS, q_var=best_q_kf, r_noise=best_r_kf)
        predictions['preds_ukf'] = run_ukf_predictor(X_test, cfg.PREDICTION_STEPS, q_var=best_q_ukf, r_noise=best_r_ukf)
        predictions['preds_ar'] = run_autoregression_model(X_train_main, X_test, cfg.LOOKBACK_STEPS, cfg.PREDICTION_STEPS)
        
        # 6. Guardar todos los resultados
        results_path = cfg.get_results_path(task)
        np.savez_compressed(
            results_path,
            y_test=y_test,
            **predictions
        )
        print(f"\nTodas las predicciones para {task} guardadas en {results_path}")

    print(f"\n--- 02_Modelado_y_Baseline COMPLETO (Tareas: {cfg.TASKS}) --- :^)")

if __name__ == "__main__":
    main()