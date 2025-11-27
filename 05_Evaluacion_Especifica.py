# 05_Evaluacion_Especifica.py

import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from statsmodels.tsa.ar_model import AutoReg

import config as cfg

# --- 1. FUNCIONES AUXILIARES ---

def create_windowed_sequences(data, lookback, horizon, target_indices):
    # Ventanas Deslizantes 
    X, y = [], []
    target_data = data[:, target_indices]
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
        y.append(target_data[i + lookback + horizon - 1, :])
    return np.array(X), np.array(y)

def inverse_transform_predictions(scaled_data, scaler):

    n_samples = len(scaled_data)
    # Matriz temporal llena de ceros con la forma original (N_FEATURES)
    temp_array = np.zeros((n_samples, cfg.N_FEATURES))
    
    # Colocamos las predicciones en las columnas correspondientes al ROBOT
    temp_array[:, cfg.TARGET_INDICES_IN_FEATURES] = scaled_data
    
    # Invertimos la transformación
    inversed_array = scaler.inverse_transform(temp_array)
    
    # Devolvemos solo las columnas del target
    return inversed_array[:, cfg.TARGET_INDICES_IN_FEATURES]

def calculate_mean_squared_jerk(trajectory, dt):
    # Calcula Jerk (Suavidad)
    if len(trajectory) < 4: return 0
    velocity = np.diff(trajectory, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    msj_per_axis = np.mean(jerk**2, axis=0)
    return np.mean(msj_per_axis)

# --- 2. MODELOS BASELINE Y HÍBRIDOS ---

def run_lstm_kf_hybrid(lstm_predictions_inv):
    n_samples, n_dims = lstm_predictions_inv.shape
    kf_predictions = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([lstm_predictions_inv[0, dim], 0.])
        kf.F = np.array([[1., cfg.DT], [0., 1.]])
        kf.H = np.array([[1., 0.]])
        kf.R = np.eye(1) * cfg.HYBRID_KF_R_NOISE
        kf.Q = Q_discrete_white_noise(dim=2, dt=cfg.DT, var=cfg.HYBRID_KF_Q_VAR)
        kf.P = np.eye(2) * 1.      
        dim_preds = []
        for i in range(n_samples):
            z = lstm_predictions_inv[i, dim]
            kf.predict(); kf.update(z)
            dim_preds.append(kf.x[0])
        kf_predictions[:, dim] = np.array(dim_preds)
    return kf_predictions

def run_kalman_filter_predictor(X_data, horizon_steps, q_var, r_noise):
    observed_positions = X_data[:, -1, cfg.TARGET_INDICES_IN_FEATURES]
    n_samples, n_dims = observed_positions.shape
    predictions = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0., 0.]) 
        kf.F = np.array([[1., cfg.DT], [0., 1.]])
        kf.H = np.array([[1., 0.]])
        kf.R = np.eye(1) * r_noise
        kf.Q = Q_discrete_white_noise(dim=2, dt=cfg.DT, var=q_var)
        kf.P = np.eye(2) * 1.
                
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
        ukf.x = np.array([0., 0., 0.])
        ukf.R = np.eye(1) * r_noise
        ukf.Q = Q_discrete_white_noise(dim=3, dt=cfg.DT, var=q_var)
        ukf.P = np.eye(3) * 1.
        dim_preds = []
        F_k = np.linalg.matrix_power(np.array([[1, cfg.DT, 0.5*cfg.DT**2], [0, 1, cfg.DT], [0, 0, 1]]), horizon_steps)
        for i in range(n_samples):
            z = observed_positions[i, dim]
            ukf.predict(); ukf.update(z)
            x_predicted_future = F_k @ ukf.x
            dim_preds.append(x_predicted_future[0])
        predictions[:, dim] = np.array(dim_preds)
    return predictions

def run_autoregression_model(X_train_data, X_test_data, lookback, horizon_steps):
    train_positions = X_train_data[:, -1, cfg.TARGET_INDICES_IN_FEATURES]
    test_positions = X_test_data[:, -1, cfg.TARGET_INDICES_IN_FEATURES]
    n_samples_test, n_dims = test_positions.shape
    predictions = np.zeros((n_samples_test, n_dims))
    for dim in range(n_dims):

        try:
            ar_model = AutoReg(train_positions[:, dim], lags=lookback)
            ar_fit = ar_model.fit()
            dim_preds = []
            for i in range(n_samples_test):
                history = X_test_data[i, :, dim]
                pred = ar_fit.predict(start=len(history), end=len(history) + horizon_steps - 1)
                dim_preds.append(pred[-1])
            predictions[:, dim] = np.array(dim_preds)
        except:
            predictions[:, dim] = np.zeros(n_samples_test) # Fallback
    return predictions

# --- 3. FUNCIONES DE GRAFICADO ---

def plot_trajectory_comparison_2d(y_true, predictions_dict, dim, title, filename):
    # Mapeo de dimensión a nombre (X, Y, Z)
    # Si dim=0 -> X, dim=1 -> Y, dim=2 -> Z
    axis_labels = ['X', 'Y', 'Z']
    label_axis = axis_labels[dim] if dim < 3 else f"Dim {dim}"
    
    start=100; end=300
    if len(y_true) < end: start=0; end=len(y_true)
    
    plt.figure(figsize=(15, 7))
    plt.plot(y_true[start:end, dim], label='Verdad Real', color='black', linewidth=2.5, linestyle='--')
    
    for name, preds in predictions_dict.items():
        plt.plot(preds[start:end, dim], label=name, alpha=0.8, linewidth=1.5)
    
    plt.title(f'{title} (Eje {label_axis})')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel(f'Posición {label_axis}')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_trajectory_3d_all_models(y_true, predictions_dict, title, filename):
    start = 100; end = 300
    if len(y_true) < end: start=0; end=len(y_true)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Asumir que las primeras 3 col son X,Y,Z
    ax.plot(y_true[start:end, 0], y_true[start:end, 1], y_true[start:end, 2], 
            label='Verdad Real', color='black', linewidth=2.5, linestyle='--')
    
    for name, preds in predictions_dict.items():
         ax.plot(preds[start:end, 0], preds[start:end, 1], preds[start:end, 2], 
                 label=name, alpha=0.7, linewidth=1)
                 
    ax.set_title(title)
    ax.set_xlabel('Posición X')
    ax.set_ylabel('Posición Y')
    ax.set_zlabel('Posición Z')
    ax.legend(loc='upper right', prop={'size': 8})
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- 4. SCRIPT PRINCIPAL ---

def main():
    print(f"--- EJECUTANDO SCRIPT: 05_Evaluacion_Especifica (ORGANIZADA POR BRAZO) ---")

    if not cfg.VISUALIZATION_MAPPING:
        print("No se seleccionaron videos en config.py.")
        return

    # Bucle sobre Tareas
    for task in cfg.TASKS:
        print(f"\n=== Cargando recursos generales para Tarea: {task} ===")
        try:
            scaler_path = cfg.get_scaler_path(task)
            scaler = joblib.load(scaler_path)
            
            models_dl = {
                'LSTM': load_model(os.path.join(cfg.MODEL_DIR, f'lstm_model_{task}.h5'), compile=False),
                'GRU': load_model(os.path.join(cfg.MODEL_DIR, f'gru_model_{task}.h5'), compile=False),
                'CNN': load_model(os.path.join(cfg.MODEL_DIR, f'cnn_1d_model_{task}.h5'), compile=False)
            }
            
            # Cargar parametros de filtros
            best_q_kf, best_r_kf = joblib.load(os.path.join(cfg.SKOPT_TUNER_DIR, f'kf_best_params_{task}.gz'))
            best_q_ukf, best_r_ukf = joblib.load(os.path.join(cfg.SKOPT_TUNER_DIR, f'ukf_best_params_{task}.gz'))
            
            # Cargar train data para AR
            data_train_path = cfg.get_processed_data_path(task)
            data_train = np.load(data_train_path)
            X_train_general = data_train['X_train']

        except Exception as e:
            print(f"Error cargando recursos para {task}: {e}")
            continue

        # Bucle sobre VIDEOS
        for (v_task, v_skill, v_id), v_info in cfg.VISUALIZATION_MAPPING.items():
            if v_task != task: continue 
                
            print(f"\n>>> Evaluando Video: {v_id} ({v_skill}) <<<")
            
            # 1. Cargar Datos Específicos
            kin_path = os.path.join(cfg.TASK_PATHS[task]['kinematics_dir'], v_info['kinematics'])
            try:
                kin_df = pd.read_csv(kin_path, sep=r'\s+', header=None, engine='python')
                kin_data_raw = kin_df.iloc[:, cfg.FEATURE_COLUMNS_INDICES].values
            except Exception as e:
                print(f"Error leyendo {kin_path}. Saltando.")
                continue

            # 2. Preprocesar
            data_scaled = scaler.transform(kin_data_raw)
            X_test, y_test = create_windowed_sequences(
                data_scaled, cfg.LOOKBACK_STEPS, cfg.PREDICTION_STEPS, cfg.TARGET_INDICES_IN_FEATURES
            )
            
            if len(X_test) == 0: continue

            # 3. Generar Predicciones GLOBALES (Todos los brazos juntos)
            predictions_scaled = {}
            for name, model in models_dl.items():
                predictions_scaled[f'preds_{name.lower()}'] = model.predict(X_test, verbose=0)
            
            predictions_scaled['preds_kalman'] = run_kalman_filter_predictor(X_test, cfg.PREDICTION_STEPS, best_q_kf, best_r_kf)
            predictions_scaled['preds_ukf'] = run_ukf_predictor(X_test, cfg.PREDICTION_STEPS, best_q_ukf, best_r_ukf)
            predictions_scaled['preds_ar'] = run_autoregression_model(X_train_general, X_test, cfg.LOOKBACK_STEPS, cfg.PREDICTION_STEPS)

            # 4. Des-normalizar GLOBALES
            # Aquí usamos la función corregida inverse_transform_predictions
            y_test_inv_full = inverse_transform_predictions(y_test, scaler)
            
            predictions_inv_full = {}
            for name, preds_scaled in predictions_scaled.items():
                clean_name = name.replace('preds_', '').replace('_', ' ').upper()
                if clean_name == 'AR': clean_name = 'AR_Model (MA)'
                if clean_name == 'UKF': clean_name = 'UKF_Baseline'
                predictions_inv_full[clean_name] = inverse_transform_predictions(preds_scaled, scaler)
            
            # Híbrido (sobre la mejor red neuronal, ej. CNN o LSTM)
            predictions_inv_full['LSTM-KF_Hybrid'] = run_lstm_kf_hybrid(predictions_inv_full['LSTM'])

            # 5. SEPARACIÓN Y ANÁLISIS POR BRAZO (Usando config.ARM_MAPPING)
            
            for arm_name, arm_info in cfg.ARM_MAPPING.items():
                indices = arm_info['indices']
                
                if max(indices) >= y_test_inv_full.shape[1]:
                    continue

                print(f"   > Procesando Brazo: {arm_name}")

                # A. Crear subcarpetas específicas
                # output/plots_specific_videos/VideoID/Brazo/
                path_plot_arm = os.path.join(cfg.PLOT_SPECIFIC_DIR, v_id, arm_name)
                path_csv_arm = os.path.join(cfg.CSV_SPECIFIC_DIR, v_id, arm_name)
                os.makedirs(path_plot_arm, exist_ok=True)
                os.makedirs(path_csv_arm, exist_ok=True)

                # B. Filtrar datos del brazo actual
                y_true_arm = y_test_inv_full[:, indices]
                preds_arm = {k: v[:, indices] for k, v in predictions_inv_full.items()}

                # C. Calcular Métricas Específicas
                print("-" * 70)
                print(f"{f'MODELOS ({arm_name})':<20} | {'RMSE':<10} | {'MAE':<10} | {'MSJ':<12}")
                print("-" * 70)
                
                msj_real = calculate_mean_squared_jerk(y_true_arm, cfg.DT)
                print(f"{'VERDAD REAL':<20} | {'N/A':<10} | {'N/A':<10} | {msj_real:<12.4f}")

                # Ordenar modelos por RMSE
                sorted_models = sorted(preds_arm.items(), key=lambda i: np.sqrt(mean_squared_error(y_true_arm, i[1])))
                
                metrics_list = []

                for name, p_data in sorted_models:
                    rmse = np.sqrt(mean_squared_error(y_true_arm, p_data))
                    mae = mean_absolute_error(y_true_arm, p_data)
                    msj = calculate_mean_squared_jerk(p_data, cfg.DT)
                    print(f"{name:<20} | {rmse:<10.4f} | {mae:<10.4f} | {msj:<12.4f}")
                    metrics_list.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'MSJ': msj})

                # D. Generar Gráficos X, Y, Z (Automático)
                base_title = f'{v_id} - {arm_name} - {cfg.PREDICTION_HORIZON_MS}ms'
                
                # Graficar dimensión 0 (X), 1 (Y), 2 (Z) LOCALMENTE para este brazo
                dims_labels = ['X', 'Y', 'Z']
                for local_dim in range(3):
                    # El archivo se llamará: comparacion_2D_Video_Brazo_X.png
                    fname_2d = os.path.join(path_plot_arm, f'2D_{dims_labels[local_dim]}.png')
                    plot_trajectory_comparison_2d(y_true_arm, preds_arm, local_dim, base_title, fname_2d)
                
                # Grafico 3D
                fname_3d = os.path.join(path_plot_arm, f'3D_Trajectory.png')
                plot_trajectory_3d_all_models(y_true_arm, preds_arm, f'{base_title} (3D)', fname_3d)

                # E. Guardar CSVs
                # CSV de Métricas
                pd.DataFrame(metrics_list).to_csv(os.path.join(path_csv_arm, 'metrics.csv'), index=False)
                
                # CSV de Trayectorias
                df_traj = pd.DataFrame(y_true_arm, columns=['Real_X', 'Real_Y', 'Real_Z'])
                for name, p_data in preds_arm.items():
                    safe_name = name.replace(' ', '_')
                    cols = [f'{safe_name}_X', f'{safe_name}_Y', f'{safe_name}_Z']
                    df_traj = pd.concat([df_traj, pd.DataFrame(p_data, columns=cols)], axis=1)
                
                df_traj.to_csv(os.path.join(path_csv_arm, 'trajectories.csv'), index=False)
                print(f"   > Resultados guardados en: {path_csv_arm}")

    print(f"\n--- 05_Evaluacion_Especifica COMPLETO --- :^)")

if __name__ == "__main__":
    main()