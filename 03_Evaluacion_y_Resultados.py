import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import config as cfg

def inverse_transform_predictions(scaled_data, scaler):
    n_samples = len(scaled_data)
    temp_array = np.zeros((n_samples, cfg.N_FEATURES))
    
    temp_array[:, cfg.TARGET_INDICES_IN_FEATURES] = scaled_data
    
    inversed_array = scaler.inverse_transform(temp_array)
    
    return inversed_array[:, cfg.TARGET_INDICES_IN_FEATURES]

def run_lstm_kf_hybrid(lstm_predictions_inv):
    print("--- Generando: Modelo Híbrido LSTM-KF ---")
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

def calculate_mean_squared_jerk(trajectory, dt):
    if len(trajectory) < 4: return 0
    velocity = np.diff(trajectory, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    msj_per_axis = np.mean(jerk**2, axis=0)
    return np.mean(msj_per_axis)

def plot_trajectory_comparison_2d(y_true, predictions_dict, dim, title, filename):
    dim_names = ['Posición X', 'Posición Y', 'Posición Z']
    start=100; end=300
    plt.figure(figsize=(15, 7))
    plt.plot(y_true[start:end, dim], label='Verdad (Ground Truth)', color='black', linewidth=2.5, linestyle='--')
    for name, preds in predictions_dict.items():
        plt.plot(preds[start:end, dim], label=name, alpha=0.8, linewidth=1.5)
    plt.title(f'{title} ({dim_names[dim]})'); plt.xlabel('Paso de Tiempo'); plt.ylabel('Posición')
    plt.legend(loc='upper right'); plt.grid(True, linestyle=':')
    plt.savefig(filename); print(f"Gráfico 2D guardado en: {filename}"); plt.close()

def plot_trajectory_3d_all_models(y_true, predictions_dict, title, filename):
    start = 100; end = 300
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(y_true[start:end, 0], y_true[start:end, 1], y_true[start:end, 2], 
            label='Verdad (Ground Truth)', color='black', linewidth=2.5, linestyle='--')
    for name, preds in predictions_dict.items():
         ax.plot(preds[start:end, 0], preds[start:end, 1], preds[start:end, 2], 
                 label=name, alpha=0.7, linewidth=1)
    ax.set_title(title); ax.set_xlabel('Posición X'); ax.set_ylabel('Posición Y'); ax.set_zlabel('Posición Z')
    ax.legend(loc='upper right', prop={'size': 8})
    plt.savefig(filename); print(f"Gráfico 3D (Todos) guardado en: {filename}"); plt.close()

def animate_trajectory_3d(y_true, predictions_dict, title, filename):
    print(f"\nGenerando animación 3D para {title}...")
    start = 100; end = 300
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(y_true[start:end, 0], y_true[start:end, 1], y_true[start:end, 2], 
            label='Verdad (Ground Truth)', color='black', linewidth=2.5, linestyle='--')
    for name, preds in predictions_dict.items():
         ax.plot(preds[start:end, 0], preds[start:end, 1], preds[start:end, 2], 
                 label=name, alpha=0.7, linewidth=1)
    ax.set_title(title); ax.set_xlabel('Posición X'); ax.set_ylabel('Posición Y'); ax.set_zlabel('Posición Z')
    ax.legend(loc='upper right', prop={'size': 8})
    def animate(i):
        ax.view_init(elev=20, azim=i*2); return fig,
    ani = animation.FuncAnimation(fig, animate, frames=180, interval=50, blit=True)
    try:
        ani.save(filename, writer='pillow', fps=15)
        print(f"Animación 3D (GIF) guardada en: {filename}")
    except Exception as e:
        print(f"Error al guardar el GIF: {e}. Asegúrate de tener 'pillow' instalado (pip install pillow)")
    plt.close()


def main():
    print(f"--- EJECUTANDO SCRIPT: 03_Evaluacion_y_Resultados (MODO ORGANIZADO) ---")
    
    for task in cfg.TASKS:
        print(f"\n=== Procesando Tarea: {task} ===")
        scaler_path = cfg.get_scaler_path(task)
        results_path = cfg.get_results_path(task)

        try:
            scaler = joblib.load(scaler_path)
            results = np.load(results_path)
        except FileNotFoundError:
            continue

        # 1. Des-normalizar TO DO el conjunto
        y_test_scaled = results['y_test']
        y_test_inv_full = inverse_transform_predictions(y_test_scaled, scaler)
        
        predictions_inv_full = {}
        for name, preds_scaled in results.items():
            if name == 'y_test': continue
            clean_name = name.replace('preds_', '').replace('_', ' ').upper()

            if clean_name == 'AR': clean_name = 'AR_Model (MA)' # Ajustes estéticos
            if clean_name == 'UKF': clean_name = 'UKF_Baseline'
            
            predictions_inv_full[clean_name] = inverse_transform_predictions(preds_scaled, scaler)
        
        predictions_inv_full['LSTM-KF_Hybrid'] = run_lstm_kf_hybrid(predictions_inv_full['CNN']) #Nota: Usar mejor modelo aquí
        
        for arm_name, arm_info in cfg.ARM_MAPPING.items():
            arm_indices = arm_info['indices']
            
            # Validar si hay datos para este brazo (por si solo entrenaste con 1)
            if max(arm_indices) >= y_test_inv_full.shape[1]:
                print(f"Saltando {arm_name}: Índices fuera de rango (¿Entrenaste solo 1 brazo?)")
                continue

            print(f"\n   >>> Generando resultados para: {arm_name} <<<")

            # A. CREAR CARPETAS ESPECÍFICAS
            path_plots_arm = os.path.join(cfg.PLOT_DIR, task, arm_name)
            path_csv_arm = os.path.join(cfg.CSV_OUT_DIR, task, arm_name)
            os.makedirs(path_plots_arm, exist_ok=True)
            os.makedirs(path_csv_arm, exist_ok=True)

            # B. FILTRAR DATOS SOLO DE ESTE BRAZO
            y_true_arm = y_test_inv_full[:, arm_indices]
            preds_arm = {k: v[:, arm_indices] for k, v in predictions_inv_full.items()}

            # C. CALCULAR MÉTRICAS (SOLO DE ESTE BRAZO)
            print("-" * 70)
            print(f"{'Modelo (' + arm_name + ')':<25} | {'RMSE':<10} | {'MAE':<10} | {'MSJ':<12}")
            print("-" * 70)
            
            msj_real = calculate_mean_squared_jerk(y_true_arm, cfg.DT)
            print(f"{'VERDAD REAL':<25} | {'N/A':<10} | {'N/A':<10} | {msj_real:<12.4f}")

            # Ordenar modelos por RMSE en este brazo
            sorted_models = sorted(preds_arm.items(), key=lambda i: np.sqrt(mean_squared_error(y_true_arm, i[1])))

            row_data_csv = [] # Para guardar un resumen de métricas

            for name, p_data in sorted_models:
                rmse = np.sqrt(mean_squared_error(y_true_arm, p_data))
                mae = mean_absolute_error(y_true_arm, p_data)
                msj = calculate_mean_squared_jerk(p_data, cfg.DT)
                print(f"{name:<25} | {rmse:<10.4f} | {mae:<10.4f} | {msj:<12.4f}")
                row_data_csv.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'MSJ': msj})

            # D. GENERAR GRÁFICOS (Guardados en la carpeta del brazo)
            print(f"      Generando gráficos en: {path_plots_arm}...")
            
            # Gráficos 2D por dimensión (X, Y, Z relativos al brazo)
            dims_labels = ['X', 'Y', 'Z']
            for local_dim in range(3):
                title = f'{task} - {arm_name} - Eje {dims_labels[local_dim]}'
                fname = os.path.join(path_plots_arm, f'2D_Comparison_{dims_labels[local_dim]}.png')
                plot_trajectory_comparison_2d(y_true_arm, preds_arm, local_dim, title, fname)

            # Gráfico 3D
            fname_3d = os.path.join(path_plots_arm, f'3D_Trajectory_All.png')
            plot_trajectory_3d_all_models(y_true_arm, preds_arm, f'{task} - {arm_name} (3D)', fname_3d)

            # E. GUARDAR CSVs (Predicciones crudas solo de este brazo)
            # Crear un DataFrame maestro con Real + Predicciones
            df_arm = pd.DataFrame(y_true_arm, columns=['Real_X', 'Real_Y', 'Real_Z'])
            
            for name, p_data in preds_arm.items():
                suffix = name.replace(' ', '_')
                cols = [f'{suffix}_X', f'{suffix}_Y', f'{suffix}_Z']
                df_temp = pd.DataFrame(p_data, columns=cols)
                df_arm = pd.concat([df_arm, df_temp], axis=1)
            
            csv_file = os.path.join(path_csv_arm, f'Predictions_{task}_{arm_name}.csv')
            df_arm.to_csv(csv_file, index=False)
            print(f"      CSV guardado: {csv_file}")
            
            # Guardar tabla de métricas
            pd.DataFrame(row_data_csv).to_csv(os.path.join(path_csv_arm, f'Metrics_Summary_{task}_{arm_name}.csv'), index=False)

    print(f"\n--- PROCESO TERMINADO ---")
    
if __name__ == "__main__":
    main()