import cv2
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from statsmodels.tsa.ar_model import AutoReg

import config as cfg

# 1. FUNCIONES DE UTILIDAD (CÁMARA Y DATOS)

def load_camera_calibration(calib_file):
    """Carga parámetros de cámara o usa valores por defecto si no existe."""
    print(f"Intentando cargar calibración: {calib_file}...")
    try:
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            camera_matrix = fs.getNode("camera_matrix").mat()
            dist_coeffs = fs.getNode("distCoeffs").mat()
            fs.release()
            if camera_matrix is not None and dist_coeffs is not None:
                return camera_matrix, dist_coeffs
    except Exception:
        pass
    
    print("AVISO: No se encontró .yml o falló carga. Usando parámetros genéricos.")
    f = 600; w, h = 640, 480
    camera_matrix = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs

def project_3d_to_2d(points_3d, camera_matrix, dist_coeffs):
    # Panel Izquierdo 
    points_3d = np.asarray(points_3d, dtype=np.float32)
    points_to_project = points_3d.copy()
    points_to_project[:, 0] *= -1
    points_to_project[:, 1] *= -1

    if points_to_project.ndim == 1:
        points_to_project = points_to_project.reshape(1, 1, 3)
    else:
        points_to_project = points_to_project.reshape(-1, 1, 3)

    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    
    points_2d, _ = cv2.projectPoints(points_to_project, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def create_windowed_sequences(data, lookback, horizon):
    X = []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
    return np.array(X)

def inverse_transform_predictions(scaled_data, scaler):
    n_samples = len(scaled_data)
    temp_array = np.zeros((n_samples, cfg.N_FEATURES))
    temp_array[:, cfg.TARGET_INDICES_IN_FEATURES] = scaled_data
    inversed_array = scaler.inverse_transform(temp_array)
    return inversed_array[:, cfg.TARGET_INDICES_IN_FEATURES]

# --- FILTROS ---
def run_kalman_filter_predictor(X_data, horizon_steps, q_var, r_noise):
    observed_positions = X_data[:, -1, cfg.TARGET_INDICES_IN_FEATURES]
    n_samples, n_dims = observed_positions.shape
    predictions = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0., 0.]); kf.F = np.array([[1., cfg.DT], [0., 1.]]); kf.H = np.array([[1., 0.]])
        kf.R = np.eye(1) * r_noise; kf.Q = Q_discrete_white_noise(dim=2, dt=cfg.DT, var=q_var); kf.P = np.eye(2) * 1.
        F_k = np.linalg.matrix_power(kf.F, horizon_steps)
        dim_preds = []
        for i in range(n_samples):
            kf.predict(); kf.update(observed_positions[i, dim])
            dim_preds.append((F_k @ kf.x)[0])
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
        F_k = np.linalg.matrix_power(np.array([[1, cfg.DT, 0.5*cfg.DT**2], [0, 1, cfg.DT], [0, 0, 1]]), horizon_steps)
        dim_preds = []
        for i in range(n_samples):
            ukf.predict(); ukf.update(observed_positions[i, dim])
            dim_preds.append((F_k @ ukf.x)[0])
        predictions[:, dim] = np.array(dim_preds)
    return predictions

def run_lstm_kf_hybrid(lstm_predictions_inv):
    n_samples, n_dims = lstm_predictions_inv.shape
    kf_predictions = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([lstm_predictions_inv[0, dim], 0.])
        kf.F = np.array([[1., cfg.DT], [0., 1.]]); kf.H = np.array([[1., 0.]])
        kf.R = np.eye(1) * cfg.HYBRID_KF_R_NOISE; kf.Q = Q_discrete_white_noise(dim=2, dt=cfg.DT, var=cfg.HYBRID_KF_Q_VAR); kf.P = np.eye(2) * 1.
        dim_preds = []
        for i in range(n_samples):
            kf.predict(); kf.update(lstm_predictions_inv[i, dim])
            dim_preds.append(kf.x[0])
        kf_predictions[:, dim] = np.array(dim_preds)
    return kf_predictions

# 2. CLASE VISUALIZADOR (ESCALADO RELATIVO A LA RESOLUCIÓN)

class HybridVisualizer:
    def __init__(self, width, height, camera_matrix, dist_coeffs, manual_offset=(0,0)):
        self.w = width
        self.h = height
        
        # Panel Video (Izquierda)
        self.cam_mtx = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.manual_off_x = manual_offset[0]
        self.manual_off_y = manual_offset[1]
        
        # Panel Lienzo (Derecha)
        self.canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        
        # Auto-Scaling params
        self.scale_x = 1.0; self.scale_y = 1.0
        self.offset_x = 0.0; self.offset_y = 0.0

        # --- CONFIGURACIÓN VISUAL DINÁMICA ---
        # Usamos 480p como altura base de referencia
        reference_h = 480.0
        self.viz_scale = height / reference_h

        # Aplicamos el factor de escala, asegurando mínimos visibles (max(min_val, ...))
        self.LEGEND_SCALE = max(0.3, 0.5 * self.viz_scale)          # Tamaño texto leyenda
        self.LEGEND_THICKNESS = max(1, int(1 * self.viz_scale))     # Grosor texto leyenda
        self.COORD_SCALE = max(0.3, 0.5 * self.viz_scale)           # Tamaño texto coordenadas
        
        self.GT_POINT_RADIUS = max(3, int(6 * self.viz_scale))      # Radio Punto Verde
        self.PRED_POINT_RADIUS = max(2, int(4 * self.viz_scale))    # Radio Puntos Predicción
        self.CONN_LINE_THICKNESS = max(1, int(1 * self.viz_scale))  # Grosor línea conexión

        print(f"  [Visual] Resolución: {width}x{height} | Factor Escala: {self.viz_scale:.2f}")
        print(f"  [Visual] Radio Puntos: {self.GT_POINT_RADIUS}px / {self.PRED_POINT_RADIUS}px")

    def reset_canvas(self):
        self.canvas.fill(255)
        # Cruz central tenue
        cx, cy = self.w // 2, self.h // 2
        cv2.line(self.canvas, (cx, 0), (cx, self.h), (245, 245, 245), 1)
        cv2.line(self.canvas, (0, cy), (self.w, cy), (245, 245, 245), 1)
        
    def compute_auto_scale(self, all_points_3d):
        xs = -all_points_3d[:, 0]
        ys = -all_points_3d[:, 1]
        
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        if range_x == 0: range_x = 1
        if range_y == 0: range_y = 1
        
        scale_x = (self.w * 0.9) / range_x
        scale_y = (self.h * 0.9) / range_y
        final_scale = min(scale_x, scale_y)
        
        self.scale_x = final_scale
        self.scale_y = final_scale
        
        center_x_data = (min_x + max_x) / 2
        center_y_data = (min_y + max_y) / 2
        
        self.offset_x = (self.w / 2) - (center_x_data * final_scale)
        self.offset_y = (self.h / 2) - (center_y_data * final_scale)
        print(f"  [Auto-Scale] Factor Geométrico: {final_scale:.2f}")

    def get_scaled_point_canvas(self, x, y):
        x_inv = -x
        y_inv = -y
        px = int(x_inv * self.scale_x + self.offset_x)
        py = int(y_inv * self.scale_y + self.offset_y)
        return px, py

    def draw_legend(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.LEGEND_SCALE
        thick = self.LEGEND_THICKNESS
        
        # Ajustar espaciado vertical según escala
        line_gap = int(20 * self.viz_scale)
        start_x = int(10 * self.viz_scale)
        start_y = int(20 * self.viz_scale)
        
        items = [
            ("Verde: Ground Truth", cfg.COLOR_REAL),
            ("Azul: LSTM", cfg.COLOR_LSTM),
            ("Rojo: GRU", cfg.COLOR_GRU),
            ("Amarillo: CNN", cfg.COLOR_CNN),
            ("Magenta: Kalman", cfg.COLOR_KALMAN),
            ("Naranja: UKF", cfg.COLOR_UKF),
            ("Cian: Hibrido (Propuesto)", cfg.COLOR_LSTM_KF)
        ]
        
        for i, (text, color) in enumerate(items):
            y_pos = start_y + (i * line_gap)
            radius = max(2, int(4 * self.viz_scale))
            cv2.circle(self.canvas, (start_x + radius, y_pos - radius), radius, color, -1)
            cv2.putText(self.canvas, text, (start_x + int(15 * self.viz_scale), y_pos), font, scale, (0, 0, 0), thick)

    def draw_frame_content(self, current_gt, preds_dict):
        self.draw_legend()

        gt_cx, gt_cy = self.get_scaled_point_canvas(current_gt[0], current_gt[1])
        
        # Ground Truth
        cv2.circle(self.canvas, (gt_cx, gt_cy), self.GT_POINT_RADIUS, cfg.COLOR_REAL, -1)
        
        # Predicciones
        for name, p3d in preds_dict.items():
            color = self._get_color(name)
            pred_cx, pred_cy = self.get_scaled_point_canvas(p3d[0], p3d[1])
            
            cv2.line(self.canvas, (gt_cx, gt_cy), (pred_cx, pred_cy), color, self.CONN_LINE_THICKNESS, cv2.LINE_AA)
            cv2.circle(self.canvas, (pred_cx, pred_cy), self.PRED_POINT_RADIUS, color, -1)

    def _get_color(self, name):
        colors = {
            'LSTM': cfg.COLOR_LSTM, 'GRU': cfg.COLOR_GRU, 'CNN': cfg.COLOR_CNN,
            'Kalman': cfg.COLOR_KALMAN, 'UKF': cfg.COLOR_UKF, 'Hybrid': cfg.COLOR_LSTM_KF
        }
        return colors.get(name, (0,0,0))

    def get_canvas(self):
        return self.canvas
        
    def get_viz_scale(self):
        return self.viz_scale

# 3. MAIN

def main():
    print(f"--- GENERANDO VIDEOS (MODO FINAL: ESCALADO PROPORCIONAL A RESOLUCIÓN) ---")
    
    if not cfg.VISUALIZATION_MAPPING:
        print("No hay videos seleccionados en config.py.")
        return

    loaded_resources = {}
    for task in cfg.TASKS:
        try:
            res = {}
            res['scaler'] = joblib.load(cfg.get_scaler_path(task))
            res['lstm'] = load_model(os.path.join(cfg.MODEL_DIR, f'lstm_model_{task}.h5'), compile=False)
            res['gru'] = load_model(os.path.join(cfg.MODEL_DIR, f'gru_model_{task}.h5'), compile=False)
            res['cnn'] = load_model(os.path.join(cfg.MODEL_DIR, f'cnn_1d_model_{task}.h5'), compile=False)
            res['kf_p'] = joblib.load(os.path.join(cfg.SKOPT_TUNER_DIR, f'kf_best_params_{task}.gz'))
            res['ukf_p'] = joblib.load(os.path.join(cfg.SKOPT_TUNER_DIR, f'ukf_best_params_{task}.gz'))
            calib_path = cfg.TASK_PATHS[task]['calibration_file']
            res['cam_mtx'], res['dist'] = load_camera_calibration(calib_path)
            loaded_resources[task] = res
        except Exception as e:
            print(f"Error cargando recursos {task}: {e}")

    TARGET_MAP_RAW = cfg.TARGET_INDICES_IN_FEATURES

    for (v_task, v_skill, v_id), v_info in cfg.VISUALIZATION_MAPPING.items():
        if v_task not in loaded_resources: continue
        
        print(f"\n>>> Procesando Video: {v_id}")
        
        kin_path = os.path.join(cfg.TASK_PATHS[v_task]['kinematics_dir'], v_info['kinematics'])
        kin_df = pd.read_csv(kin_path, sep=r'\s+', header=None, engine='python')
        raw_data_full = kin_df.iloc[:, cfg.FEATURE_COLUMNS_INDICES].values
        
        res = loaded_resources[v_task]
        scaler = res['scaler']
        data_scaled = scaler.transform(raw_data_full)
        X_seq = create_windowed_sequences(data_scaled, cfg.LOOKBACK_STEPS, cfg.PREDICTION_STEPS)
        
        print("   Calculando predicciones IA...")
        preds_scaled = {
            'LSTM': res['lstm'].predict(X_seq, verbose=0),
            'GRU': res['gru'].predict(X_seq, verbose=0),
            'CNN': res['cnn'].predict(X_seq, verbose=0),
            'Kalman': run_kalman_filter_predictor(X_seq, cfg.PREDICTION_STEPS, *res['kf_p']),
            'UKF': run_ukf_predictor(X_seq, cfg.PREDICTION_STEPS, *res['ukf_p'])
        }
        preds_inv = {k: inverse_transform_predictions(v, scaler) for k, v in preds_scaled.items()}
        preds_inv['Hybrid'] = run_lstm_kf_hybrid(preds_inv['LSTM'])

        video_path = os.path.join(cfg.TASK_PATHS[v_task]['video_dir'], v_info['video'])
        
        for arm_name, arm_cfg in cfg.ARM_MAPPING.items():
            print(f"   -> Generando brazo: {arm_name}")
            
            idx_pred = arm_cfg['indices'] 
            idx_raw = [TARGET_MAP_RAW[i] for i in idx_pred]
            manual_off = cfg.MANUAL_OFFSETS.get((v_task, v_skill, v_id), {}).get(arm_name, (0,0))
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): continue
            
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            save_dir = os.path.join(cfg.VIDEO_OUT_DIR, arm_name)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{v_id}_{arm_name}_Proportional.mp4")
            
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*2, H))
            viz = HybridVisualizer(W, H, res['cam_mtx'], res['dist'], manual_offset=manual_off)
            
            full_traj_arm = raw_data_full[:, idx_raw]
            viz.compute_auto_scale(full_traj_arm)
            
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                left = frame.copy()
                viz.reset_canvas()
                
                # Texto proporcional en video
                font_scale = max(0.4, 0.6 * viz.get_viz_scale())
                line_spacing = int(30 * viz.get_viz_scale())
                y1, y2 = int(30 * viz.get_viz_scale()), int(60 * viz.get_viz_scale())
                
                cv2.putText(left, f"{v_task}|{v_skill}", (20, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 2)
                cv2.putText(left, f"{arm_name}", (20, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), 2)
                
                pred_i = frame_idx - cfg.LOOKBACK_STEPS
                target_raw_idx = frame_idx
                
                current_gt_3d = None
                current_preds_3d = {}
                
                if 0 <= pred_i < len(preds_inv['LSTM']):
                    if target_raw_idx < len(raw_data_full):
                        raw_vec = raw_data_full[target_raw_idx]
                        current_gt_3d = (raw_vec[idx_raw[0]], raw_vec[idx_raw[1]], raw_vec[idx_raw[2]])

                    for model_name in preds_inv.keys():
                        p = preds_inv[model_name][pred_i, idx_pred]
                        current_preds_3d[model_name] = p

                    viz.draw_frame_content(current_gt_3d, current_preds_3d)

                right = viz.get_canvas().copy()
                
                if current_gt_3d:
                    # Etiqueta coordenadas proporcional
                    rect_h = int(25 * viz.get_viz_scale())
                    font_coord = max(0.3, 0.5 * viz.get_viz_scale())
                    
                    cv2.rectangle(right, (10, H - rect_h - 10), (W-10, H-5), (255,255,255), -1)
                    lbl = f"Pos: {current_gt_3d[0]:.3f}, {current_gt_3d[1]:.3f}, {current_gt_3d[2]:.3f}"
                    cv2.putText(right, lbl, (20, H - 10), cv2.FONT_HERSHEY_SIMPLEX, font_coord, (0,0,0), 1)

                out.write(cv2.hconcat([left, right]))
                frame_idx += 1
                if frame_idx % 100 == 0: print(f"      Frame {frame_idx}", end='\r')
            
            cap.release()
            out.release()
            print(f"\n      Guardado: {out_path}")

    print("\n--- FIN ---")

if __name__ == "__main__":
    main()