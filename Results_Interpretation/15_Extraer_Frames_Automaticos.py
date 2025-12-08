import cv2
import pandas as pd
import os
import config as cfg

def parse_timestamp_to_seconds(timestamp_str):
    """Convierte 'MM:SS.ms' (ej: 01:05.43) a segundos totales (float)."""
    try:
        parts = timestamp_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return (minutes * 60) + seconds
    except:
        return 0.0

def extract_frame(video_path, timestamp_str, output_path):
    """Abre el video, busca el segundo exacto y guarda el frame."""
    if not os.path.exists(video_path):
        print(f"   [ERROR] No existe video: {os.path.basename(video_path)}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   [ERROR] No se pudo abrir: {os.path.basename(video_path)}")
        return False

    # Obtener FPS para calcular el número de frame exacto
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_seconds = parse_timestamp_to_seconds(timestamp_str)
    target_frame = int(target_seconds * fps)

    # Ir al frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_path, frame)
        print(f"   [OK] Guardado: {os.path.basename(output_path)}")
    else:
        print(f"   [FAIL] No se pudo leer frame en {timestamp_str}")
    
    cap.release()
    return True

def main():
    print("--- AUTOMATIZACIÓN DE CAPTURAS DE PANTALLA (CRITICAL MOMENTS) ---")
    
    # 1. Cargar el CSV de momentos críticos
    csv_path = os.path.join(cfg.CSV_OUT_DIR, 'CRITICAL_MOMENTS_TIMEFRAMES.csv')
    if not os.path.exists(csv_path):
        print("Error: No se encontró CRITICAL_MOMENTS_TIMEFRAMES.csv. Ejecuta el script 12 primero.")
        return

    df = pd.read_csv(csv_path)
    print(f"Cargados {len(df)} momentos críticos.")

    # 2. Iterar sobre cada momento
    for index, row in df.iterrows():
        video_id = row['Video']
        arm = row['Arm']
        model = row['Model']
        
        # Tiempos de interés
        best_time = row['Best_Time_Center']
        worst_time = row['Worst_Time_Center']
        
        # Construir ruta del video fuente (generado por el script 04)
        # Nombre esperado: VideoID_Arm_Proportional.mp4
        video_filename = f"{video_id}_{arm}_Proportional.mp4"
        video_path = os.path.join(cfg.VIDEO_OUT_DIR, arm, video_filename)
        
        # Carpeta de salida (dentro de plots_specific_videos/VideoID/Arm/Screenshots)
        out_dir = os.path.join(cfg.PLOT_SPECIFIC_DIR, video_id, arm, 'Screenshots')
        os.makedirs(out_dir, exist_ok=True)
        
        # A. Extraer PEOR Momento (Falla)
        safe_time_worst = worst_time.replace(':', 'm').replace('.', 's')
        fname_worst = f"Fail_{model}_{safe_time_worst}.png"
        extract_frame(video_path, worst_time, os.path.join(out_dir, fname_worst))
        
        # B. Extraer MEJOR Momento (Éxito) - Opcional, descomentar 
        # safe_time_best = best_time.replace(':', 'm').replace('.', 's')
        # fname_best = f"Success_{model}_{safe_time_best}.png"
        # extract_frame(video_path, best_time, os.path.join(out_dir, fname_best))

    print("\n--- PROCESO TERMINADO ---")
    print("Revisa la carpeta 'output/plots_specific_videos/.../Screenshots'")

if __name__ == "__main__":
    main()