import numpy as np
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# --- 1. MODO DE DEPURACIÓN (DRY RUN) ---
# - Debug: TRUE -
# DEBUG_MODE = True

# - Final: FALSE - 
DEBUG_MODE = False

# ----- ----- ----- ----- ----- ----- -----

# --- 2. CONFIGURACIÓN DEL EXPERIMENTO ---
TASKS_FINAL = ['Suturing', 'Knot_Tying', 'Needle_Passing']
SKILLS_FINAL = ['N', 'I', 'E']

# --- 3. RUTAS DE ENTRADA (INPUTS) ---
TASK_PATHS = {
    'Suturing': {
        'base': 'D:\\Estancia\\JIGSAWS\\Suturing\\Suturing',
        'meta_file': 'D:\\Estancia\\JIGSAWS\\Suturing\\Suturing\\meta_file_Suturing.txt',
        'video_dir': 'D:\\Estancia\\JIGSAWS\\Suturing\\Suturing\\video',
        'kinematics_dir': 'D:\\Estancia\\JIGSAWS\\Suturing\\Suturing\\kinematics\\AllGestures',
        'transcriptions_dir': 'D:\\Estancia\\JIGSAWS\\Suturing\\Suturing\\transcriptions',
        'calibration_file': 'D:\\Estancia\\JIGSAWS\\meta_file_Suturing.avi.yml'
    },
    'Knot_Tying': {
        'base': 'D:\\Estancia\\JIGSAWS\\Knot_Tying\\Knot_Tying',
        'meta_file': 'D:\\Estancia\\JIGSAWS\\Knot_Tying\\Knot_Tying\\meta_file_Knot_Tying.txt',
        'video_dir': 'D:\\Estancia\\JIGSAWS\\Knot_Tying\\Knot_Tying\\video',
        'kinematics_dir': 'D:\\Estancia\\JIGSAWS\\Knot_Tying\\Knot_Tying\\kinematics\\AllGestures',
        'transcriptions_dir': 'D:\\Estancia\\JIGSAWS\\Knot_Tying\\Knot_Tying\\transcriptions',
        'calibration_file': 'D:\\Estancia\\JIGSAWS\\meta_file_Knot_Tying.avi.yml'
    },
    'Needle_Passing': {
        'base': 'D:\\Estancia\\JIGSAWS\\Needle_Passing\\Needle_Passing',
        'meta_file': 'D:\\Estancia\\JIGSAWS\\Needle_Passing\\Needle_Passing\\meta_file_Needle_Passing.txt',
        'video_dir': 'D:\\Estancia\\JIGSAWS\\Needle_Passing\\Needle_Passing\\video',
        'kinematics_dir': 'D:\\Estancia\\JIGSAWS\\Needle_Passing\\Needle_Passing\\kinematics\\AllGestures',
        'transcriptions_dir': 'D:\\Estancia\\JIGSAWS\\Needle_Passing\\Needle_Passing\\transcriptions',
        'calibration_file': 'D:\\Estancia\\JIGSAWS\\meta_file_Needle_Passing.avi.yml'
    }
}

_MASTER_VISUALIZATION_MAPPING = {
    ('Suturing', 'N', 'Suturing_G002'):       {'video': 'Suturing_G002_capture1.avi',     'kinematics': 'Suturing_G002.txt'},
    ('Suturing', 'I', 'Suturing_F002'):       {'video': 'Suturing_F002_capture1.avi',     'kinematics': 'Suturing_F002.txt'},
    ('Suturing', 'E', 'Suturing_E001'):       {'video': 'Suturing_E001_capture1.avi',     'kinematics': 'Suturing_E001.txt'},
    ('Knot_Tying', 'N', 'Knot_Tying_G002'):   {'video': 'Knot_Tying_G002_capture1.avi',   'kinematics': 'Knot_Tying_G002.txt'},
    ('Knot_Tying', 'I', 'Knot_Tying_C003'):   {'video': 'Knot_Tying_C003_capture1.avi',   'kinematics': 'Knot_Tying_C003.txt'},
    ('Knot_Tying', 'E', 'Knot_Tying_E004'):   {'video': 'Knot_Tying_E004_capture1.avi',   'kinematics': 'Knot_Tying_E004.txt'},
    ('Needle_Passing', 'N', 'Needle_Passing_I004'): {'video': 'Needle_Passing_I004_capture1.avi', 'kinematics': 'Needle_Passing_I004.txt'},
    ('Needle_Passing', 'I', 'Needle_Passing_F003'): {'video': 'Needle_Passing_F003_capture1.avi', 'kinematics': 'Needle_Passing_F003.txt'},
    ('Needle_Passing', 'E', 'Needle_Passing_D005'): {'video': 'Needle_Passing_D005_capture1.avi', 'kinematics': 'Needle_Passing_D005.txt'},
}

# --- 4. FILTRADO DINÁMICO ---
if DEBUG_MODE:
    print("="*50)
    print("¡¡¡ MODO DE DEPURACIÓN ACTIVO (DRY RUN) !!!")
    print("Se usarán parámetros rápidos para probar el pipeline.")
    print("="*50)
    TASKS = ['Suturing']  
    SKILLS = ['N']        
else:
    TASKS = TASKS_FINAL
    SKILLS = SKILLS_FINAL

print(f"\n--- Configuración de Tareas Cargada ---")
print(f"Tareas a procesar (01, 02, 03): {TASKS}")
print(f"Niveles a visualizar (04): {SKILLS}")
print("---------------------------------------\n")

VISUALIZATION_MAPPING = {}
for (task, skill, video_id), info in _MASTER_VISUALIZATION_MAPPING.items():
    if task in TASKS and skill in SKILLS:
        VISUALIZATION_MAPPING[(task, skill, video_id)] = info

# --- 5. RUTAS DE SALIDA (OUTPUTS) ---
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
VIDEO_OUT_DIR = os.path.join(OUTPUT_DIR, 'videos')
DATA_OUT_DIR = os.path.join(OUTPUT_DIR, 'data')
CSV_OUT_DIR = os.path.join(OUTPUT_DIR, 'csv_results')
KERAS_TUNER_DIR = os.path.join(OUTPUT_DIR, 'keras_tuner') 
SKOPT_TUNER_DIR = os.path.join(OUTPUT_DIR, 'skopt_tuner')

PLOT_SPECIFIC_DIR = os.path.join(OUTPUT_DIR, 'plots_specific_videos')
CSV_SPECIFIC_DIR = os.path.join(OUTPUT_DIR, 'csv_specific_videos')

OFFSET_TEST_DIR = os.path.join(OUTPUT_DIR, 'offset_test_frames')

for d in [MODEL_DIR, PLOT_DIR, VIDEO_OUT_DIR, DATA_OUT_DIR, CSV_OUT_DIR, KERAS_TUNER_DIR, SKOPT_TUNER_DIR, PLOT_SPECIFIC_DIR, CSV_SPECIFIC_DIR, OFFSET_TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# Funciones para generar nombres de archivo por TAREA
def get_scaler_path(task):
    return os.path.join(DATA_OUT_DIR, f'scaler_{task}.gz')

# config.py
def get_processed_data_path(task, finetune=False):
    if finetune:
        return os.path.join(DATA_OUT_DIR, f'processed_data_finetune_{task}.npz')
    return os.path.join(DATA_OUT_DIR, f'processed_data_{task}.npz')

def get_results_path(task):
    return os.path.join(OUTPUT_DIR, f'model_predictions_{task}.npz')

# --- EN config.py ---

# --- 6. PARÁMETROS DE PREPROCESAMIENTO Y MODELO ---
SAMPLING_RATE_HZ = 30
DT = 1.0 / SAMPLING_RATE_HZ
PREDICTION_HORIZON_MS = 200
PREDICTION_STEPS = int(np.round(PREDICTION_HORIZON_MS / (1000.0 / SAMPLING_RATE_HZ)))
LOOKBACK_STEPS = 20  
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2 

# INTENCIÓN HUMANA (MTM) -> MOVIMIENTO ROBOT (PSM)


# EN CONFIG.PY

# --- CONFIGURACIÓN PARA: AMBOS BRAZOS (BIMANUAL) ---

# 1. FEATURES (ENTRADAS): 18 columnas en total
# (9 de la Derecha + 9 de la Izquierda)
FEATURE_COLUMNS_INDICES = [
    # --- MANO DERECHA (LO QUE YA TIENES) ---
    19, 20, 21,  # MTM Right Pos
    31, 32, 33,  # MTM Right Vel
    38, 39, 40,  # PSM1 Right Pos (Contexto)
    
    # --- MANO IZQUIERDA (LO NUEVO) ---
    0, 1, 2,     # MTM Left Pos (Indices 1-3 en txt)
    12, 13, 14,  # MTM Left Vel (Indices 13-15 en txt)
    57, 58, 59   # PSM2 Left Pos (Indices 58-60 en txt)
]

# 2. TARGETS (OBJETIVOS): 6 columnas en total (3 Der + 3 Izq)
TARGET_COLUMNS_INDICES = [
    38, 39, 40,  # PSM1 Right Pos
    57, 58, 59   # PSM2 Left Pos
]


# --- EN config.py (Sección 6) ---

# (Definición de FEATURE_COLUMNS_INDICES y TARGET_COLUMNS_INDICES que ya hicimos)
# Asegúrate de que TARGET_COLUMNS tenga 6 valores (3 der + 3 izq)

# === NUEVO: MAPEO DE BRAZOS PARA ORGANIZACIÓN ===
# Estos índices son RELATIVOS a la salida de la predicción (0 a 5), 
# no al archivo de texto original.
# Si TARGET_COLUMNS_INDICES tiene 6 columnas (3 PSM1 + 3 PSM2):
# 0,1,2 -> Primeras 3 columnas del Target (PSM1 - Derecha)
# 3,4,5 -> Siguientes 3 columnas del Target (PSM2 - Izquierda)

ARM_MAPPING = {
    'Right_PSM1': {'indices': [0, 1, 2], 'color_suffix': '_R'},
    'Left_PSM2':  {'indices': [3, 4, 5], 'color_suffix': '_L'}
}

# NOTA: Si en el futuro solo entrenas 1 brazo, comenta la línea de 'Left_PSM2'.

# ==============================================================================
# NO TOCAR NADA DEBAJO DE ESTA LÍNEA (Cálculos Automáticos)
# ==============================================================================

N_FEATURES = len(FEATURE_COLUMNS_INDICES)
N_TARGETS = len(TARGET_COLUMNS_INDICES)
INPUT_SHAPE = (LOOKBACK_STEPS, N_FEATURES)

# Esta línea busca automáticamente dónde quedaron los Targets dentro de los Features
TARGET_INDICES_IN_FEATURES = [FEATURE_COLUMNS_INDICES.index(i) for i in TARGET_COLUMNS_INDICES]

# --- 6B. PARÁMETROS DE FINE-TUNING ---
USE_FINE_TUNING = True # True para activar el flujo de Pre-entrenamiento/Fine-Tuning

# Configuración del meta_file (asumiendo archivo de texto separado por espacios)
META_FILE_ID_COLUMN = 0     # ID
META_FILE_SCORE_COLUMN = 2  # Puntaje de habilidad
FINE_TUNING_SCORE_THRESHOLD = 18.0 # Puntaje mínimo 
FINE_TUNING_LEARNING_RATE_MULTIPLIER = 0.1 # Reduce el LR para el fine-tuning

if DEBUG_MODE:
    USE_FINE_TUNING = True


# config.py

# --- 7. PARÁMETROS DE TUNING ---
if DEBUG_MODE:
    EPOCHS_TUNING = 2   # (KerasTuner)
    EPOCHS_FINAL = 2    # (Todos)
    EPOCHS_FINETUNE = 1 # ('Expertos')
    
    PATIENCE = 1
    DL_TUNER_TRIALS = 2   
    FILTER_TUNER_TRIALS = 2 
    FILTER_TUNER_N_INITIAL_POINTS = 1 
    
    DEBUG_DATA_LIMIT = 5000 
    DEBUG_FRAME_LIMIT = 50  
else:
    # --- Parámetros reales para el paper ---
    EPOCHS_TUNING = 50   # (KerasTuner)
    EPOCHS_FINAL = 100   # ('Todos')
    EPOCHS_FINETUNE = 25 # ('Expertos')
    
    PATIENCE = 5       
    DL_TUNER_TRIALS = 15   
    FILTER_TUNER_TRIALS = 30 
    FILTER_TUNER_N_INITIAL_POINTS = 10

# Rangos de búsqueda para Filtros (Q y R)
FILTER_R_SPACE = (1e-6, 1e+1, 'log-uniform') 
FILTER_Q_SPACE = (1e-6, 1e+1, 'log-uniform') 

# Hiperparámetros de Filtros (fijos)
UKF_ALPHA = 0.1
UKF_BETA = 2.0
UKF_KAPPA = 0.0
HYBRID_KF_R_NOISE = 0.5
HYBRID_KF_Q_VAR = 0.01

# --- 8. PARÁMETROS DE VISUALIZACIÓN (Colores) ---
KF_R_NOISE_VIZ = 0.1
KF_Q_VAR_VIZ = 0.01

COLOR_REAL = (0, 255, 0)
COLOR_LSTM = (0, 0, 255)
COLOR_GRU = (255, 0, 0)
COLOR_CNN = (0, 255, 255)
COLOR_KALMAN = (255, 0, 255)
COLOR_UKF = (255, 128, 0)
COLOR_LSTM_KF = (150, 150, 150)


# --- 9. OFFSETS MANUALES DE VISUALIZACIÓN ---
# Si la calibración (YML) es incorrecta, los puntos pueden
# aparecer fuera de lugar. Usa este diccionario para
# añadir un offset (X, Y) manual a cada video.
# Encontrar estos valores por prueba y error.
# Revisar archivos Calculadora_Offset.py y Offset_Visualization.py
# --------------------------------------------------

# --- EN config.py ---

# FORMATO: (TASK, SKILL, VIDEO): { 'Right_PSM1': (X, Y), 'Left_PSM2': (X, Y) }

MANUAL_OFFSETS = {
    # --- SUTURING ---
    ('Suturing', 'N', 'Suturing_G002'): {
        'Right_PSM1': (-2477, -2264), 
        'Left_PSM2':  (-2336, -1463)
    },
    ('Suturing', 'I', 'Suturing_F002'): {
        'Right_PSM1': (-1580, -1748), 
        'Left_PSM2':  (-1123, -445)
    },
    ('Suturing', 'E', 'Suturing_E001'): {
        'Right_PSM1': (0, 0), # Datos corruptos en raw (-1025080), mejor dejar en 0 o ignorar
        'Left_PSM2':  (-4234, -1944)
    },

    # --- KNOT TYING ---
    ('Knot_Tying', 'N', 'Knot_Tying_G002'): {
        'Right_PSM1': (-1271, -1985), 
        'Left_PSM2':  (-705, -209)
    },
    ('Knot_Tying', 'I', 'Knot_Tying_C003'): {
        'Right_PSM1': (-1291, -1189), 
        'Left_PSM2':  (-1116, -216)
    },
    ('Knot_Tying', 'E', 'Knot_Tying_E004'): {
        'Right_PSM1': (-2516, -1052), 
        'Left_PSM2':  (-1295, -225)
    },

    # --- NEEDLE PASSING ---
    ('Needle_Passing', 'N', 'Needle_Passing_I004'): {
        'Right_PSM1': (-457, 3831), 
        'Left_PSM2':  (-4862, -1313)
    },
    ('Needle_Passing', 'I', 'Needle_Passing_F003'): {
        'Right_PSM1': (-722, 2271), 
        'Left_PSM2':  (-2276, -799)
    },
    ('Needle_Passing', 'E', 'Needle_Passing_D005'): {
        'Right_PSM1': (377, -6388), 
        'Left_PSM2':  (-2496, -856)
    }
}

OFFSET_TEST_FRAME_NUMBER = 1

# Video de salida "lado a lado"
CREATE_SIDE_BY_SIDE_VIZ = True

# Fotograma donde se basa el "centro"?
CENTERING_FRAME_NUMBER = 1