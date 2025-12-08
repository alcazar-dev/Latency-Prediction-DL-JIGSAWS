import numpy as np
import os
import pandas as pd
from scipy.stats import wilcoxon

# --- CONFIGURACIÓN ---
BASE_DIR = r"D:\Estancia\TESTS\CODE\PYTHON\output"
TASKS = ['Suturing', 'Knot_Tying', 'Needle_Passing']
DT = 1.0 / 30.0 # 30 Hz

# Función para calcular Jerk Vectorizado (Misma lógica que tu script 05)
def get_jerk_distribution(trajectory, dt):
    # Derivada 1: Velocidad
    vel = np.diff(trajectory, axis=0) / dt
    # Derivada 2: Aceleración
    acc = np.diff(vel, axis=0) / dt
    # Derivada 3: Jerk
    jerk = np.diff(acc, axis=0) / dt
    
    # Magnitud del Jerk al cuadrado (Escalar por paso de tiempo)
    # Forma resultante: (N_muestras - 3,)
    jerk_squared_magnitude = np.sum(jerk**2, axis=1)
    return jerk_squared_magnitude

def run_jerk_statistics():
    print("--- ANÁLISIS ESTADÍSTICO DE SUAVIDAD (JERK/BIO-FIDELIDAD) ---")
    results = []

    for task in TASKS:
        print(f"\nProcesando: {task}")
        file_path = os.path.join(BASE_DIR, f"model_predictions_{task}.npz")
        
        if not os.path.exists(file_path):
            continue
            
        data = np.load(file_path)
        y_true = data['y_test'] # Humano
        
        # 1. Obtener distribuciones de Jerk
        # (Cortamos y_true para que coincida en tamaño tras las derivadas)
        jerk_human = get_jerk_distribution(y_true, DT)
        
        models_jerk = {}
        for name in ['preds_lstm', 'preds_cnn', 'preds_gru']:
            if name in data:
                models_jerk[name] = get_jerk_distribution(data[name], DT)

        # 2. COMPARACIÓN 1: ¿Es la LSTM más suave que la CNN?
        # Hipótesis: LSTM < CNN
        if 'preds_lstm' in models_jerk and 'preds_cnn' in models_jerk:
            stat, p_val = wilcoxon(models_jerk['preds_lstm'], models_jerk['preds_cnn'], alternative='less')
            winner = "LSTM es más suave" if p_val < 0.05 else "Iguales/CNN más suave"
            
            results.append({
                "Task": task,
                "Comparison": "LSTM vs CNN (Smoothness)",
                "P-Value": p_val,
                "Conclusion": winner
            })
            print(f"  > LSTM vs CNN: {winner} (p={p_val:.4e})")

        # 3. COMPARACIÓN 2: ¿Es la LSTM más suave que el HUMANO?
        # Hipótesis: LSTM < Humano (Filtrado de temblor)
        if 'preds_lstm' in models_jerk:
            # Ajustamos tamaños si difieren por 1 o 2 frames debido al diff
            min_len = min(len(jerk_human), len(models_jerk['preds_lstm']))
            stat, p_val = wilcoxon(models_jerk['preds_lstm'][:min_len], jerk_human[:min_len], alternative='less')
            
            is_smoother = "SÍ (Filtra temblor)" if p_val < 0.05 else "NO (Igual/Peor)"
            results.append({
                "Task": task,
                "Comparison": "LSTM vs HUMAN (Tremor Reduction)",
                "P-Value": p_val,
                "Conclusion": is_smoother
            })
            print(f"  > LSTM vs Humano: {is_smoother} (p={p_val:.4e})")

    # Guardar tabla
    df = pd.DataFrame(results)
    csv_path = os.path.join(BASE_DIR, "csv_results", "jerk_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en {csv_path}")

if __name__ == "__main__":
    run_jerk_statistics()