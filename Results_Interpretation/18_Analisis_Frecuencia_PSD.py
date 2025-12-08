import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import pandas as pd

# --- CONFIGURACIÓN ---
BASE_DIR = r"D:\Estancia\TESTS\CODE\PYTHON\output"
PLOT_DIR = os.path.join(BASE_DIR, "plots", "psd_analysis")
CSV_DIR = os.path.join(BASE_DIR, "csv_results")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

TASKS = ['Suturing', 'Knot_Tying', 'Needle_Passing']
FS = 30.0  # Frecuencia de muestreo (30 Hz del da Vinci)

def calculate_average_psd(trajectory, fs):
    """
    Calcula la PSD promedio de los 3 ejes (x, y, z).
    Retorna frecuencias y densidades.
    """
    psd_accum = None
    freqs = None
    
    # trajectory shape: (Samples, 3)
    for axis in range(3): 
        signal = trajectory[:, axis]
        # nperseg=256 nos da buena resolución para señales cortas
        f, Pxx = welch(signal, fs=fs, nperseg=256, scaling='density')
        
        if psd_accum is None:
            psd_accum = Pxx
            freqs = f
        else:
            psd_accum += Pxx
            
    # Promediar los 3 ejes
    psd_avg = psd_accum / 3.0
    return freqs, psd_avg

def run_psd_analysis():
    print("--- GENERANDO ANÁLISIS DE FRECUENCIA (PSD + CSV) ---")
    
    # Lista para guardar datos del CSV
    all_psd_data = []
    
    for task in TASKS:
        print(f"Procesando: {task}...")
        file_path = os.path.join(BASE_DIR, f"model_predictions_{task}.npz")
        
        if not os.path.exists(file_path):
            print(f"  [ERROR] No encontrado: {file_path}")
            continue
            
        data = np.load(file_path)
        y_true = data['y_test']
        
        # Validar existencia de modelos
        preds_lstm = data['preds_lstm'] if 'preds_lstm' in data else None
        preds_cnn = data['preds_cnn'] if 'preds_cnn' in data else None
        
        if preds_lstm is None or preds_cnn is None:
            print("  Faltan modelos en el archivo. Saltando.")
            continue

        # --- CALCULAR PSD ---
        f_human, p_human = calculate_average_psd(y_true, FS)
        f_lstm, p_lstm = calculate_average_psd(preds_lstm, FS)
        f_cnn, p_cnn = calculate_average_psd(preds_cnn, FS)
        
        # --- GUARDAR DATOS PARA CSV ---
        # Como las frecuencias (f_human, f_lstm, etc.) son iguales para todos (mismo nperseg/fs),
        # podemos guardar una fila por cada frecuencia.
        for i in range(len(f_human)):
            all_psd_data.append({
                "Task": task,
                "Frequency_Hz": f_human[i],
                "PSD_Human": p_human[i],
                "PSD_LSTM": p_lstm[i],
                "PSD_CNN": p_cnn[i]
            })

        # --- GRAFICAR (Igual que antes) ---
        plt.figure(figsize=(10, 6))
        plt.semilogy(f_human, p_human, label='Humano (GT)', color='black', alpha=0.6, linewidth=2)
        plt.semilogy(f_cnn, p_cnn, label='CNN (1D)', color='red', linestyle='--', linewidth=1.5)
        plt.semilogy(f_lstm, p_lstm, label='LSTM', color='blue', linewidth=2)
        
        plt.title(f'Análisis Espectral: {task}', fontsize=14)
        plt.xlabel('Frecuencia [Hz]', fontsize=12)
        plt.ylabel('Densidad de Potencia [$m^2/Hz$]', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.axvline(x=3, color='gray', linestyle=':', label='Inicio Temblor')
        
        out_name = os.path.join(PLOT_DIR, f"PSD_{task}.png")
        plt.savefig(out_name, dpi=300)
        plt.close()
        print(f"  Gráfica guardada en: {out_name}")

    # --- EXPORTAR CSV ---
    df = pd.DataFrame(all_psd_data)
    csv_path = os.path.join(CSV_DIR, "psd_spectral_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDatos crudos guardados en: {csv_path}")

if __name__ == "__main__":
    run_psd_analysis()