import numpy as np
import os
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = r"D:\Estancia\TESTS\CODE\PYTHON\output"
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "csv_results", "statistical_results_summary.csv")

TASKS = ['Suturing', 'Knot_Tying', 'Needle_Passing']
MODELS_TO_COMPARE = ['preds_lstm', 'preds_cnn', 'preds_gru', 'preds_ar']

def run_statistical_tests():
    print("--- INICIANDO ANÁLISIS ESTADÍSTICO (Generando CSV) ---")
    
    # Lista para acumular todos los resultados y luego hacer el CSV
    all_results_data = []

    for task in TASKS:
        print(f"\nPROCESANDO TAREA: {task}")
        file_path = os.path.join(BASE_DIR, f"model_predictions_{task}.npz")
        
        if not os.path.exists(file_path):
            print(f"[ERROR] No se encontró: {file_path}")
            continue
            
        try:
            data = np.load(file_path)
            y_true = data['y_test']
            model_errors = {}
            
            # Calcular errores por muestra
            for model_name in MODELS_TO_COMPARE:
                if model_name in data:
                    preds = data[model_name]
                    # MSE por muestra
                    mse_per_sample = np.mean((y_true - preds)**2, axis=1)
                    model_errors[model_name] = mse_per_sample

            # --- TEST 1: FRIEDMAN ---
            if len(model_errors) >= 3:
                stat, p_global = friedmanchisquare(*model_errors.values())
                
                # Guardar en la lista para el CSV
                all_results_data.append({
                    "Task": task,
                    "Test_Type": "Friedman (Global)",
                    "Comparison": "All Models",
                    "P-Value": p_global, # Guardamos el valor exacto (float)
                    "Statistic": stat,
                    "Winner_or_Result": "Significant" if p_global < 0.05 else "No Diff"
                })
                print(f"> Friedman P-value: {p_global:.4e}")

            # --- TEST 2: WILCOXON (vs LSTM) ---
            champion = 'preds_lstm' 
            if champion in model_errors:
                champion_err = model_errors[champion]
                
                for rival, rival_err in model_errors.items():
                    if rival == champion: continue
                    
                    # Test Two-Sided
                    stat, p_val = wilcoxon(champion_err, rival_err, alternative='two-sided')
                    
                    # Determinar ganador
                    if p_val > 0.05:
                        winner = "Tie"
                    else:
                        if np.mean(champion_err) < np.mean(rival_err):
                            winner = f"{champion} Wins"
                        else:
                            winner = f"{rival} Wins"

                    # Guardar en la lista
                    all_results_data.append({
                        "Task": task,
                        "Test_Type": "Wilcoxon (Pairwise)",
                        "Comparison": f"{champion} vs {rival}",
                        "P-Value": p_val,
                        "Statistic": stat,
                        "Winner_or_Result": winner
                    })
                    print(f"> Wilcoxon {rival}: {winner} (p={p_val:.4e})")

        except Exception as e:
            print(f"Error en {task}: {e}")

    # --- GUARDAR CSV FINAL ---
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    df = pd.DataFrame(all_results_data)
    
    # Ordenar columnas para que se vea bonito
    df = df[["Task", "Test_Type", "Comparison", "Winner_or_Result", "P-Value", "Statistic"]]
    
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("\n" + "="*60)
    print(f"¡LISTO! Archivo CSV guardado en:\n{OUTPUT_CSV_PATH}")
    print("="*60)

if __name__ == "__main__":
    run_statistical_tests()