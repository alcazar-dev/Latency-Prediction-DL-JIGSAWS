import pandas as pd
import numpy as np
import os
import config as cfg

def main():
    # 1. Cargar el archivo maestro
    # Asegúrate de que este archivo exista (se genera con el script 06)
    file_path = os.path.join(cfg.CSV_OUT_DIR, 'MASTER_METRICS_ALL_VIDEOS.csv')
    
    print(f"--- LEYENDO DATOS DE: {file_path} ---")
    
    if not os.path.exists(file_path):
        print("ERROR: No se encontró el archivo MASTER_METRICS_ALL_VIDEOS.csv")
        print("Ejecuta primero el script '06_Analisis_Estadistico.py'.")
        return

    df = pd.read_csv(file_path)
    
    # Vamos a analizar dos métricas principales
    metrics = ['RMSE', 'MSJ']
    
    print("\n" + "="*80)
    print("ESTADÍSTICAS DETALLADAS PARA INTERPRETACIÓN DE BOXPLOTS")
    print("="*80)

    for metric in metrics:
        print(f"\n>>> ANÁLISIS DE {metric} (Menor es mejor) <<<")
        
        # Agrupamos por Modelo y Brazo (opcional, si quieres ver diferencias izq/der)
        # Calculamos los estadísticos clave de un boxplot
        stats = df.groupby(['Model', 'Arm'])[metric].describe(percentiles=[0.25, 0.5, 0.75])
        
        # Seleccionamos y renombramos para que sea fácil de leer en la tesis
        # 50% es la Mediana (la línea en medio de la caja)
        # 25% (Q1) y 75% (Q3) son los bordes de la caja
        summary = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        summary.columns = ['N_Muestras', 'Media', 'Desv_Std', 'Min', 'Q1 (25%)', 'Mediana', 'Q3 (75%)', 'Max']
        
        # Ordenamos por la Media para ver el Ranking automáticamente
        summary = summary.sort_values(by='Media', ascending=True)
        
        print(summary)
        
        # Calcular mejora porcentual respecto al Baseline (AR) si existe
        # (Esto es oro para el texto de resultados)
        try:
            # Tomamos el promedio global por modelo (sin separar brazos) para el ranking general
            global_stats = df.groupby('Model')[metric].mean().sort_values()
            best_model = global_stats.index[0]
            worst_model = global_stats.index[-1]
            
            if 'AR_Model (MA)' in global_stats:
                baseline_val = global_stats['AR_Model (MA)']
                best_val = global_stats[best_model]
                improvement = ((baseline_val - best_val) / baseline_val) * 100
                
                print(f"\n[INSIGHT {metric}]:")
                print(f"   - Mejor Modelo Global: {best_model} ({best_val:.4f})")
                print(f"   - Baseline (AR): {baseline_val:.4f}")
                print(f"   - Mejora porcentual del mejor modelo vs AR: {improvement:.2f}%")
        except Exception as e:
            pass

        print("-" * 80)

    # ---------------------------------------------------------
    # TABLA DE RANKING GLOBAL (Para la Figura de Barras)
    # ---------------------------------------------------------
    print("\n>>> RANKING GLOBAL DE MODELOS (Promedio de ambos brazos) <<<")
    ranking = df.groupby('Model')[['RMSE', 'MSJ']].agg(['mean', 'std'])
    
    # Aplanamos las columnas
    ranking.columns = ['RMSE_Mean', 'RMSE_Std', 'MSJ_Mean', 'MSJ_Std']
    ranking = ranking.sort_values('RMSE_Mean')
    
    print(ranking)
    
    # Guardar en un CSV limpio para copiar y pegar a Excel/LaTeX
    out_file = os.path.join(cfg.CSV_OUT_DIR, 'TESIS_DATOS_NUMERICOS.csv')
    ranking.to_csv(out_file)
    print(f"\nArchivo guardado para tu reporte: {out_file}")

if __name__ == "__main__":
    main()