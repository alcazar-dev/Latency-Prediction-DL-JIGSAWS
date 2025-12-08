import pandas as pd
import os
import config as cfg

def main():
    print("--- EXTRAYENDO COORDENADAS NUMÉRICAS DE LOS GRÁFICOS SCATTER ---")
    
    # 1. Cargar el Archivo Maestro (donde están todos los datos crudos)
    master_path = os.path.join(cfg.CSV_OUT_DIR, 'MASTER_METRICS_ALL_VIDEOS.csv')
    if not os.path.exists(master_path):
        print(f"Error: No se encontró {master_path}")
        return

    df = pd.read_csv(master_path)
    
    # 2. Cargar Baseline Humano (para incluirlo en la tabla como referencia)
    baseline_path = os.path.join(cfg.CSV_OUT_DIR, 'HUMAN_BASELINE_MSJ.csv')
    baseline_dict = {}
    if os.path.exists(baseline_path):
        bs_df = pd.read_csv(baseline_path)
        # Diccionario {Tarea: Valor_MSJ}
        baseline_dict = dict(zip(bs_df['Task'], bs_df['Baseline_MSJ']))
    
    # 3. Calcular Promedios por Tarea y Modelo
    # Agrupamos por Tarea y Modelo, y calculamos la media de RMSE y MSJ
    # (Esto es exactamente lo que hace el script de graficación)
    grouped = df.groupby(['Task', 'Model'])[['RMSE', 'MSJ']].mean().reset_index()
    
    # 4. Agregar la Referencia Humana (Ground Truth) a la tabla
    # Creamos filas "artificiales" para el Humano, para que aparezca en tu CSV
    human_rows = []
    tasks = df['Task'].unique()
    for task in tasks:
        msj_val = baseline_dict.get(task, 524.0)
        human_rows.append({
            'Task': task,
            'Model': 'Ground Truth (Humano)',
            'RMSE': 0.0,      # El humano es la referencia, error 0 relativo a sí mismo
            'MSJ': msj_val
        })
    
    # Unir datos de modelos con datos humanos
    df_human = pd.DataFrame(human_rows)
    final_df = pd.concat([grouped, df_human], ignore_index=True)
    
    # 5. Ordenar para que se vea bonito
    final_df = final_df.sort_values(by=['Task', 'RMSE'])
    
    # 6. Guardar y Mostrar
    out_path = os.path.join(cfg.CSV_OUT_DIR, 'SCATTER_PLOT_COORDINATES.csv')
    final_df.to_csv(out_path, index=False)
    
    print("\n" + "="*80)
    print(f"DATOS GENERADOS EN: {out_path}")
    print("="*80)
    print(final_df)
    print("-" * 80)
    print("Copia estos valores para tu sección 4.2.2 o para tablas comparativas.")

if __name__ == "__main__":
    main()