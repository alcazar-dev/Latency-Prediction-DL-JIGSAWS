import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config as cfg

def main():
    print("--- GENERANDO GRÁFICOS DE DISPERSIÓN (TRADE-OFF) ---")
    
    # 1. Cargar Datos Maestros
    master_path = os.path.join(cfg.CSV_OUT_DIR, 'MASTER_METRICS_ALL_VIDEOS.csv')
    if not os.path.exists(master_path):
        print(f"Error: No se encontró {master_path}")
        return

    df = pd.read_csv(master_path)
    
    # 2. Cargar Baseline Humano (si existe, sino usar 524 global)
    baseline_path = os.path.join(cfg.CSV_OUT_DIR, 'HUMAN_BASELINE_MSJ.csv')
    baseline_dict = {}
    if os.path.exists(baseline_path):
        bs_df = pd.read_csv(baseline_path)
        # Crear diccionario {Task: MSJ_Value}
        baseline_dict = dict(zip(bs_df['Task'], bs_df['Baseline_MSJ']))
    
    # 3. Configurar Estilo
    sns.set_theme(style="whitegrid", context="talk") # Contexto 'talk' hace letras más grandes para papers
    tasks = sorted(df['Task'].unique())
    
    # Mapeo de Colores (Consistente con tus otros plots)
    model_colors = {
        'LSTM': 'blue', 
        'GRU': 'red', 
        'CNN': 'orange', 
        'KALMAN': 'purple', 
        'UKF_Baseline': 'brown',
        'LSTM-KF_Hybrid': 'cyan',
        'AR_Model (MA)': 'gray'
    }

    # 4. Generar Gráficos
    for task in tasks:
        print(f"   -> Procesando: {task}")
        plt.figure(figsize=(12, 10))
        
        # Filtrar datos de la tarea y agrupar por modelo (promedio)
        task_df = df[df['Task'] == task]
        summary = task_df.groupby('Model')[['RMSE', 'MSJ']].mean().reset_index()
        
        # Obtener Baseline para esta tarea
        human_msj = baseline_dict.get(task, 524.0) 
        
        # A. DIBUJAR MODELOS
        # Usamos scatterplot de seaborn
        sns.scatterplot(
            data=summary, 
            x='RMSE', 
            y='MSJ', 
            hue='Model', 
            style='Model', # Diferentes formas para accesibilidad
            s=400,         # Tamaño grande de puntos
            palette=model_colors,
            zorder=5
        )
        
        # B. DIBUJAR GROUND TRUTH (HUMANO)
        # Coordenada (0, human_msj)
        plt.scatter([0], [human_msj], color='green', s=600, marker='*', label='Ground Truth (Ideal)', zorder=6, edgecolors='black')
        
        # C. ETIQUETAS Y ZONAS
        # Etiquetas de texto para cada punto
        for i, row in summary.iterrows():
            # Ajuste fino de la posición del texto para que no tape el punto
            plt.text(
                row['RMSE'] + (summary['RMSE'].max() * 0.02), 
                row['MSJ'], 
                row['Model'], 
                fontsize=11, 
                color='black',
                weight='bold',
                va='center'
            )

        # Línea de Referencia Humana
        plt.axhline(y=human_msj, color='green', linestyle='--', alpha=0.5)
        plt.text(summary['RMSE'].max(), human_msj + 5, ' Bio-Fidelidad (Humano)', color='green', va='bottom', ha='right', fontsize=12, fontweight='bold')
        
        # Zona de "Over-smoothing" (Cerca de 0)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.text(summary['RMSE'].max(), 5, ' Over-smoothing (Robótico)', color='gray', va='bottom', ha='right', fontsize=10, fontstyle='italic')

        # Títulos y Ejes
        plt.title(f'Trade-off Precisión vs. Naturalidad: {task}', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Error de Posición (RMSE) [mm] $\leftarrow$ (Mejor)', fontsize=14, fontweight='bold')
        plt.ylabel('Suavidad (MSJ) $\leftarrow$ (Más Natural)', fontsize=14, fontweight='bold')
        
        # Ajustar límites para que se vea bien el cero
        plt.xlim(left=-0.001) # Un poco a la izquierda de 0 para ver la estrella
        plt.ylim(bottom=-10)
        
        # Leyenda
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Modelo')
        
        plt.tight_layout()
        
        # Guardar
        save_path = os.path.join(cfg.PLOT_DIR, f'TradeOff_Scatter_{task}.png')
        plt.savefig(save_path, dpi=300)
        print(f"      Guardado: {save_path}")
        plt.close()

if __name__ == "__main__":
    main()