import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


# -----------------------------------------------------------------
# SCRIPT MAESTRO PARA EJECUTAR TODA LA TUBERÍA DE INVESTIGACIÓN
# -----------------------------------------------------------------
#
# Pasos:
# 1. (MANUAL) Edita 'config.py' para seleccionar las tareas
#    y habilidades que quieres procesar.
#
# 2. (AUTOMÁTICO) Ejecuta este script: python run_pipeline.py
#
# -----------------------------------------------------------------

# 1. Define el orden de ejecución
scripts_a_ejecutar = [
    'config.py',
    '01_Preprocesamiento_JIGSAWS.py',
    '02_Modelado_y_Baseline.py',
    '03_Evaluacion_y_Resultados.py',
    '04_Visualizacion_Video.py',
    '05_Evaluacion_Especifica.py'
]

# 2. Encuentra el ejecutable de Python
python_executable = sys.executable

print(f"--- INICIANDO PIPELINE DE INVESTIGACIÓN ---")
print(f"Usando el intérprete de Python: {python_executable}\n")
print(f"Directorio de trabajo de los scripts: {SCRIPT_DIR}\n")

pipeline_start_time = time.time()
total_scripts = len(scripts_a_ejecutar)

# 3. Itera y ejecuta cada script
for script in scripts_a_ejecutar:

    script_path = os.path.join(SCRIPT_DIR, script)
    
    if not os.path.exists(script_path):
        print(f"ADVERTENCIA: No se encontró el script '{script}' en {SCRIPT_DIR}. Saltando.")
        continue
        
    print(f"\n[PASO]: Ejecutando {script}...")
    print("-" * (len(script) + 18))
    
    script_start_time = time.time()

    try:
        subprocess.run(
            [python_executable, script_path], 
            check=True, 
            cwd=SCRIPT_DIR
        )
        print(f"[ÉXITO]: {script} completado.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR]: ¡Falló la ejecución de {script}!")
        print(f"El pipeline se ha detenido.")
        print(f"Error: {e}")
        break
    except KeyboardInterrupt:
        print("\n[DETENIDO]: Pipeline interrumpido por el usuario.")
        break

pipeline_end_time = time.time()
pipeline_duration = pipeline_end_time - pipeline_start_time
print("\n" + "="*50)
print(f"TIEMPO TOTAL DEL PIPELINE: {format_time(pipeline_duration)}")
print("="*50)

print("\n--- PIPELINE DE INVESTIGACIÓN FINALIZADO ---")