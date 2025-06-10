import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Elijo fase 1 o fase 2 (fase1=false):
fase1 = True

# --- Datos Finales para el Gráfico de Radar ---
# Metricas de Entrenamiento en RWF-2000:
data_fase1 = {
    'Modelo': ['ViViT', 'I3D', 'SlowFast', 'TSM'],
    'Rendimiento (RWF-2000)': [0.876, 0.841, 0.846, 0.845], # F1-Score en RWF-2000
    'Robustez (RLVS)': [0.887, 0.825, 0.741, 0.867], # F1-Score en cross-inf RLVS
    'Robustez (Hockey)': [0.396, 0.643, 0.755, 0.723], # F1-Score en cross-inf Hockey
    'Eficiencia (Velocidad)': [4.9, 12.8, 38.9, 72.7], # FPS
    'Eficiencia (Coste)': [1/451.8, 1/149.1, 1/50.6, 1/32.9] # 1 / GFLOPs
}
# Metricas de Entrenamiento en RLVS:
data_fase2 = {
    'Modelo': ['ViViT', 'I3D', 'SlowFast', 'TSM'],
    'Rendimiento (RLVS)': [0.992, 0.970, 0.967, 0.983], # F1-Score en RLVS
    'Robustez (RWF-2000)': [0.775, 0.692, 0.770, 0.700], # F1-Score en cross-inf RWF
    'Robustez (Hockey)': [0.696, 0.681, 0.796, 0.684], # F1-Score en cross-inf Hockey
    'Eficiencia (Velocidad)': [4.9, 12.8, 38.9, 72.7], # FPS
    'Eficiencia (Coste)': [1/451.8, 1/149.1, 1/50.6, 1/32.9] # 1 / GFLOPs
}

if fase1:
    df = pd.DataFrame(data_fase1)
else:
    df = pd.DataFrame(data_fase2)

# --- Normalización de los Datos ---
# Se normalizan los datos para que todos estén en la misma escala (0 a 1)
labels = df.columns[1:]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # Repetir el primer ángulo para cerrar el polígono

df_normalized = df.copy()

# Valor minimo que se tomará como 0 en metricas de rendimiento:
MIN_VAL = 0.396

for col in labels:
    # Normalización para ejes de eficiencia (0 al máximo)
    if 'Eficiencia' in col:
        max_val = df[col].max()
        if max_val > 0:
            df_normalized[col] = df[col] / max_val
        else:
            df_normalized[col] = 0.0
    # Normalización con "zoom" para ejes de rendimiento/robustez
    else:
        max_val = df[col].max()
        min_val = MIN_VAL
        
        # Escalar los valores al rango [0, 1] dentro de la ventana [min_val, max_val]
        # Se usa .clip(lower=min_val) para que valores muy bajos no den negativos
        if max_val - min_val > 0:
            df_normalized[col] = (df[col].clip(lower=min_val) - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 1.0 # Si todos los valores son iguales
# --- Creación del Gráfico ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

colors = plt.cm.viridis(np.linspace(0, 1, len(df_normalized)))
for i, row in df_normalized.iterrows():
    values = row.drop('Modelo').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=row['Modelo'])
    ax.fill(angles, values, color=colors[i], alpha=0.25)

# Formatear el gráfico
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=18)

ax.set_ylim(0, 1.4)

if fase1:
    plt.title('Perfil Comparativo de Arquitecturas (Entrenamiento en RWF-2000)', size=20, color='black', y=1.1)
else:
    plt.title('Perfil Comparativo de Arquitecturas (Entrenamiento en RLVS)', size=20, color='black', y=1.1)
    
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),prop={'size': 18})

if fase1:
    plt.savefig('grafico_discusion_radar_fase1.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig('grafico_discusion_radar_fase2.png', dpi=300, bbox_inches='tight')
    
plt.show()