import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix_subplot(ax, cm, class_names, title):
    """
    Función para dibujar una matriz de confusión en un subplot (ax) específico.
    """
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 16}, cbar=False)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Valor Real', fontsize=12)
    ax.xaxis.set_ticklabels(class_names, rotation=0)
    ax.yaxis.set_ticklabels(class_names, rotation=90, va='center')

# --- DATOS DE LAS MATRICES DE CONFUSIÓN ---
# Formato: [[TN, FP], [FN, TP]]
# Extraídos de los JSON para la MEJOR ÉPOCA de cada modelo en su set de validación

# --- FASE 1: Entrenamiento en RWF-2000 ---
phase1_cms = {
    'ViViT': np.array([[187, 13], [34, 166]]),    # Época 4
    'I3D': np.array([[178, 22], [39, 161]]),      # Época 2
    'SlowFast': np.array([[153, 47], [19, 181]]), # Época 5
    'TSM': np.array([[184, 16], [42, 158]])       # Época 6
}

# --- FASE 2: Entrenamiento en RLVS ---
phase2_cms = {
    'ViViT': np.array([[199, 1], [2, 198]]),      # Época 4
    'I3D': np.array([[192, 8], [4, 196]]),        # Época 10
    'SlowFast': np.array([[195, 5], [8, 192]]),   # Época 5
    'TSM': np.array([[196, 4], [3, 197]])         # Época 9
}

class_names = ['No Violento', 'Violento']

# --- GENERAR FIGURA PARA LA FASE 1 ---
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 13))
fig1.suptitle('Matrices de Confusión en Validación de RWF-2000 (Fase 1)', fontsize=20)

model_names = list(phase1_cms.keys())
axes_flat1 = axes1.flatten()

for i, model_name in enumerate(model_names):
    plot_confusion_matrix_subplot(axes_flat1[i], phase1_cms[model_name], class_names, model_name)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('cm_fase1_rwf2000.png', dpi=300)
plt.show()


# --- GENERAR FIGURA PARA LA FASE 2 ---
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 13))
fig2.suptitle('Matrices de Confusión en Validación de RLVS (Fase 2)', fontsize=20)

model_names = list(phase2_cms.keys())
axes_flat2 = axes2.flatten()

for i, model_name in enumerate(model_names):
    plot_confusion_matrix_subplot(axes_flat2[i], phase2_cms[model_name], class_names, model_name)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('cm_fase2_rlvs.png', dpi=300)
plt.show()