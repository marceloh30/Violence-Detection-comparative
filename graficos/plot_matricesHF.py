import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix_subplot(ax, cm, class_names, title):
    """
    Función para dibujar una matriz de confusión en un subplot (ax) específico.
    """
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 14}, cbar=False)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Valor Real', fontsize=12)
    ax.xaxis.set_ticklabels(class_names, rotation=0)
    ax.yaxis.set_ticklabels(class_names, rotation=90, va='center')

# --- Creación de la figura con 4 subplots (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
fig.suptitle('Comparativa de Robustez en Hockey Fights: ViViT vs. SlowFast', fontsize=20)

# --- DATOS DE LAS 4 MATRICES DE CONFUSIÓN ---
# Formato: [[TN, FP], [FN, TP]]

# -- Fila Superior: ViViT --
# (a) ViViT entrenado en RWF-2000, probado en Hockey
cm_vivit_fase1 = np.array([[474, 26], 
                           [370, 130]])

# (b) ViViT entrenado en RLVS, probado en Hockey
cm_vivit_fase2 = np.array([[80, 420], 
                           [9, 491]])


# -- Fila Inferior: SlowFast --
# (c) SlowFast entrenado en RWF-2000, probado en Hockey
cm_slowfast_fase1 = np.array([[461, 39], 
                              [173, 327]])

# (d) SlowFast entrenado en RLVS, probado en Hockey
cm_slowfast_fase2 = np.array([[442, 58], 
                              [131, 369]])


class_names = ['No Violento', 'Violento']

# --- Dibujar cada matriz en su subplot ---

# Fila superior para ViViT
plot_confusion_matrix_subplot(axes[0, 0], cm_vivit_fase1, class_names, '(a) ViViT (Ent. en RWF-2000)')
plot_confusion_matrix_subplot(axes[0, 1], cm_vivit_fase2, class_names, '(b) ViViT (Ent. en RLVS)')

# Fila inferior para SlowFast
plot_confusion_matrix_subplot(axes[1, 0], cm_slowfast_fase1, class_names, '(c) SlowFast (Ent. en RWF-2000)')
plot_confusion_matrix_subplot(axes[1, 1], cm_slowfast_fase2, class_names, '(d) SlowFast (Ent. en RLVS)')


# Ajustar el layout y guardar
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('cm_vivit_vs_slowfast_hockey.png', dpi=300)
plt.show()
