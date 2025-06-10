import matplotlib.pyplot as plt
import numpy as np

# --- Datos Completos para RLVS (Cargados a Mano) ---
results_data_rlvs = {
    'ViViT': {
        'epochs_run': [1, 2, 3, 4, 5],
        'train_loss': [0.168, 0.059, 0.043, 0.017, 0.025],
        'val_loss': [0.101, 0.098, 0.055, 0.031, 0.054],
        'val_f1': [0.963, 0.973, 0.980, 0.992, 0.985]
    },
    'I3D': {
        'epochs_run': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'train_loss': [0.341, 0.198, 0.163, 0.109, 0.101, 0.093, 0.067, 0.066, 0.066, 0.054],
        'val_loss': [0.311, 0.199, 0.184, 0.404, 0.184, 0.144, 0.204, 0.158, 0.173, 0.116],
        'val_f1': [0.907, 0.943, 0.945, 0.822, 0.956, 0.944, 0.949, 0.961, 0.952, 0.970]
    },
    'SlowFast': {
        'epochs_run': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'train_loss': [0.331, 0.177, 0.143, 0.117, 0.100, 0.075, 0.081, 0.066, 0.073, 0.069],
        'val_loss': [0.298, 0.132, 0.137, 0.172, 0.099, 0.127, 0.153, 0.151, 0.233, 0.147],
        'val_f1': [0.893, 0.942, 0.958, 0.950, 0.967, 0.949, 0.949, 0.948, 0.928, 0.962]
    },
    'TSM': {
        'epochs_run': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'train_loss': [0.504, 0.317, 0.185, 0.146, 0.107, 0.077, 0.068, 0.049, 0.039, 0.047],
        'val_loss': [0.194, 0.142, 0.107, 0.109, 0.088, 0.064, 0.073, 0.065, 0.062, 0.072],
        'val_f1': [0.913, 0.944, 0.970, 0.971, 0.972, 0.982, 0.979, 0.980, 0.982, 0.972]
    }
}

# --- Figura 3: F1-Score de Validación vs. Época (en RLVS) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig3, ax3 = plt.subplots(figsize=(12, 8))

best_f1_scores = {}
for model, data in results_data_rlvs.items():
    best_epoch_idx = np.argmax(data['val_f1'])
    best_f1 = data['val_f1'][best_epoch_idx]
    best_epoch = data['epochs_run'][best_epoch_idx]
    best_f1_scores[model] = (best_epoch, best_f1)

colors = plt.cm.viridis(np.linspace(0, 1, len(results_data_rlvs)))

for i, (model, data) in enumerate(results_data_rlvs.items()):
    ax3.plot(data['epochs_run'], data['val_f1'], marker='o', linestyle='-', label=model, color=colors[i])
    best_epoch, best_f1 = best_f1_scores[model]
    ax3.plot(best_epoch, best_f1, marker='*', markersize=15, color=colors[i], markeredgecolor='black', label=f'Mejor {model}: {best_f1:.3f}')

ax3.set_title('Evolución del F1-Score de Validación en RLVS', fontsize=16, fontweight='bold')
ax3.set_xlabel('Época', fontsize=12)
ax3.set_ylabel('F1-Score', fontsize=12)
ax3.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.set_xticks(range(1, 11))
ax3.tick_params(axis='both', which='major', labelsize=10)
# Ajustar el límite inferior del eje Y para apreciar mejor las diferencias
ax3.set_ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig('f1_score_validation_rlvs.png', dpi=300)
plt.show()

# --- Figura 4: Loss (Entrenamiento vs. Validación) vs. Época (en RLVS) ---
fig4, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True, sharex=True)
fig4.suptitle('Evolución del Loss (Entrenamiento vs. Validación) en RLVS', fontsize=18, fontweight='bold')

axes = axes.flatten()

for i, (model, data) in enumerate(results_data_rlvs.items()):
    ax = axes[i]
    ax.plot(data['epochs_run'], data['train_loss'], marker='o', linestyle='--', label='Loss de Entrenamiento', color='C0')
    ax.plot(data['epochs_run'], data['val_loss'], marker='o', linestyle='-', label='Loss de Validación', color='C1')
    ax.set_title(model, fontsize=14)
    ax.set_xlabel('Época', fontsize=10)
    if i % 2 == 0:
        ax.set_ylabel('Loss (Cross-Entropy)', fontsize=10)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(data['epochs_run'])

for j in range(len(results_data_rlvs), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig4.subplots_adjust(hspace=0.4) 
plt.savefig('loss_curves_rlvs.png', dpi=300)
plt.show()