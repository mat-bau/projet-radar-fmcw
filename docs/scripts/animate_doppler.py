import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Configuration des paramètres de l'animation
fps = 30
duration = 10  # en secondes
frame_count = fps * duration

# Paramètres du signal
signal_freq = 0.7  # fréquence du signal émis (Hz)
speed_of_wave = 3.0  # vitesse de propagation de l'onde (unités arbitraires)

# Paramètres de l'objet
object_speed = 0.8  # vitesse de l'objet (unités arbitraires)
# Positions initiales distinctes pour les objets
approaching_initial_position = 8.0  # position initiale de l'objet qui s'approche
receding_initial_position = 2.0    # position initiale de l'objet qui s'éloigne

# Paramètres d'affichage
x_min, x_max = 0, 10
amplitude = 1.0

# Configuration de la figure et des sous-graphiques, cette fois avec 2 graphes côte à côte
fig, (ax_approaching, ax_receding) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Visualisation de l'effet Doppler", fontsize=16)

# Configuration des axes pour les deux scénarios
for ax, title in zip([ax_approaching, ax_receding], 
                     ["Objet qui s'approche", "Objet qui s'éloigne"]):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-amplitude*1.2, amplitude*1.2)
    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

# Points d'observation (position du radar/émetteur)
observer_position = 0.5

# Émetteur/récepteur (radar) représenté par un point noir
ax_approaching.plot(observer_position, 0, 'ko', markersize=8, label='Émetteur/Récepteur')
ax_receding.plot(observer_position, 0, 'ko', markersize=8, label='Émetteur/Récepteur')

# Initialisation des courbes pour les signaux émis et réfléchis
x = np.linspace(x_min, x_max, 1000)
emitted_line_approaching, = ax_approaching.plot([], [], 'b-', lw=2, label='Signal émis')
reflected_line_approaching, = ax_approaching.plot([], [], 'g-', lw=2, label='Signal réfléchi')
object_approaching = ax_approaching.axvline(x=approaching_initial_position, color='r', lw=3, label='Objet')

emitted_line_receding, = ax_receding.plot([], [], 'b-', lw=2, label='Signal émis')
reflected_line_receding, = ax_receding.plot([], [], 'g-', lw=2, label='Signal réfléchi')
object_receding = ax_receding.axvline(x=receding_initial_position, color='r', lw=3, label='Objet')

# Marqueurs pour indiquer la direction du mouvement
# Flèche de direction pour l'objet s'approchant (vers la gauche)
arrow_approaching = ax_approaching.annotate('', 
                                          xy=(approaching_initial_position-1, 0.8*amplitude), 
                                          xytext=(approaching_initial_position, 0.8*amplitude),
                                          arrowprops=dict(arrowstyle='->', color='r', lw=2))

# Flèche de direction pour l'objet s'éloignant (vers la droite)
arrow_receding = ax_receding.annotate('', 
                                    xy=(receding_initial_position+1, 0.8*amplitude), 
                                    xytext=(receding_initial_position, 0.8*amplitude),
                                    arrowprops=dict(arrowstyle='->', color='r', lw=2))

# Ajout des légendes
ax_approaching.legend(loc='upper right')
ax_receding.legend(loc='upper right')

# Texte pour afficher les fréquences
approaching_freq_text = ax_approaching.text(0.02, 0.92, '', transform=ax_approaching.transAxes)
receding_freq_text = ax_receding.text(0.02, 0.92, '', transform=ax_receding.transAxes)


def init():
    """Initialisation de l'animation"""
    emitted_line_approaching.set_data([], [])
    reflected_line_approaching.set_data([], [])
    emitted_line_receding.set_data([], [])
    reflected_line_receding.set_data([], [])
    approaching_freq_text.set_text('')
    receding_freq_text.set_text('')
    return (emitted_line_approaching, reflected_line_approaching, object_approaching,
            emitted_line_receding, reflected_line_receding, object_receding,
            approaching_freq_text, receding_freq_text)

def update(frame):
    """Mise à jour de l'animation pour chaque frame"""
    t = frame / fps  # temps en secondes
    
    # Mise à jour des positions des objets (approche et éloignement)
    approaching_position = max(observer_position, approaching_initial_position - object_speed * t)
    receding_position = min(x_max, receding_initial_position + object_speed * t)
    
    # Mise à jour des lignes verticales représentant les objets
    object_approaching.set_xdata([approaching_position, approaching_position])
    object_receding.set_xdata([receding_position, receding_position])
    
    # Mise à jour des flèches de direction
    arrow_approaching.xy = (approaching_position-1, 0.8*amplitude)
    arrow_approaching.xytext = (approaching_position, 0.8*amplitude)
    
    arrow_receding.xy = (receding_position+1, 0.8*amplitude)
    arrow_receding.xytext = (receding_position, 0.8*amplitude)
    
    # Calcul de l'effet Doppler pour les deux scénarios
    # Approche: fréquence augmente (facteur > 1)
    approaching_reflected_freq = signal_freq * (1 + object_speed / speed_of_wave)
    
    # Éloignement: fréquence diminue (facteur < 1)
    receding_reflected_freq = signal_freq * (1 - object_speed / speed_of_wave)
    
    # Calcul des signaux
    # Pour chaque position x, on calcule à quel moment le signal est présent
    t_emission = np.maximum(0, t - (x - observer_position) / speed_of_wave)
    
    # MODIFICATION: Le signal émis s'arrête aux objets
    # Pour l'objet qui s'approche
    emitted_signal_approaching = np.zeros_like(x)
    # Le signal émis n'existe que jusqu'à l'objet
    mask_emitted_approaching = (x >= observer_position) & (x <= approaching_position)
    emitted_signal_approaching[mask_emitted_approaching] = amplitude * np.sin(
        2 * np.pi * signal_freq * t_emission[mask_emitted_approaching]
    )
    
    # Pour l'objet qui s'éloigne
    emitted_signal_receding = np.zeros_like(x)
    # Le signal émis n'existe que jusqu'à l'objet
    mask_emitted_receding = (x >= observer_position) & (x <= receding_position)
    emitted_signal_receding[mask_emitted_receding] = amplitude * np.sin(
        2 * np.pi * signal_freq * t_emission[mask_emitted_receding]
    )
    
    # Pour les signaux réfléchis, nous simulons un délai basé sur la distance de l'objet
    # et un changement de fréquence dû à l'effet Doppler
    
    # Cas de l'objet qui s'approche
    reflected_signal_approaching = np.zeros_like(x)
    mask_approaching = (x >= observer_position) & (x <= approaching_position)
    if np.any(mask_approaching):
        # Temps pour atteindre l'objet puis revenir à la position x
        time_to_object_approaching = (approaching_position - observer_position) / speed_of_wave
        time_from_object_to_x_approaching = (approaching_position - x[mask_approaching]) / speed_of_wave
        
        # Temps total avec effet Doppler pour la réflexion
        reflected_time_approaching = t - time_to_object_approaching - time_from_object_to_x_approaching
        valid_mask = reflected_time_approaching >= 0
        
        if np.any(valid_mask):
            indices = np.where(mask_approaching)[0][valid_mask]
            reflected_signal_approaching[indices] = amplitude * 0.7 * np.sin(
                2 * np.pi * approaching_reflected_freq * reflected_time_approaching[valid_mask]
            )
    
    # Cas de l'objet qui s'éloigne
    reflected_signal_receding = np.zeros_like(x)
    mask_receding = (x >= observer_position) & (x <= receding_position)
    if np.any(mask_receding):
        # Temps pour atteindre l'objet puis revenir à la position x
        time_to_object_receding = (receding_position - observer_position) / speed_of_wave
        time_from_object_to_x_receding = (receding_position - x[mask_receding]) / speed_of_wave
        
        # Temps total avec effet Doppler pour la réflexion
        reflected_time_receding = t - time_to_object_receding - time_from_object_to_x_receding
        valid_mask = reflected_time_receding >= 0
        
        if np.any(valid_mask):
            indices = np.where(mask_receding)[0][valid_mask]
            reflected_signal_receding[indices] = amplitude * 0.7 * np.sin(
                2 * np.pi * receding_reflected_freq * reflected_time_receding[valid_mask]
            )
    
    # Mise à jour des courbes
    emitted_line_approaching.set_data(x, emitted_signal_approaching)
    reflected_line_approaching.set_data(x, reflected_signal_approaching)
    
    emitted_line_receding.set_data(x, emitted_signal_receding)
    reflected_line_receding.set_data(x, reflected_signal_receding)
    
    # Mise à jour des textes d'information
    approaching_freq_text.set_text(f'Freq. émise: {signal_freq:.2f} Hz\nFreq. réfléchie: {approaching_reflected_freq:.2f} Hz')
    receding_freq_text.set_text(f'Freq. émise: {signal_freq:.2f} Hz\nFreq. réfléchie: {receding_reflected_freq:.2f} Hz')
    
    return (emitted_line_approaching, reflected_line_approaching, object_approaching,
            emitted_line_receding, reflected_line_receding, object_receding,
            approaching_freq_text, receding_freq_text)

# Création de l'animation
ani = FuncAnimation(fig, update, frames=frame_count, init_func=init, blit=True, interval=1000/fps)

# Création du dossier de sortie si nécessaire
output_dir = 'docs/animations'
os.makedirs(output_dir, exist_ok=True)

# Enregistrement de l'animation au format GIF
output_path = os.path.join(output_dir, 'doppler_effect_comparison.gif')
print(f"Enregistrement de l'animation dans {output_path}")

# Configuration pour réduire la taille du fichier GIF
ani.save(output_path, writer='pillow', fps=fps, dpi=100)

print(f"Animation enregistrée avec succès dans {output_path}")

# Afficher l'animation (décommenter pour voir en direct)
plt.show()