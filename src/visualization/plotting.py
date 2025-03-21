import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import scipy.signal as signal

def plot_range_profile(range_profile, range_axis, title="Profil de distance", save_path=None):
    """
    Affiche un profil de distance
    
    Parameters:
    -----------
    range_profile : ndarray
        Profil de distance en dB
    range_axis : ndarray
        Axe de distance en mètres
    title : str
        Titre du graphique
    save_path : str, optional
        Chemin où sauvegarder l'image
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range_axis, range_profile[:len(range_axis)])
    plt.xlabel('Distance (m)')
    plt.ylabel('Magnitude (dB)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # lignes verticales tous les 10m ca me semble mieux
    max_range = range_axis[-1]
    for d in range(0, int(max_range)+10, 10):
        plt.axvline(x=d, color='r', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        # Créer le dossier s'il n'existe pas
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return plt.gcf()

def plot_range_doppler(range_doppler_map, range_axis, velocity_axis, 
                       title="Range-Doppler Map", 
                       dynamic_range=40, 
                       save_path=None,
                       cmap='jet',
                       remove_static_components=True):
    """
    Affiche une Range-Doppler map en supprimant des composantes statiques
    
    Parameters:
    -----------
    range_doppler_map : ndarray
        Carte Range-Doppler en dB
    range_axis : ndarray
        Axe de distance en mètres
    velocity_axis : ndarray
        Axe de vitesse en m/s
    title : str
        Titre du graphique
    dynamic_range : int
        Plage dynamique en dB pour la visualisation
    save_path : str, optional
        Chemin où sauvegarder l'image
    cmap : str ou matplotlib.colors.Colormap, optional
        La colormap à utiliser (défaut: 'jet')
    remove_static_components : bool, optional
        Si True, supprime les moyennes des colonnes pour éliminer les composantes statiques
    """
    # Créer une copie pour ne pas modifier les données originales
    map_data = range_doppler_map.copy()
    
    # Supprimer les composantes statiques en soustrayant la moyenne de chaque colonne
    if remove_static_components:
        # Note: Cette opération doit être effectuée sur les données linéaires, pas en dB
        # Si les données sont déjà en dB, il faut les convertir en linéaire d'abord
        # Ici, nous supposons qu'elles sont déjà en dB
        linear_data = 10**(map_data/20)  # Conversion de dB à linéaire
        
        # Soustraire la moyenne de chaque colonne
        column_means = np.mean(linear_data, axis=0)
        linear_data_no_static = linear_data - column_means[np.newaxis, :]
        
        # Remettre en dB, en évitant les valeurs négatives
        map_data = 20 * np.log10(np.maximum(np.abs(linear_data_no_static), 1e-10))
    
    # S'assurer que les dimensions correspondent
    map_to_plot = map_data[:len(velocity_axis), :len(range_axis)]
    
    # Créer la figure
    plt.figure(figsize=(12, 8))
    
    # Calculer les limites de couleur basées sur la plage dynamique
    vmax = np.max(map_to_plot)
    vmin = vmax - dynamic_range
    
    plt.pcolormesh(range_axis, velocity_axis, map_to_plot, 
                   cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Vitesse (m/s)')
    plt.title(title)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Créer le dossier s'il n'existe pas
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return plt.gcf()

def plot_spectrogram(data, params, title="Spectrogramme", save_path=None):
    """
    Affiche un spectrogramme des données radar
    
    Parameters:
    -----------
    data : ndarray
        Données complexes de forme (num_chirps, samples_per_chirp)
    params : dict
        Dictionnaire des paramètres
    title : str
        Titre du graphique
    save_path : str, optional
        Chemin où sauvegarder l'image
    """
    # paramètres du spectrogramme
    fs = params['sample_rate']
    f, t, Sxx = signal.spectrogram(data.flatten(), fs=fs, 
                                   nperseg=256, noverlap=128,
                                   scaling='spectrum')
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f/1e6, 10 * np.log10(Sxx + 1e-15), shading='gouraud')
    plt.ylabel('Fréquence (MHz)')
    plt.xlabel('Temps (s)')
    plt.title(title)
    plt.colorbar(label='Puissance (dB)')
    
    plt.tight_layout()
    
    if save_path:
        # Créer le dossier s'il n'existe pas
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return plt.gcf()

def visualize_3d_range_doppler(range_doppler_map, range_axis, velocity_axis, 
                              title="Visualisation 3D Range-Doppler",
                              save_path=None):
    """
    Crée une visualisation 3D de la carte Range-Doppler
    
    Parameters:
    -----------
    range_doppler_map : ndarray
        Carte Range-Doppler en dB
    range_axis : ndarray
        Axe de distance en mètres
    velocity_axis : ndarray
        Axe de vitesse en m/s
    title : str
        Titre du graphique
    save_path : str, optional
        Chemin où sauvegarder l'image
    """
    from mpl_toolkits.mplot3d import Axes3D

    R, V = np.meshgrid(range_axis, velocity_axis)
    
    # Créer la figure 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    vmax = np.max(range_doppler_map)
    vmin = vmax - 60
    Z = np.maximum(range_doppler_map[:len(velocity_axis), :len(range_axis)], vmin)
    
    surf = ax.plot_surface(R, V, Z, cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.8)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Magnitude (dB)')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Vitesse (m/s)')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title(title)
    ax.view_init(elev=30, azim=225)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return fig

def create_combined_visualization(range_doppler_map, range_profile, range_axis, velocity_axis, 
                                  title="Visualisation combinée", dynamic_range=40):
    """
    Crée une visualisation combinée avec la Range-Doppler map à droite
    et les profils de distance et de vitesse à gauche
    """
    # Créer une figure avec une disposition en grille
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])
    
    # Axes pour le profil de distance (en haut à gauche)
    ax_range = fig.add_subplot(gs[0, 0])
    
    # Correction: s'assurer que les dimensions correspondent
    min_len = min(len(range_axis), len(range_profile))
    ax_range.plot(range_axis[:min_len], range_profile[:min_len])
    
    ax_range.set_xlabel('Distance (m)')
    ax_range.set_ylabel('Magnitude (dB)')
    ax_range.set_title('Profil de distance')
    ax_range.grid(True, linestyle='--', alpha=0.7)
    
    # Axes pour le profil de vitesse (en bas à gauche)
    ax_velocity = fig.add_subplot(gs[1, 0])
    # Calculer le profil de vitesse en moyennant la carte Range-Doppler le long de l'axe distance
    velocity_profile = np.mean(range_doppler_map, axis=1)
    ax_velocity.plot(velocity_axis, velocity_profile[:len(velocity_axis)])
    ax_velocity.set_xlabel('Vitesse (m/s)')
    ax_velocity.set_ylabel('Magnitude (dB)')
    ax_velocity.set_title('Profil de vitesse')
    ax_velocity.grid(True, linestyle='--', alpha=0.7)
    ax_velocity.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # Axes pour la Range-Doppler map (à droite, sur toute la hauteur)
    ax_rdm = fig.add_subplot(gs[:, 1])
    
    # Limiter la plage dynamique
    vmax = np.max(range_doppler_map)
    vmin = vmax - dynamic_range
    
    mesh = ax_rdm.pcolormesh(range_axis, velocity_axis, range_doppler_map[:len(velocity_axis), :len(range_axis)], 
                            cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    fig.colorbar(mesh, ax=ax_rdm, label='Magnitude (dB)')
    ax_rdm.set_xlabel('Distance (m)')
    ax_rdm.set_ylabel('Vitesse (m/s)')
    ax_rdm.set_title('Range-Doppler Map')
    ax_rdm.grid(True, linestyle='--', alpha=0.5)
    ax_rdm.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Titre global
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig, (ax_range, ax_velocity, ax_rdm)

def plot_dual_range_doppler(unbalanced_rdm, balanced_rdm, range_axis, velocity_axis, 
                           title="Comparaison des cartes Range-Doppler", 
                           dynamic_range=40, 
                           save_path=None,
                           interactive=True):
    """
    Affiche deux cartes Range-Doppler côte à côte avec fonctionnalité de survol pour voir l'amplitude
    
    Parameters:
    -----------
    unbalanced_rdm : ndarray
        Carte Range-Doppler des signaux déséquilibrés (canaux 0,1)
    balanced_rdm : ndarray
        Carte Range-Doppler des signaux équilibrés (canaux 2,3)
    range_axis : ndarray
        Axe de distance en mètres
    velocity_axis : ndarray
        Axe de vitesse en m/s
    title : str
        Titre du graphique
    dynamic_range : int
        Plage dynamique en dB pour la visualisation
    save_path : str, optional
        Chemin où sauvegarder l'image
    interactive : bool
        Activer l'interactivité pour afficher les valeurs au survol
    
    Returns:
    --------
    fig : Figure
        Figure matplotlib
    """
    # Création de la figure avec 2 sous-graphes horizontaux
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # S'assurer que les dimensions correspondent
    unbalanced_plot = unbalanced_rdm[:len(velocity_axis), :len(range_axis)]
    balanced_plot = balanced_rdm[:len(velocity_axis), :len(range_axis)]
    
    # Calculer la plage dynamique commune pour une meilleure comparaison
    vmax = max(np.max(unbalanced_plot), np.max(balanced_plot))
    vmin = vmax - dynamic_range
    
    # Tracé de la carte Range-Doppler déséquilibrée (canaux 0,1)
    im1 = axes[0].pcolormesh(range_axis, velocity_axis, unbalanced_plot, 
                           cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('Signal déséquilibré (canaux 0,1)')
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Vitesse (m/s)')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    fig.colorbar(im1, ax=axes[0], label='Magnitude (dB)')
    
    # Tracé de la carte Range-Doppler équilibrée (canaux 2,3)
    im2 = axes[1].pcolormesh(range_axis, velocity_axis, balanced_plot, 
                           cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title('Signal équilibré (canaux 2,3)')
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Vitesse (m/s)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    fig.colorbar(im2, ax=axes[1], label='Magnitude (dB)')
    
    # Titre global
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Créer des annotations pour l'affichage des valeurs
    annot1 = axes[0].annotate("", xy=(0,0), xytext=(10,10),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.7),
                            arrowprops=dict(arrowstyle="->"))
    annot1.set_visible(False)
    
    annot2 = axes[1].annotate("", xy=(0,0), xytext=(10,10),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.7),
                            arrowprops=dict(arrowstyle="->"))
    annot2.set_visible(False)
    
    # Créer des événements interactifs si demandé
    if interactive:
        def hover(event):
            # Fonction appelée lors du survol de la souris
            if event.inaxes == axes[0]:
                update_annot(event, annot1, im1, unbalanced_plot, range_axis, velocity_axis)
            elif event.inaxes == axes[1]:
                update_annot(event, annot2, im2, balanced_plot, range_axis, velocity_axis)
                
        def update_annot(event, annot, im, data, ranges, velocities):
            # Met à jour l'annotation avec la valeur sous le curseur
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # Trouver les indices les plus proches dans les axes
                range_idx = np.abs(ranges - x).argmin()
                vel_idx = np.abs(velocities - y).argmin()
                
                # S'assurer que les indices sont dans les limites
                if 0 <= vel_idx < len(velocities) and 0 <= range_idx < len(ranges):
                    val = data[vel_idx, range_idx]
                    
                    # Mettre à jour l'annotation
                    annot.xy = (x, y)
                    text = f"Distance: {ranges[range_idx]:.1f}m\nVitesse: {velocities[vel_idx]:.1f}m/s\nAmplitude: {val:.1f}dB"
                    annot.set_text(text)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
            else:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                
        def leave_axes(event):
            # Cache les annotations quand la souris quitte les axes
            annot1.set_visible(False)
            annot2.set_visible(False)
            fig.canvas.draw_idle()
        
        # Connecter les événements
        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect("axes_leave_event", leave_axes)
    
    # Sauvegarde si un chemin est fourni
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return fig