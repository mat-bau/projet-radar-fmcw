import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

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
                       title="Range-Doppler map", 
                       dynamic_range=60, 
                       save_path=None):
    """
    Affiche une carte Range-Doppler
    
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
    """
    plt.figure(figsize=(12, 8))
    vmax = np.max(range_doppler_map)
    vmin = vmax - dynamic_range
    map_to_plot = range_doppler_map[:len(velocity_axis), :len(range_axis)]
    
    # colormap personnalisé bleu
    #cmap_blue = LinearSegmentedColormap.from_list('BlueMap', 
    #                                            [(0, 'white'), 
    #                                           (0.2, 'lightblue'),
    #                                           (0.4, 'dodgerblue'),
    #                                           (0.6, 'mediumblue'),
    #                                           (0.8, 'darkblue'),
    #                                           (1, 'black')])
        
    plt.pcolormesh(range_axis, velocity_axis, map_to_plot, 
                   cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Vitesse (m/s)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3) # ligne vitesse nulle
    
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
    
    # Créer une grille pour les axes X et Y
    R, V = np.meshgrid(range_axis, velocity_axis)
    
    # Créer la figure 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Limiter la plage dynamique pour une meilleure visualisation
    vmax = np.max(range_doppler_map)
    vmin = vmax - 60
    Z = np.maximum(range_doppler_map[:len(velocity_axis), :len(range_axis)], vmin)
    
    # Tracer la surface
    surf = ax.plot_surface(R, V, Z, cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Ajouter une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Magnitude (dB)')
    
    # Étiquettes des axes
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Vitesse (m/s)')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title(title)
    
    # Régler l'angle de vue
    ax.view_init(elev=30, azim=225)
    
    plt.tight_layout()
    
    if save_path:
        # Créer le dossier s'il n'existe pas
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return fig