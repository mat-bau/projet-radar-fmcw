import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

def create_range_doppler_map(frame_data, chirp_params, range_padding=4, doppler_padding=4, 
                            velocity_threshold=2.0, filter_method='mask', 
                            remove_zero_bin=False, normalize=True):
    """
    Crée une carte range-Doppler avec différentes options de traitement des vitesses.
    
    Args:
        frame_data: Données du frame radar
        chirp_params: Paramètres du chirp FMCW
        range_padding: Facteur de zero-padding pour la FFT de distance
        doppler_padding: Facteur de zero-padding pour la FFT Doppler
        velocity_threshold: Seuil de vitesse (en m/s) - les vitesses absolues inférieures
                           à ce seuil seront traitées selon filter_method
        filter_method: 'mask' pour masquer les valeurs, 'remove' pour supprimer ces bins
        remove_zero_bin: Si True, met à zéro spécifiquement le bin de vitesse le plus proche de zéro
        normalize: Si True, normalise la carte range-Doppler avant application du log
    """
    f0, B, Ms, Mc, Ts, Tc = chirp_params
    Ms = int(Ms)
    Mc = int(Mc)

    I1 = frame_data[0, :]
    Q1 = frame_data[1, :]
    
    # Attention aux signes ici on utilise +1j et non -1j
    complex_signal = I1 + 1j * Q1

    expected_size = Mc * Ms
    actual_size = len(complex_signal)

    if actual_size != expected_size:
        total_samples = len(complex_signal)
        if total_samples % Mc == 0:
            Ms = total_samples // Mc
        else:
            Ms = total_samples // Mc
            complex_signal = complex_signal[:Ms * Mc]

    signal_matrix = np.reshape(complex_signal, (Mc, Ms))

    range_window = windows.hann(Ms)
    doppler_window = windows.hann(Mc)

    windowed_signal = signal_matrix * doppler_window[:, np.newaxis]
    windowed_signal = windowed_signal * range_window[np.newaxis, :]

    n_range = Ms * range_padding
    n_doppler = Mc * doppler_padding

    # FFT pour la distance (deuxième dimension)
    range_fft = np.fft.fft(windowed_signal, n=n_range, axis=1)
    
    # FFT pour la vitesse (première dimension) et fftshift pour centrer le spectre
    range_doppler_map = np.fft.fftshift(np.fft.fft(range_fft, n=n_doppler, axis=0), axes=0)

    # Calcul de l'amplitude du spectre
    if normalize:
        # Normalisation pour éviter les problèmes d'échelle
        range_doppler_map_abs = np.abs(range_doppler_map)
        max_val = np.max(range_doppler_map_abs) if np.max(range_doppler_map_abs) > 0 else 1
        range_doppler_map_normalized = range_doppler_map_abs / max_val
        range_doppler_map_db = 20 * np.log10(range_doppler_map_normalized + 1e-10)
    else:
        range_doppler_map_db = 20 * np.log10(np.abs(range_doppler_map) + 1e-10)

    # calcul des axes
    c = 3e8               # Vitesse de la lumière en m/s
    lambda_wave = c / f0  # Longueur d'onde

    range_res = c / (2 * B)  # Résolution en distance
    max_range = range_res * n_range / 2
    range_axis = np.linspace(0, max_range, n_range)

    doppler_res = lambda_wave / (2 * Tc * Mc)  # Résolution en vitesse
    max_velocity = doppler_res * n_doppler / 2
    velocity_axis = np.linspace(-max_velocity, max_velocity, n_doppler)

    # Option: Supprimer spécifiquement le bin de vitesse zéro
    if remove_zero_bin:
        # L'indice de la vitesse la plus proche de zéro
        zero_velocity_idx = np.argmin(np.abs(velocity_axis))
        # Mettre cette colonne à la valeur minimale de la carte
        min_value = np.min(range_doppler_map_db)
        range_doppler_map_db[:, zero_velocity_idx] = min_value
        
    # Masque pour les vitesses inférieures au seuil (en valeur absolue)
    low_vel_mask = np.abs(velocity_axis) < velocity_threshold
    high_vel_mask = np.abs(velocity_axis) >= velocity_threshold
    
    if filter_method == 'mask':
        # OPTION 1: Masquer les valeurs en les remplaçant par le minimum
        min_value = np.min(range_doppler_map_db)
        range_doppler_map_db_filtered = range_doppler_map_db.copy()
        range_doppler_map_db_filtered[low_vel_mask, :] = min_value
        velocity_axis_filtered = velocity_axis
    elif filter_method == 'remove':
        # OPTION 2: Supprimer complètement ces bins
        range_doppler_map_db_filtered = range_doppler_map_db[high_vel_mask, :]
        velocity_axis_filtered = velocity_axis[high_vel_mask]
    else:
        # Par défaut, ne rien filtrer
        range_doppler_map_db_filtered = range_doppler_map_db
        velocity_axis_filtered = velocity_axis

    return range_doppler_map_db_filtered, range_axis, velocity_axis_filtered

def plot_multiple_range_doppler_maps(radar_data, chirp_params, num_frames=6, max_range=30, 
                                    max_velocity=20, velocity_threshold=2.0, filter_method='mask',
                                    remove_zero_bin=False, normalize=True, dynamic_range=40):
    """
    Affiche plusieurs cartes range-Doppler avec diverses options de filtrage.
    
    Args:
        radar_data: Données radar pour plusieurs frames
        chirp_params: Paramètres du chirp FMCW
        num_frames: Nombre de frames à afficher
        max_range: Portée maximale à afficher (en m)
        max_velocity: Vitesse maximale à afficher (en m/s)
        velocity_threshold: Seuil de vitesse pour le filtrage (en m/s)
        filter_method: Méthode de filtrage ('mask' ou 'remove')
        remove_zero_bin: Si True, supprime spécifiquement le bin de vitesse zéro
        normalize: Si True, normalise les données
        dynamic_range: Plage dynamique en dB pour l'affichage
    """
    # Disposition optimale pour les subplots
    num_frames = min(num_frames, radar_data.shape[0])
    if num_frames <= 3:
        nrows, ncols = 1, num_frames
    else:
        nrows = (num_frames + 2) // 3  # Arrondie vers le haut
        ncols = min(3, num_frames)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))  
    if num_frames == 1:
        axes = np.array([axes])  # Si 1 seul subplot
    axes = axes.flatten()  

    vmin_global = None
    vmax_global = None
    all_maps = []
    all_range_axes = []
    all_velocity_axes = []

    for i in range(num_frames):
        frame_data = radar_data[i]
        rd_map, range_axis, velocity_axis = create_range_doppler_map(
            frame_data, chirp_params, range_padding=4, doppler_padding=4, 
            velocity_threshold=velocity_threshold, filter_method=filter_method,
            remove_zero_bin=remove_zero_bin, normalize=normalize
        )
        
        all_maps.append(rd_map)
        all_range_axes.append(range_axis)
        all_velocity_axes.append(velocity_axis)

        curr_max = np.max(rd_map)
        if vmax_global is None or curr_max > vmax_global:
            vmax_global = curr_max

    # Définir les limites de couleur pour toutes les cartes
    vmin_global = vmax_global - dynamic_range

    for i in range(num_frames):
        ax = axes[i]
        rd_map = all_maps[i]
        range_axis = all_range_axes[i]
        velocity_axis = all_velocity_axes[i]

        transposed_map = rd_map.T  # Maintien de la transposition

        # Déterminer l'étendue en fonction de la méthode de filtrage
        if filter_method == 'remove':
            vel_min = np.min(velocity_axis)
            vel_max = np.max(velocity_axis)
            extent = [vel_min, vel_max, 0, range_axis[-1]]
        else:
            if max_velocity is not None:
                extent = [-max_velocity, max_velocity, 0, range_axis[-1]]
            else:
                extent = [velocity_axis[0], velocity_axis[-1], 0, range_axis[-1]]
        
        im = ax.imshow(transposed_map, aspect='auto', origin='lower', 
                      extent=extent,
                      cmap='jet', vmin=vmin_global, vmax=vmax_global,
                      interpolation='bilinear')
        
        ax.set_xlabel('Vitesse (m/s)')
        if i % ncols == 0:  # Ajouter le ylabel uniquement pour la première colonne
            ax.set_ylabel('Distance (m)')
        ax.set_title(f'Frame {i}')
        
        # Limiter l'affichage si besoin et selon la méthode
        if filter_method == 'remove':
            ax.set_xlim([vel_min, vel_max])
        else:
            if max_velocity is not None:
                ax.set_xlim([-max_velocity, max_velocity])
        
        if max_range is not None:
            ax.set_ylim([0, max_range])
        else:
            ax.set_ylim([0, range_axis[-1]])
        
        ax.grid(True, color='w', linestyle='-', alpha=0.2)

    # Masquer les axes supplémentaires si nécessaire
    for i in range(num_frames, len(axes)):
        axes[i].set_visible(False)

    # Ajouter la barre de couleur
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Magnitude (dB)')

    # Titre en fonction des options de filtrage
    title_components = []
    if remove_zero_bin:
        title_components.append("bin zéro supprimé")
    
    if filter_method != 'none':
        action = "masquées" if filter_method == 'mask' else "supprimées"
        title_components.append(f"vitesses < {velocity_threshold} m/s {action}")
    
    title = "Range-Doppler map"
    if title_components:
        title += f" ({', '.join(title_components)})"
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Nom de fichier incluant les paramètres
    filename_parts = ['RDM']
    if remove_zero_bin:
        filename_parts.append('noZero')
    if filter_method != 'none':
        filename_parts.append(f"{filter_method}{velocity_threshold}")
    
    filename = 'images/' + '_'.join(filename_parts) + '.pdf'
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    file_path = 'MS1-FMCW.npz'
    
    try:
        data_dict = np.load(file_path, allow_pickle=True)
        
        radar_data = data_dict["data"]
        chirp_params = tuple(data_dict["chirp"])

        print(f"Fichier chargé avec succès! Dimensions des données: {radar_data.shape}")
        print(f"Paramètres du chirp: {chirp_params}")

        # Afficher les statistiques pour chaque trame
        for i in range(min(6, radar_data.shape[0])):
            print(f"Frame {i} - Min: {np.min(radar_data[i])}, Max: {np.max(radar_data[i])}")
        
        # Configuration des paramètres
        velocity_threshold = 1.5      # Seuil de vitesse en m/s
        max_range = 30               # Distance maximale à afficher (m)
        max_velocity = 20            # Vitesse maximale à afficher (m/s)
        dynamic_range = 40           # Plage dynamique en dB
        
        # Exemple 1: Filtrage standard avec seuil de vitesse
        print("\nGénération des cartes avec masquage des vitesses faibles...")
        plot_multiple_range_doppler_maps(
            radar_data, 
            chirp_params, 
            num_frames=min(6, radar_data.shape[0]), 
            max_range=max_range,  
            max_velocity=max_velocity,
            velocity_threshold=velocity_threshold,
            filter_method='mask',
            remove_zero_bin=False,
            normalize=False,
            dynamic_range=dynamic_range
        )

        # Exemple 1 mais avec normalisation
        plot_multiple_range_doppler_maps(
            radar_data, 
            chirp_params, 
            num_frames=min(6, radar_data.shape[0]), 
            max_range=max_range,  
            max_velocity=max_velocity,
            velocity_threshold=velocity_threshold,
            filter_method='mask',
            remove_zero_bin=False,
            normalize=True,
            dynamic_range=dynamic_range
        )
        
        # Exemple 2: Suppression uniquement du bin de vitesse zéro
        print("\nGénération des cartes avec suppression du bin de vitesse zéro uniquement...")
        plot_multiple_range_doppler_maps(
            radar_data, 
            chirp_params, 
            num_frames=min(6, radar_data.shape[0]), 
            max_range=max_range,  
            max_velocity=max_velocity,
            velocity_threshold=velocity_threshold,
            filter_method='none',  # Pas de filtrage par seuil
            remove_zero_bin=True,  # Suppression du bin zéro uniquement
            normalize=False,
            dynamic_range=dynamic_range
        )
        
        # Exemple 3: Combinaison des deux 
        print("\nGénération des cartes avec combinaison des deux approches...")
        plot_multiple_range_doppler_maps(
            radar_data, 
            chirp_params, 
            num_frames=min(6, radar_data.shape[0]), 
            max_range=max_range,  
            max_velocity=max_velocity,
            velocity_threshold=velocity_threshold,
            filter_method='mask',
            remove_zero_bin=True,
            normalize=False,
            dynamic_range=dynamic_range
        )
        
        print(f"\nCartes range-Doppler générées avec succès!")
        
    except Exception as e:
        import traceback
        print(f"Erreur lors du chargement ou du traitement du fichier: {e}")
        traceback.print_exc()