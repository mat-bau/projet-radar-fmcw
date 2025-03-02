import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

def create_range_doppler_map_configurable(frame_data, chirp_params, 
                                         use_range_fft=True,
                                         use_doppler_fft=True,
                                         range_padding=1, 
                                         doppler_padding=1, 
                                         use_windowing=True,
                                         velocity_threshold=2.0, 
                                         filter_method='mask', 
                                         remove_zero_bin=False, 
                                         normalize=False):
    """
    Crée une carte range-Doppler avec différentes options de traitement configurables.
    
    Args:
        frame_data: Données du frame radar
        chirp_params: Paramètres du chirp FMCW
        use_range_fft: Si True, utilise la FFT pour la dimension distance
        use_doppler_fft: Si True, utilise la FFT pour la dimension Doppler
        range_padding: Facteur de zero-padding pour la FFT de distance (1 = pas de padding)
        doppler_padding: Facteur de zero-padding pour la FFT Doppler (1 = pas de padding)
        use_windowing: Si True, applique les fenêtres de Hann
        velocity_threshold: Seuil de vitesse (en m/s) pour le filtrage
        filter_method: 'mask', 'remove' ou 'none'
        remove_zero_bin: Si True, supprime spécifiquement le bin de vitesse zéro
        normalize: Si True, normalise la carte range-Doppler
    """
    f0, B, Ms, Mc, Ts, Tc = chirp_params
    Ms = int(Ms)
    Mc = int(Mc)

    I1 = frame_data[0, :]
    Q1 = frame_data[1, :]
    
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
    
    # Application du fenêtrage (optionnel)
    if use_windowing:
        range_window = windows.hann(Ms)
        doppler_window = windows.hann(Mc)
        signal_matrix = signal_matrix * doppler_window[:, np.newaxis]
        signal_matrix = signal_matrix * range_window[np.newaxis, :]
    
    n_range = Ms * range_padding
    n_doppler = Mc * doppler_padding
    
    # Traitement de la dimension distance
    if use_range_fft:
        # FFT pour la distance (avec ou sans padding)
        range_processed = np.fft.fft(signal_matrix, n=n_range, axis=1)
    else:
        # Pas de FFT, uniquement un redimensionnement si padding demandé
        if range_padding > 1:
            # Redimensionner avec des zéros pour simuler l'effet visuel du padding
            range_processed = np.zeros((Mc, n_range), dtype=complex)
            range_processed[:, :Ms] = signal_matrix
        else:
            range_processed = signal_matrix
    
    # Traitement de la dimension Doppler
    if use_doppler_fft:
        # FFT pour la vitesse (avec ou sans padding et fftshift)
        range_doppler_map = np.fft.fftshift(np.fft.fft(range_processed, n=n_doppler, axis=0), axes=0)
    else:
        # Pas de FFT, uniquement un redimensionnement si padding demandé
        if doppler_padding > 1:
            # Redimensionner avec des zéros
            temp = np.zeros((n_doppler, n_range), dtype=complex)
            temp[:Mc, :] = range_processed
            # Simuler l'effet du fftshift
            mid = n_doppler // 2
            shifted = np.zeros((n_doppler, n_range), dtype=complex)
            shifted[mid:, :] = temp[:mid, :]
            shifted[:mid, :] = temp[mid:, :]
            range_doppler_map = shifted
        else:
            # Simuler l'effet du fftshift
            mid = Mc // 2
            range_doppler_map = np.zeros((Mc, n_range), dtype=complex)
            range_doppler_map[mid:, :] = range_processed[:mid, :]
            range_doppler_map[:mid, :] = range_processed[mid:, :]
    
    # Calcul de l'amplitude du spectre
    if normalize:
        # Normalisation pour éviter les problèmes d'échelle
        range_doppler_map_abs = np.abs(range_doppler_map)
        max_val = np.max(range_doppler_map_abs) if np.max(range_doppler_map_abs) > 0 else 1
        range_doppler_map_normalized = range_doppler_map_abs / max_val
        range_doppler_map_db = 20 * np.log10(range_doppler_map_normalized + 1e-10)
    else:
        range_doppler_map_db = 20 * np.log10(np.abs(range_doppler_map) + 1e-10)

    # Calcul des axes
    c = 3e8
    lambda_wave = c / f0

    if use_range_fft:
        range_res = c / (2 * B)
        max_range = range_res * n_range / 2
        range_axis = np.linspace(0, max_range, n_range)
    else:
        # Sans FFT, l'axe de distance est juste un indice
        range_axis = np.arange(n_range)
    
    if use_doppler_fft:
        doppler_res = lambda_wave / (2 * Tc * Mc)
        max_velocity = doppler_res * n_doppler / 2
        velocity_axis = np.linspace(-max_velocity, max_velocity, n_doppler)
    else:
        # Sans FFT, l'axe Doppler est juste un indice centré
        velocity_axis = np.arange(-n_doppler/2, n_doppler/2)

    # Option: Supprimer spécifiquement le bin de vitesse zéro
    if remove_zero_bin:
        zero_velocity_idx = np.argmin(np.abs(velocity_axis))
        min_value = np.min(range_doppler_map_db)
        range_doppler_map_db[:, zero_velocity_idx] = min_value
    
    # Traitement des vitesses faibles selon l'option choisie
    if filter_method != 'none':
        low_vel_mask = np.abs(velocity_axis) < velocity_threshold
        high_vel_mask = np.abs(velocity_axis) >= velocity_threshold
        
        if filter_method == 'mask':
            min_value = np.min(range_doppler_map_db)
            range_doppler_map_db_filtered = range_doppler_map_db.copy()
            range_doppler_map_db_filtered[low_vel_mask, :] = min_value
            velocity_axis_filtered = velocity_axis
        elif filter_method == 'remove':
            range_doppler_map_db_filtered = range_doppler_map_db[high_vel_mask, :]
            velocity_axis_filtered = velocity_axis[high_vel_mask]
        else:
            range_doppler_map_db_filtered = range_doppler_map_db
            velocity_axis_filtered = velocity_axis
    else:
        range_doppler_map_db_filtered = range_doppler_map_db
        velocity_axis_filtered = velocity_axis

    return range_doppler_map_db_filtered, range_axis, velocity_axis_filtered

def plot_technique_comparison(radar_data, chirp_params, frame_index=0, max_range=30, max_velocity=20, dynamic_range=40):
    """
    Compare différentes techniques de traitement sur un même frame de données radar.
    
    Args:
        radar_data: Données radar
        chirp_params: Paramètres du chirp FMCW
        frame_index: Index du frame à utiliser
        max_range: Distance maximale à afficher
        max_velocity: Vitesse maximale à afficher
        dynamic_range: Plage dynamique en dB
    """
    # Sélection d'un frame spécifique pour la comparaison
    frame_data = radar_data[frame_index]
    
    # Configurations à tester
    configs = [
        # Configuration complète standard (référence)
        {"title": "Standard (avec fenêtrage, FFT et padding)", 
         "params": {"use_range_fft": True, "use_doppler_fft": True, 
                   "range_padding": 4, "doppler_padding": 4, 
                   "use_windowing": True, "filter_method": 'none'}},
        
        # Sans zero-padding
        {"title": "Sans zero padding", 
         "params": {"use_range_fft": True, "use_doppler_fft": True, 
                   "range_padding": 1, "doppler_padding": 1, 
                   "use_windowing": True, "filter_method": 'none'}},
        
        # Sans fenêtrage
        {"title": "Sans fenêtrage", 
         "params": {"use_range_fft": True, "use_doppler_fft": True, 
                   "range_padding": 4, "doppler_padding": 4, 
                   "use_windowing": False, "filter_method": 'none'}},
        
        # Sans FFT de distance
        {"title": "Sans FFT de distance", 
         "params": {"use_range_fft": False, "use_doppler_fft": True, 
                   "range_padding": 1, "doppler_padding": 4, 
                   "use_windowing": True, "filter_method": 'none'}},
        
        # Sans FFT Doppler
        {"title": "Sans FFT Doppler", 
         "params": {"use_range_fft": True, "use_doppler_fft": False, 
                   "range_padding": 4, "doppler_padding": 1, 
                   "use_windowing": True, "filter_method": 'none'}},
        
        # Sans aucune FFT
        {"title": "Sans aucune FFT", 
         "params": {"use_range_fft": False, "use_doppler_fft": False, 
                   "range_padding": 1, "doppler_padding": 1, 
                   "use_windowing": True, "filter_method": 'none'}}
    ]
    
    # Création des sous-figures
    n_configs = len(configs)
    nrows = (n_configs + 1) // 2
    ncols = min(2, n_configs)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = axes.flatten()
    
    vmin_global = None
    vmax_global = None
    all_maps = []
    
    # Génération de toutes les cartes pour déterminer la plage de valeurs globale
    for config in configs:
        rd_map, range_axis, velocity_axis = create_range_doppler_map_configurable(
            frame_data, 
            chirp_params,
            **config["params"],
            normalize=False
        )
        
        all_maps.append((rd_map, range_axis, velocity_axis))
        
        curr_max = np.max(rd_map)
        if vmax_global is None or curr_max > vmax_global:
            vmax_global = curr_max
    
    # Définir les limites de couleur pour toutes les cartes
    vmin_global = vmax_global - dynamic_range
    
    # Affichage de chaque carte
    for i, ((rd_map, range_axis, velocity_axis), config) in enumerate(zip(all_maps, configs)):
        ax = axes[i]
        
        # Transposition de la carte pour l'affichage
        transposed_map = rd_map.T
        
        # Déterminer l'étendue des axes
        if config["params"]["use_doppler_fft"]:
            if max_velocity is not None and np.max(velocity_axis) > max_velocity:
                vel_extent = [-max_velocity, max_velocity]
            else:
                vel_extent = [velocity_axis[0], velocity_axis[-1]]
        else:
            vel_extent = [velocity_axis[0], velocity_axis[-1]]
        
        if config["params"]["use_range_fft"]:
            if max_range is not None and np.max(range_axis) > max_range:
                range_extent = [0, max_range]
            else:
                range_extent = [0, range_axis[-1]]
        else:
            range_extent = [0, range_axis[-1]]
        
        extent = [vel_extent[0], vel_extent[1], range_extent[0], range_extent[1]]
        
        im = ax.imshow(transposed_map, aspect='auto', origin='lower', 
                      extent=extent,
                      cmap='jet', vmin=vmin_global, vmax=vmax_global,
                      interpolation='bilinear')
        
        # Libellés des axes adaptés selon les configurations
        if config["params"]["use_doppler_fft"]:
            ax.set_xlabel('Vitesse (m/s)')
        else:
            ax.set_xlabel('Indice Doppler')
            
        if i % ncols == 0:
            if config["params"]["use_range_fft"]:
                ax.set_ylabel('Distance (m)')
            else:
                ax.set_ylabel('Indice de distance')
                
        ax.set_title(config["title"])
        ax.grid(True, color='w', linestyle='-', alpha=0.2)
    
    # Masquer les axes supplémentaires
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)
    
    # Barre de couleur
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Magnitude (dB)')
    
    plt.suptitle(f"Comparaison des techniques de traitement - Frame {frame_index}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f"images/Comparaison_techniques_frame{frame_index}.pdf")
    plt.show()

def compare_zero_padding_levels(radar_data, chirp_params, frame_index=0, padding_levels=[1, 2, 4, 8], max_range=30, max_velocity=20, dynamic_range=40):
    """
    Compare différents niveaux de zero-padding sur un même frame.
    
    Args:
        radar_data: Données radar
        chirp_params: Paramètres du chirp FMCW
        frame_index: Index du frame à utiliser
        padding_levels: Liste des niveaux de padding à tester
        max_range: Distance maximale à afficher
        max_velocity: Vitesse maximale à afficher
        dynamic_range: Plage dynamique en dB
    """
    frame_data = radar_data[frame_index]
    
    # Configuration des sous-figures
    n_levels = len(padding_levels)
    nrows = (n_levels + 1) // 2
    ncols = min(2, n_levels)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if n_levels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    vmin_global = None
    vmax_global = None
    all_maps = []
    
    # Générer toutes les cartes
    for padding in padding_levels:
        rd_map, range_axis, velocity_axis = create_range_doppler_map_configurable(
            frame_data, 
            chirp_params,
            range_padding=padding,
            doppler_padding=padding,
            use_windowing=True,
            filter_method='none',
            normalize=False
        )
        
        all_maps.append((rd_map, range_axis, velocity_axis))
        
        curr_max = np.max(rd_map)
        if vmax_global is None or curr_max > vmax_global:
            vmax_global = curr_max
    
    # Définir les limites de couleur
    vmin_global = vmax_global - dynamic_range
    
    # Afficher chaque carte
    for i, ((rd_map, range_axis, velocity_axis), padding) in enumerate(zip(all_maps, padding_levels)):
        ax = axes[i]
        
        transposed_map = rd_map.T
        
        # Déterminer l'étendue
        if max_velocity is not None:
            vel_extent = [-max_velocity, max_velocity]
        else:
            vel_extent = [velocity_axis[0], velocity_axis[-1]]
        
        if max_range is not None:
            range_extent = [0, max_range]
        else:
            range_extent = [0, range_axis[-1]]
        
        extent = [vel_extent[0], vel_extent[1], range_extent[0], range_extent[1]]
        
        im = ax.imshow(transposed_map, aspect='auto', origin='lower', 
                      extent=extent,
                      cmap='jet', vmin=vmin_global, vmax=vmax_global,
                      interpolation='bilinear')
        
        ax.set_xlabel('Vitesse (m/s)')
        if i % ncols == 0:
            ax.set_ylabel('Distance (m)')
        
        ax.set_title(f"Padding x{padding}")
        ax.grid(True, color='w', linestyle='-', alpha=0.2)

    # Masquer les axes supplémentaires
    for i in range(n_levels, len(axes)):
        axes[i].set_visible(False)
    
    # Barre de couleur
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Magnitude (dB)')
    
    plt.suptitle(f"Impact du niveau de zero-padding - Frame {frame_index}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f"images/Comparaison_padding_frame{frame_index}.pdf")
    plt.show()

def compare_window_functions(radar_data, chirp_params, frame_index=0, window_types=['none', 'hann', 'hamming', 'blackman'], max_range=30, max_velocity=20, dynamic_range=40):
    """
    Compare différentes fonctions de fenêtrage sur un même frame.
    
    Args:
        radar_data: Données radar
        chirp_params: Paramètres du chirp FMCW
        frame_index: Index du frame à utiliser
        window_types: Liste des types de fenêtres à tester
        max_range: Distance maximale à afficher
        max_velocity: Vitesse maximale à afficher
        dynamic_range: Plage dynamique en dB
    """
    frame_data = radar_data[frame_index]
    f0, B, Ms, Mc, Ts, Tc = chirp_params
    Ms = int(Ms)
    Mc = int(Mc)
    
    # Configuration des sous-figures
    n_windows = len(window_types)
    nrows = (n_windows + 1) // 2
    ncols = min(2, n_windows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if n_windows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    vmin_global = None
    vmax_global = None
    all_maps = []
    
    # Générer toutes les cartes
    for window_type in window_types:
        I1 = frame_data[0, :]
        Q1 = frame_data[1, :]
        complex_signal = I1 + 1j * Q1
        
        # Redimensionner le signal
        actual_size = len(complex_signal)
        if actual_size != Mc * Ms:
            total_samples = len(complex_signal)
            if total_samples % Mc == 0:
                Ms = total_samples // Mc
            else:
                Ms = total_samples // Mc
                complex_signal = complex_signal[:Ms * Mc]
        
        signal_matrix = np.reshape(complex_signal, (Mc, Ms))
        
        # Appliquer le fenêtrage selon le type choisi
        if window_type != 'none':
            if window_type == 'hann':
                range_window = windows.hann(Ms)
                doppler_window = windows.hann(Mc)
            elif window_type == 'hamming':
                range_window = windows.hamming(Ms)
                doppler_window = windows.hamming(Mc)
            elif window_type == 'blackman':
                range_window = windows.blackman(Ms)
                doppler_window = windows.blackman(Mc)
            
            signal_matrix = signal_matrix * doppler_window[:, np.newaxis]
            signal_matrix = signal_matrix * range_window[np.newaxis, :]
        
        # Appliquer les FFT avec padding
        range_padding = 4
        doppler_padding = 4
        n_range = Ms * range_padding
        n_doppler = Mc * doppler_padding
        
        range_fft = np.fft.fft(signal_matrix, n=n_range, axis=1)
        range_doppler_map = np.fft.fftshift(np.fft.fft(range_fft, n=n_doppler, axis=0), axes=0)
        
        # Calcul de l'amplitude
        range_doppler_map_db = 20 * np.log10(np.abs(range_doppler_map) + 1e-10)
        
        # Calcul des axes
        c = 3e8
        lambda_wave = c / f0
        
        range_res = c / (2 * B)
        max_range_val = range_res * n_range / 2
        range_axis = np.linspace(0, max_range_val, n_range)
        
        doppler_res = lambda_wave / (2 * Tc * Mc)
        max_velocity_val = doppler_res * n_doppler / 2
        velocity_axis = np.linspace(-max_velocity_val, max_velocity_val, n_doppler)
        
        all_maps.append((range_doppler_map_db, range_axis, velocity_axis))
        
        curr_max = np.max(range_doppler_map_db)
        if vmin_global is None or curr_max > vmax_global:
            vmax_global = curr_max
    
    # Définir les limites de couleur
    vmin_global = vmax_global - dynamic_range
    
    # Afficher chaque carte
    for i, ((rd_map, range_axis, velocity_axis), window_type) in enumerate(zip(all_maps, window_types)):
        ax = axes[i]
        
        transposed_map = rd_map.T
        
        # Déterminer l'étendue
        if max_velocity is not None:
            vel_extent = [-max_velocity, max_velocity]
        else:
            vel_extent = [velocity_axis[0], velocity_axis[-1]]
        
        if max_range is not None:
            range_extent = [0, max_range]
        else:
            range_extent = [0, range_axis[-1]]
        
        extent = [vel_extent[0], vel_extent[1], range_extent[0], range_extent[1]]
        
        im = ax.imshow(transposed_map, aspect='auto', origin='lower', 
                      extent=extent,
                      cmap='jet', vmin=vmin_global, vmax=vmax_global,
                      interpolation='bilinear')
        
        ax.set_xlabel('Vitesse (m/s)')
        if i % ncols == 0:
            ax.set_ylabel('Distance (m)')
        
        window_name = window_type.capitalize() if window_type != 'none' else 'Sans fenêtrage'
        ax.set_title(f"Fenêtre: {window_name}")
        ax.grid(True, color='w', linestyle='-', alpha=0.2)
    
    # Masquer les axes supplémentaires
    for i in range(n_windows, len(axes)):
        axes[i].set_visible(False)
    
    # Barre de couleur
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Magnitude (dB)')
    
    plt.suptitle(f"Impact du type de fenêtrage - Frame {frame_index}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f"images/Comparaison_fenetrage_frame{frame_index}.pdf")
    plt.show()

def compare_velocity_thresholds(radar_data, chirp_params, frame_index=0, 
                               velocity_thresholds=[0.0, 1.0, 2.0, 5.0, 10.0], 
                               filter_method='mask',
                               max_range=30, max_velocity=20, dynamic_range=40):
    """
    Compare l'effet de différents seuils de vitesse sur un même frame de données radar.
    
    Args:
        radar_data: Données radar
        chirp_params: Paramètres du chirp FMCW
        frame_index: Index du frame à utiliser
        velocity_thresholds: Liste des seuils de vitesse à tester (m/s)
        filter_method: Méthode de filtrage ('mask' ou 'remove')
        max_range: Distance maximale à afficher (m)
        max_velocity: Vitesse maximale à afficher (m/s)
        dynamic_range: Plage dynamique en dB
    """
    frame_data = radar_data[frame_index]
    
    # Configuration des sous-figures
    n_thresholds = len(velocity_thresholds)
    nrows = (n_thresholds + 1) // 2
    ncols = min(2, n_thresholds)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if n_thresholds == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    vmin_global = None
    vmax_global = None
    all_maps = []
    
    # Générer toutes les cartes pour chaque seuil de vitesse
    for threshold in velocity_thresholds:
        rd_map, range_axis, velocity_axis = create_range_doppler_map_configurable(
            frame_data, 
            chirp_params,
            range_padding=4,
            doppler_padding=4,
            use_windowing=True,
            velocity_threshold=threshold,
            filter_method=filter_method,
            normalize=False
        )
        
        all_maps.append((rd_map, range_axis, velocity_axis))
        
        curr_max = np.max(rd_map)
        if vmax_global is None or curr_max > vmax_global:
            vmax_global = curr_max
    
    # Définir les limites de couleur pour une comparaison cohérente
    vmin_global = vmax_global - dynamic_range
    
    # Afficher chaque carte avec son seuil de vitesse associé
    for i, ((rd_map, range_axis, velocity_axis), threshold) in enumerate(zip(all_maps, velocity_thresholds)):
        ax = axes[i]
        
        # Transposition de la carte pour l'affichage (distance en y, vitesse en x)
        transposed_map = rd_map.T
        
        # Déterminer l'étendue des axes
        if max_velocity is not None:
            vel_extent = [-max_velocity, max_velocity]
        else:
            vel_extent = [np.min(velocity_axis), np.max(velocity_axis)]
        
        if max_range is not None:
            range_extent = [0, max_range]
        else:
            range_extent = [0, np.max(range_axis)]
        
        extent = [vel_extent[0], vel_extent[1], range_extent[0], range_extent[1]]
        
        # Afficher la carte range-Doppler
        im = ax.imshow(transposed_map, aspect='auto', origin='lower', 
                      extent=extent,
                      cmap='jet', vmin=vmin_global, vmax=vmax_global,
                      interpolation='bilinear')
        
        # Ajouter les lignes verticales pour indiquer le seuil de vitesse
        if threshold > 0:
            ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.7)
            ax.axvline(x=-threshold, color='r', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Vitesse (m/s)')
        if i % ncols == 0:
            ax.set_ylabel('Distance (m)')
        
        ax.set_title(f"Seuil de vitesse: {threshold} m/s")
        ax.grid(True, color='w', linestyle='-', alpha=0.2)
    
    # Masquer les axes supplémentaires si nécessaire
    for i in range(n_thresholds, len(axes)):
        axes[i].set_visible(False)
    
    # Ajouter une barre de couleur
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Magnitude (dB)')
    
    plt.suptitle(f"Impact du seuil de vitesse (méthode: {filter_method}) - Frame {frame_index}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f"images/Comparaison_seuils_vitesse_frame{frame_index}.pdf")
    plt.show()

if __name__ == "__main__":
    file_path = 'MS1-FMCW.npz'
    
    try:
        data_dict = np.load(file_path, allow_pickle=True)
        
        radar_data = data_dict["data"]
        chirp_params = tuple(data_dict["chirp"])

        print(f"Fichier chargé avec succès! Dimensions des données: {radar_data.shape}")
        print(f"Paramètres du chirp: {chirp_params}")
        
        # Configuration des paramètres généraux
        max_range = 30               # Distance maximale à afficher (m)
        max_velocity = 20            # Vitesse maximale à afficher (m/s)
        dynamic_range = 40           # Plage dynamique en dB
        
        # 1. Comparer différentes techniques de traitement
        print("\nGénération de la comparaison des techniques de traitement...")
        plot_technique_comparison(
            radar_data, 
            chirp_params, 
            frame_index=0,
            max_range=max_range,
            max_velocity=max_velocity,
            dynamic_range=dynamic_range
        )
        
        # 2. Comparer différents niveaux de zero-padding
        print("\nGénération de la comparaison des niveaux de zero-padding...")
        compare_zero_padding_levels(
            radar_data, 
            chirp_params, 
            frame_index=0,
            padding_levels=[1, 2, 4, 8],
            max_range=max_range,
            max_velocity=max_velocity,
            dynamic_range=dynamic_range
        )
        
        # 3. Comparer différentes fonctions de fenêtrage
        print("\nGénération de la comparaison des fonctions de fenêtrage...")
        compare_window_functions(
            radar_data, 
            chirp_params, 
            frame_index=0,
            window_types=['none', 'hann', 'hamming', 'blackman'],
            max_range=max_range,
            max_velocity=max_velocity,
            dynamic_range=dynamic_range
        )

        # 4. Comparer différents seuils de vitesse
        print("\nGénération de la comparaison des seuils de vitesse...")
        compare_velocity_thresholds(
            radar_data, 
            chirp_params, 
            frame_index=0,
            velocity_thresholds=[0.0, 1.0, 2.0, 5.0, 10.0],
            filter_method='mask',  # Utiliser 'mask' pour masquer les vitesses faibles
            max_range=max_range,
            max_velocity=max_velocity,
            dynamic_range=dynamic_range
        )
        
        # Facultatif: tester également la méthode 'remove' qui supprime les vitesses faibles
        compare_velocity_thresholds(
            radar_data, 
            chirp_params, 
            frame_index=0,
            velocity_thresholds=[0.0, 1.0, 2.0, 5.0, 10.0],
            filter_method='remove',  # Utiliser 'remove' pour supprimer les vitesses faibles
            max_range=max_range,
            max_velocity=max_velocity,
            dynamic_range=dynamic_range
        )
        
        print(f"\nToutes les comparaisons ont été générées avec succès!")
        
    except Exception as e:
        import traceback
        print(f"Erreur lors du chargement ou du traitement du fichier: {e}")
        traceback.print_exc()